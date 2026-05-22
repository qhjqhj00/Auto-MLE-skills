#!/usr/bin/env python3
"""
check_quality.py — scan an SFT/chat dataset for the issues that silently hurt
training: empty/duplicate/malformed samples, length & truncation risk, and
label-alignment problems.

Emits a JSON report with tiered FLAGS (error/warn/info) and an overall verdict
✅ clean / ⚠️ review / ❌ will-harm-training — same convention as ml-env-probe.

Pure stdlib for everything except token-length, which uses a HF tokenizer ONLY
if you pass --tokenizer and `transformers` is importable; otherwise it falls
back to character length and says so.

Usage:
    python3 check_quality.py --input data.jsonl --max-len 4096
    python3 check_quality.py --input data.json --tokenizer /path/to/model --max-len 8192
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

# reuse the canonical-IR parser/detector from the sibling converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_format import detect_format, PARSERS, read_records  # noqa: E402

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return round(values[f] + (values[c] - values[f]) * (k - f), 1)


def load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"# tokenizer load failed ({e}); falling back to char length",
              file=sys.stderr)
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Scan a chat/SFT dataset for quality issues.")
    p.add_argument("--input", required=True)
    p.add_argument("--format", choices=["auto", "alpaca", "sharegpt", "openai"],
                   default="auto")
    p.add_argument("--max-len", type=int, help="max sequence length you'll train at")
    p.add_argument("--tokenizer", help="HF tokenizer path/name for true token length")
    p.add_argument("--dup-warn", type=float, default=0.01, help="exact-dup ratio to warn")
    p.add_argument("--trunc-warn", type=float, default=0.05)
    p.add_argument("--trunc-error", type=float, default=0.20)
    args = p.parse_args()

    records = read_records(args.input)
    n = len(records)
    tok = load_tokenizer(args.tokenizer)

    # accumulators
    fmts: dict[str, int] = {}
    no_assistant = empty_assistant = empty_user = 0
    not_end_assistant = non_alternating = 0
    char_lens, tok_lens, completion_chars = [], [], []
    exact_seen, norm_seen = {}, {}
    exact_dups = norm_dups = 0
    over_len = answer_at_risk = 0
    parse_errors = 0

    def text_of(ir):
        parts = []
        if ir["system"]:
            parts.append(ir["system"])
        parts += [t["content"] for t in ir["turns"]]
        return "\n".join(parts)

    for rec in records:
        try:
            fmt = detect_format(rec) if args.format == "auto" else args.format
            ir = PARSERS[fmt](rec, lambda *_: None)
        except Exception:
            parse_errors += 1
            continue
        fmts[fmt] = fmts.get(fmt, 0) + 1
        turns = ir["turns"]
        roles = [t["role"] for t in turns]

        if "assistant" not in roles:
            no_assistant += 1
        else:
            if roles[-1] != "assistant":
                not_end_assistant += 1
        if any(t["role"] == "assistant" and not t["content"].strip() for t in turns):
            empty_assistant += 1
        if any(t["role"] == "user" and not t["content"].strip() for t in turns):
            empty_user += 1
        # alternating: no two consecutive same role among user/assistant
        for a, b in zip(roles, roles[1:]):
            if a == b:
                non_alternating += 1
                break

        full = text_of(ir)
        clen = len(full)
        char_lens.append(clen)
        completion_chars.append(sum(len(t["content"]) for t in turns
                                    if t["role"] == "assistant"))
        if tok is not None:
            try:
                tlen = len(tok(full, add_special_tokens=True)["input_ids"])
            except Exception:
                tlen = clen // 4
            tok_lens.append(tlen)
            length_for_trunc = tlen
        else:
            length_for_trunc = clen

        # duplicates
        ex_key = full
        exact_dups += 1 if ex_key in exact_seen else 0
        exact_seen[ex_key] = 1
        nm_key = _norm(full)
        norm_dups += 1 if nm_key in norm_seen else 0
        norm_seen[nm_key] = 1

        # truncation risk
        if args.max_len:
            unit_max = args.max_len if tok is not None else args.max_len * 4  # char approx
            if length_for_trunc > unit_max:
                over_len += 1
                # answer-at-risk: assistant content sits in the tail that gets cut
                if turns and turns[-1]["role"] == "assistant":
                    answer_at_risk += 1

    # ---- build flags ----
    flags = []

    def add(level, code, msg):
        flags.append({"level": level, "code": code, "message": msg})

    valid = n - parse_errors
    if parse_errors:
        add("error", "parse-errors",
            f"{parse_errors}/{n} records failed to parse/detect format.")
    if len(fmts) > 1:
        add("warn", "mixed-formats",
            f"records span multiple formats {fmts}; convert to one before training.")
    if no_assistant:
        add("error", "no-assistant",
            f"{no_assistant} samples have NO assistant turn — nothing to learn from; drop them.")
    if empty_assistant:
        add("error", "empty-assistant",
            f"{empty_assistant} samples have an empty assistant turn — teaches the model "
            "to output nothing; drop or fix.")
    if empty_user:
        add("warn", "empty-user", f"{empty_user} samples have an empty user turn.")
    if valid and exact_dups / valid >= args.dup_warn:
        add("warn", "exact-duplicates",
            f"{exact_dups} exact-duplicate samples ({round(100*exact_dups/valid,1)}%) — "
            "inflates those examples / memorization risk; dedup.")
    elif exact_dups:
        add("info", "exact-duplicates", f"{exact_dups} exact duplicates (below warn threshold).")
    if norm_dups > exact_dups:
        add("info", "near-duplicates",
            f"{norm_dups} normalized-duplicate samples (whitespace/case) — {norm_dups-exact_dups} "
            "beyond exact dups; consider near-dedup.")
    if not_end_assistant:
        add("warn", "not-ending-on-assistant",
            f"{not_end_assistant} conversations don't end on an assistant turn — for SFT the "
            "last turn should be the assistant response.")
    if non_alternating:
        add("warn", "non-alternating-roles",
            f"{non_alternating} samples have consecutive same-role turns — most chat templates "
            "assume strict user/assistant alternation; verify.")
    if args.max_len and valid:
        ratio = over_len / valid
        unit = "tokens" if tok is not None else "≈chars/4 (no tokenizer)"
        lvl = ("error" if ratio >= args.trunc_error else
               "warn" if ratio >= args.trunc_warn else "info")
        add(lvl, "truncation",
            f"{over_len} samples ({round(100*ratio,1)}%) exceed max_len={args.max_len} [{unit}]; "
            f"{answer_at_risk} of them end on the assistant turn → its answer gets cut "
            "(model learns not to finish). Raise max_len, truncate the prompt side, or filter.")
    if not flags:
        add("info", "ok", "No quality issues detected above thresholds.")

    level_rank = {"error": 2, "warn": 1, "info": 0}
    worst = max((level_rank[f["level"]] for f in flags), default=0)
    verdict = {2: "❌ will-harm-training", 1: "⚠️ review", 0: "✅ clean"}[worst]

    report = {
        "schema": "ml-data-prep/quality/1",
        "input": args.input,
        "n_records": n,
        "valid_records": valid,
        "formats": fmts,
        "length": {
            "unit": "tokens" if tok is not None else "chars",
            "p50": _pct(tok_lens or char_lens, 0.5),
            "p90": _pct(tok_lens or char_lens, 0.9),
            "p99": _pct(tok_lens or char_lens, 0.99),
            "max": max(tok_lens or char_lens) if (tok_lens or char_lens) else None,
            "mean": round(statistics.fmean(tok_lens or char_lens), 1) if (tok_lens or char_lens) else None,
        },
        "issues": {
            "no_assistant": no_assistant, "empty_assistant": empty_assistant,
            "empty_user": empty_user, "exact_duplicates": exact_dups,
            "normalized_duplicates": norm_dups, "not_ending_on_assistant": not_end_assistant,
            "non_alternating": non_alternating,
            "over_max_len": over_len, "answer_at_risk": answer_at_risk,
        },
        "flags": flags,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
