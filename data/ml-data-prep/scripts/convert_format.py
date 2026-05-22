#!/usr/bin/env python3
"""
convert_format.py — convert SFT/chat datasets between alpaca, sharegpt, and
openai-messages formats, via a single canonical intermediate representation.

Why an IR: pairwise converters (6 directions) drift and disagree on edge cases
(role names, multi-turn, system prompts). Parse ANY format -> canonical IR
-> emit ANY format. One place to get the nuances right.

Canonical IR (per example):
    {"system": str|None, "turns": [{"role": "user"|"assistant", "content": str}]}

Pure Python stdlib. Reads/writes JSON array or JSONL (auto-detected on read).

Usage:
    python3 convert_format.py --input data.json --to openai --out data.jsonl
    python3 convert_format.py --input sg.jsonl --from sharegpt --to alpaca --out a.json
    python3 convert_format.py --input data.jsonl --to openai --stats   # report only
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

# role-name normalization across dialects
USER_ROLES = {"human", "user", "prompter"}
ASSISTANT_ROLES = {"gpt", "assistant", "bot", "model"}
SYSTEM_ROLES = {"system"}
# roles we don't model in v1 (tool/function calling) — preserved as a warning
OTHER_ROLES = {"function_call", "observation", "tool", "tool_call", "function"}


# --------------------------------------------------------------------------- #
# read / write (JSON array or JSONL)
# --------------------------------------------------------------------------- #


def read_records(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    # JSONL
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"JSONL parse error at line {i}: {e}")
    return out


def write_records(path: Optional[str], records: list[dict[str, Any]], jsonl: bool) -> None:
    if jsonl:
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    else:
        text = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


# --------------------------------------------------------------------------- #
# detect format
# --------------------------------------------------------------------------- #


def detect_format(rec: dict[str, Any]) -> str:
    if "messages" in rec and isinstance(rec["messages"], list):
        return "openai"
    if "conversations" in rec and isinstance(rec["conversations"], list):
        return "sharegpt"
    if "instruction" in rec or ("output" in rec and "input" in rec):
        return "alpaca"
    raise ValueError(f"cannot detect format of record with keys {list(rec.keys())}")


# --------------------------------------------------------------------------- #
# parse <format> -> canonical IR
# --------------------------------------------------------------------------- #


def _norm_role(role: str) -> Optional[str]:
    r = (role or "").strip().lower()
    if r in USER_ROLES:
        return "user"
    if r in ASSISTANT_ROLES:
        return "assistant"
    if r in SYSTEM_ROLES:
        return "system"
    if r in OTHER_ROLES:
        return "other"
    return None


def parse_alpaca(rec: dict[str, Any], warn) -> dict[str, Any]:
    system = rec.get("system") or None
    turns: list[dict[str, str]] = []
    # LLaMA-Factory extension: history = [[user, assistant], ...]
    for pair in rec.get("history") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            turns.append({"role": "user", "content": str(pair[0])})
            turns.append({"role": "assistant", "content": str(pair[1])})
    instr = (rec.get("instruction") or "").strip()
    inp = (rec.get("input") or "").strip()
    user_content = instr if not inp else f"{instr}\n\n{inp}" if instr else inp
    if user_content:
        turns.append({"role": "user", "content": user_content})
    if rec.get("output") is not None:
        turns.append({"role": "assistant", "content": str(rec["output"])})
    return {"system": system, "turns": turns}


def parse_sharegpt(rec: dict[str, Any], warn) -> dict[str, Any]:
    system = rec.get("system") or None
    turns: list[dict[str, str]] = []
    conv_key = "conversations" if "conversations" in rec else "conversation"
    for msg in rec.get(conv_key, []):
        role = _norm_role(msg.get("from", ""))
        val = msg.get("value", msg.get("content", ""))
        if role == "system":
            system = val if not system else system
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": str(val)})
        elif role == "other":
            warn("tool/function turns present (sharegpt) — dropped in v1 IR")
    return {"system": system, "turns": turns}


def parse_openai(rec: dict[str, Any], warn) -> dict[str, Any]:
    system = None
    turns: list[dict[str, str]] = []
    for msg in rec.get("messages", []):
        role = _norm_role(msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):  # multimodal/parts -> join text parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            system = content if not system else system
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": str(content)})
        elif role == "other" or msg.get("tool_calls"):
            warn("tool/function-call messages present (openai) — dropped in v1 IR")
    return {"system": system, "turns": turns}


PARSERS = {"alpaca": parse_alpaca, "sharegpt": parse_sharegpt, "openai": parse_openai}


# --------------------------------------------------------------------------- #
# emit canonical IR -> <format>
# --------------------------------------------------------------------------- #


def emit_openai(ir: dict[str, Any]) -> dict[str, Any]:
    msgs = []
    if ir["system"]:
        msgs.append({"role": "system", "content": ir["system"]})
    msgs.extend({"role": t["role"], "content": t["content"]} for t in ir["turns"])
    return {"messages": msgs}


def emit_sharegpt(ir: dict[str, Any]) -> dict[str, Any]:
    convs = [{"from": ("human" if t["role"] == "user" else "gpt"),
              "value": t["content"]} for t in ir["turns"]]
    out: dict[str, Any] = {"conversations": convs}
    if ir["system"]:
        out["system"] = ir["system"]
    return out


def emit_alpaca(ir: dict[str, Any], warn) -> dict[str, Any]:
    turns = ir["turns"]
    # split into (user, assistant) pairs
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i]["content"], turns[i + 1]["content"]))
            i += 2
        else:
            i += 1
    rec: dict[str, Any] = {"instruction": "", "input": "", "output": ""}
    if ir["system"]:
        rec["system"] = ir["system"]
    if not pairs:
        warn("no clean user/assistant pair — alpaca record may be empty")
        return rec
    # last pair = the trainable instruction/output; earlier = history
    if len(pairs) > 1:
        rec["history"] = [list(p) for p in pairs[:-1]]
        warn("multi-turn -> alpaca uses LLaMA-Factory 'history' field (non-standard)")
    rec["instruction"], rec["output"] = pairs[-1][0], pairs[-1][1]
    return rec


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description="Convert SFT/chat datasets between formats.")
    p.add_argument("--input", required=True)
    p.add_argument("--from", dest="src", choices=["auto", "alpaca", "sharegpt", "openai"],
                   default="auto")
    p.add_argument("--to", choices=["alpaca", "sharegpt", "openai"], required=True)
    p.add_argument("--out", help="output path (default: stdout)")
    p.add_argument("--out-format", choices=["json", "jsonl"], default="jsonl")
    p.add_argument("--stats", action="store_true", help="report only, no conversion output")
    args = p.parse_args()

    records = read_records(args.input)
    warnings: dict[str, int] = {}

    def warn(msg: str) -> None:
        warnings[msg] = warnings.get(msg, 0) + 1

    src_fmt = args.src
    converted = []
    n_turns_total = 0
    multiturn = 0
    for rec in records:
        fmt = detect_format(rec) if src_fmt == "auto" else src_fmt
        ir = PARSERS[fmt](rec, warn)
        n_turns_total += len(ir["turns"])
        if sum(1 for t in ir["turns"] if t["role"] == "user") > 1:
            multiturn += 1
        if args.to == "openai":
            converted.append(emit_openai(ir))
        elif args.to == "sharegpt":
            converted.append(emit_sharegpt(ir))
        else:
            converted.append(emit_alpaca(ir, warn))

    report = {
        "input": args.input,
        "detected_format": (detect_format(records[0]) if records and src_fmt == "auto"
                            else src_fmt),
        "target_format": args.to,
        "n_records": len(records),
        "n_turns_total": n_turns_total,
        "n_multiturn_records": multiturn,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)

    if not args.stats:
        write_records(args.out, converted, jsonl=(args.out_format == "jsonl"))
        if args.out:
            print(f"# wrote {len(converted)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
