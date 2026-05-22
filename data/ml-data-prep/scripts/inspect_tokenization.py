#!/usr/bin/env python3
"""
inspect_tokenization.py — surface the SILENT tokenization/packing bugs that
don't raise errors but quietly wreck SFT quality:

  1. chat template — what the model actually sees after apply_chat_template
  2. BOS/EOS — missing EOS (model never learns to stop) or double-BOS
  3. label mask — are prompt tokens masked (-100) so loss is on the completion only?
  4. packing — concatenated samples need a block-diagonal attention mask +
     per-document position_ids, or tokens attend ACROSS documents (cross-
     contamination), silently degrading the model.

Needs `transformers` + a tokenizer (--tokenizer). If unavailable, prints the
manual checklist and points to references/tokenize-packing.md — never crashes.

Usage:
    python3 inspect_tokenization.py --tokenizer /path/to/model --input sample.jsonl
    python3 inspect_tokenization.py --tokenizer Qwen/Qwen2.5-7B-Instruct --demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_format import detect_format, PARSERS, read_records  # noqa: E402

CHECKLIST = """\
tokenization / packing checklist (verify each — none of these raise an error):
  [ ] chat template applied with the MODEL'S OWN template (tokenizer.apply_chat_template);
      training and inference must use the same template.
  [ ] EOS token ends every assistant turn — without it the model never learns to stop.
  [ ] no double-BOS — template adds BOS *and* tokenizer(add_special_tokens=True) adds
      another. Tokenize the already-templated string with add_special_tokens=False.
  [ ] label mask: prompt/user tokens set to -100, loss computed on the assistant
      completion only (unless you deliberately train on inputs).
  [ ] padding side: right for training, left for batched generation. Or use packing.
  [ ] packing: block-diagonal attention (samples must NOT attend across the join) +
      position_ids reset per document. Naive concatenation contaminates silently.
  [ ] truncation cuts the PROMPT side, never the answer.
See references/tokenize-packing.md for the why and the fixes."""


def to_messages(rec, fmt):
    ir = PARSERS[fmt](rec, lambda *_: None)
    msgs = []
    if ir["system"]:
        msgs.append({"role": "system", "content": ir["system"]})
    msgs += [{"role": t["role"], "content": t["content"]} for t in ir["turns"]]
    return msgs


DEMO_MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect tokenization & packing correctness.")
    p.add_argument("--tokenizer", help="HF tokenizer path/name")
    p.add_argument("--input", help="dataset file (uses first record)")
    p.add_argument("--format", choices=["auto", "alpaca", "sharegpt", "openai"], default="auto")
    p.add_argument("--demo", action="store_true", help="use a built-in sample")
    p.add_argument("--max-len", type=int, default=4096)
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        print(CHECKLIST)
        print("\n# `transformers` not importable — install it + pass --tokenizer "
              "to run the live inspection.", file=sys.stderr)
        return 0
    if not args.tokenizer:
        print(CHECKLIST)
        print("\n# pass --tokenizer <path/name> to run the live inspection.", file=sys.stderr)
        return 0

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.demo or not args.input:
        msgs = DEMO_MSGS
    else:
        recs = read_records(args.input)
        fmt = detect_format(recs[0]) if args.format == "auto" else args.format
        msgs = to_messages(recs[0], fmt)

    report = {"tokenizer": args.tokenizer,
              "special_tokens": {"bos": tok.bos_token, "eos": tok.eos_token,
                                 "pad": tok.pad_token, "pad_side": tok.padding_side},
              "findings": []}

    def find(level, code, msg):
        report["findings"].append({"level": level, "code": code, "message": msg})

    # 1. render chat template
    has_template = tok.chat_template is not None
    if not has_template:
        find("warn", "no-chat-template",
             "tokenizer has no chat_template — you must format prompts manually and "
             "keep it identical at train & inference.")
    rendered = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False) if has_template else \
        "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    report["rendered_preview"] = rendered[:400]
    report["n_tokens"] = len(ids)

    # 2. BOS/EOS
    bos_id, eos_id = tok.bos_token_id, tok.eos_token_id
    if bos_id is not None and tok.bos_token:
        template_emits_bos = rendered.startswith(tok.bos_token)
        tokenizer_prepends_bos = tok("x", add_special_tokens=True)["input_ids"][:1] == [bos_id]
        if template_emits_bos and tokenizer_prepends_bos:
            find("warn", "double-bos-risk",
                 "template already emits BOS and the tokenizer would prepend another with "
                 "add_special_tokens=True → double BOS. Tokenize already-templated text with "
                 "add_special_tokens=False.")
        # also catch an actual double-BOS in the produced ids
        if ids[:2] == [bos_id, bos_id]:
            find("warn", "double-bos-found", "two BOS tokens at the start of the sequence.")
    if eos_id is not None:
        if ids and ids[-1] != eos_id and (tok.eos_token not in rendered[-len(tok.eos_token)-20:]):
            find("warn", "missing-eos",
                 "rendered assistant turn doesn't end with EOS — the model may never learn "
                 "to stop. Ensure the template appends EOS after the assistant content.")
        else:
            find("info", "eos-ok", "EOS present at/near the end of the assistant turn.")

    # 3. label mask (needs a template with {% generation %} support)
    try:
        out = tok.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True,
                                      add_generation_prompt=False)
        mask = out.get("assistant_masks") or out.get("assistant_tokens_mask")
        if mask is not None and sum(mask) > 0:
            trainable = sum(mask)
            total = len(mask)
            find("info", "label-mask",
                 f"assistant(trainable) tokens = {trainable}/{total} "
                 f"({round(100*trainable/max(total,1))}%); prompt tokens should be -100. "
                 "Use this mask (or DataCollatorForCompletionOnlyLM) so loss is on the "
                 "completion only.")
        else:
            raise ValueError("no assistant region marked")
    except Exception:
        find("warn", "label-mask-unknown",
             "this template does NOT mark an assistant token region (no {% generation %} "
             "block), so transformers can't auto-build the loss mask. You MUST mask prompt "
             "tokens to -100 yourself (DataCollatorForCompletionOnlyLM / train_on_inputs=false), "
             "else the model trains on the prompt too.")

    # 4. packing demo — show position_ids AROUND the document boundary
    b = len(ids)
    pack_n = max(2, (args.max_len // max(b, 1)))
    correct_pos = list(range(b)) + list(range(b))   # reset per doc
    naive_pos = list(range(2 * b))                  # continuous (wrong)
    lo, hi = max(b - 2, 0), b + 3
    find("warn", "packing-attention",
         f"if you pack ~{pack_n} samples into one {args.max_len}-token block: a NAIVE concat "
         "lets later samples attend to earlier ones (cross-contamination, no error raised). "
         "Require a block-diagonal mask (FlashAttention varlen) + position_ids reset per doc. "
         f"At the join (doc1 end | doc2 start) position_ids should be "
         f"correct={correct_pos[lo:hi]} (resets to 0), NOT naive={naive_pos[lo:hi]} (keeps counting).")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
