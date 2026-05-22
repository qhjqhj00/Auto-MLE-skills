#!/usr/bin/env python3
"""
estimate_vram.py — estimate GPU memory BEFORE you launch, so you don't OOM
3 hours into a run.

Given a model (HF config.json / model dir, or manual params) + a scenario
(inference|training, batch, seq, dtype, parallelism), it computes a memory
breakdown:  weights + KV-cache + (gradients + optimizer + activations) +
overhead, applies the parallelism split, and — if you pass available VRAM —
returns a fits / tight / OOM verdict with fallbacks.

Design (same as ml-env-probe):
  * Pure stdlib. Never crashes; missing inputs => documented assumptions.
  * FACTS + ARITHMETIC only. The script does the math; the SKILL + references
    own the heuristics, the safety margins, and the fallback advice.
  * Estimates are intentionally on the conservative (slightly-high) side and
    reported with a margin — the goal is "avoid OOM", not byte-exactness.

Key correctness points it gets right (where naive estimators fail):
  * HYBRID attention: only counts KV-cache for full-attention layers, reads
    `layer_types` (linear/SSM layers have constant state, not growing KV).
  * GQA: KV-cache uses num_key_value_heads, not num_attention_heads.
  * MoE: all expert weights live in VRAM (total params), but the weight term
    is taken from real on-disk size when a model dir is given (quant-accurate).
  * Quantization: weight bytes come from the actual stored files when present.

Usage:
    python3 estimate_vram.py --model /path/to/model_dir --mode inference \
        --batch 1 --seq 8192 --available-vram 120
    python3 estimate_vram.py --config config.json --mode training \
        --batch 4 --seq 4096 --weight-dtype bf16 --optimizer adamw --grad-ckpt
    python3 estimate_vram.py --num-params-b 7 --hidden 4096 --layers 32 \
        --kv-heads 8 --head-dim 128 --mode inference --batch 1 --seq 32768
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Optional

# bytes per element by dtype name
DTYPE_BYTES = {
    "fp32": 4, "float32": 4, "tf32": 4,
    "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
    "fp8": 1, "float8": 1, "int8": 1,
    "fp4": 0.5, "nvfp4": 0.5, "int4": 0.5, "nf4": 0.5,
}

# optimizer state bytes per parameter (mixed-precision training conventions)
#   adamw: fp32 master(4) + momentum(4) + variance(4) = 12
#   sgd-momentum: fp32 master(4) + momentum(4) = 8
#   adafactor: factored, ~4 (approx)
#   none / inference: 0
OPT_BYTES = {"adamw": 12, "adam": 12, "sgd": 8, "adafactor": 4, "none": 0}


def _gb(num_bytes: float) -> float:
    return round(num_bytes / (1024 ** 3), 2)


# --------------------------------------------------------------------------- #
# load architecture
# --------------------------------------------------------------------------- #


def load_arch(args: argparse.Namespace) -> dict[str, Any]:
    """Return a normalized arch dict from --model dir, --config, or manual flags."""
    cfg: dict[str, Any] = {}
    model_dir: Optional[str] = None
    cfg_path: Optional[str] = None

    if args.model:
        model_dir = args.model
        cfg_path = os.path.join(args.model, "config.json")
    elif args.config:
        cfg_path = args.config
        model_dir = os.path.dirname(os.path.abspath(args.config))

    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = json.load(f)
        # many models nest the LM under text_config / llm_config
        cfg = raw.get("text_config") or raw.get("llm_config") or raw
        cfg["_raw"] = raw

    def g(*names, default=None):
        for n in names:
            if cfg.get(n) is not None:
                return cfg[n]
        return default

    layers = args.layers or g("num_hidden_layers", "n_layer", default=None)
    hidden = args.hidden or g("hidden_size", "n_embd", "d_model", default=None)
    n_heads = g("num_attention_heads", "n_head", default=None)
    kv_heads = args.kv_heads or g("num_key_value_heads", "num_kv_heads", default=n_heads)
    head_dim = args.head_dim or g("head_dim", default=None)
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads
    vocab = g("vocab_size", default=None)

    # hybrid attention: count layers that actually keep a growing KV-cache.
    layer_types = g("layer_types", default=None)
    if isinstance(layer_types, list) and layer_types:
        full_attn_layers = sum(1 for t in layer_types if "full" in str(t).lower())
        # if naming is unknown, assume every listed layer keeps KV
        if full_attn_layers == 0 and not any("linear" in str(t).lower()
                                             or "mamba" in str(t).lower()
                                             or "ssm" in str(t).lower()
                                             for t in layer_types):
            full_attn_layers = len(layer_types)
    else:
        full_attn_layers = layers  # assume dense full-attention model

    moe = {
        "num_experts": g("num_experts", "n_routed_experts", default=None),
        "experts_per_tok": g("num_experts_per_tok", "moe_topk", default=None),
        "moe_intermediate": g("moe_intermediate_size", default=None),
    }

    # weight bytes: prefer REAL on-disk size (quant-accurate); else analytic.
    weight_bytes, weight_src = resolve_weight_bytes(args, model_dir, layers, hidden,
                                                    kv_heads, n_heads, head_dim,
                                                    vocab, cfg, moe)

    return {
        "name": os.path.basename(model_dir.rstrip("/")) if model_dir else "manual",
        "layers": layers,
        "hidden": hidden,
        "n_heads": n_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "full_attn_layers": full_attn_layers,
        "is_hybrid": isinstance(layer_types, list) and full_attn_layers != (layers or -1),
        "moe": moe if moe["num_experts"] else None,
        "weight_bytes": weight_bytes,
        "weight_source": weight_src,
    }


def resolve_weight_bytes(args, model_dir, layers, hidden, kv_heads, n_heads,
                         head_dim, vocab, cfg, moe) -> tuple[Optional[float], str]:
    # 1) explicit override
    if args.weights_gb:
        return args.weights_gb * (1024 ** 3), "manual --weights-gb"
    # 2) num params * chosen dtype
    if args.num_params_b:
        b = DTYPE_BYTES.get(args.weight_dtype, 2)
        return args.num_params_b * 1e9 * b, f"--num-params-b x {args.weight_dtype}"
    # 3) real on-disk safetensors / bin size (most accurate; quant-aware)
    if model_dir and os.path.isdir(model_dir):
        files = (glob.glob(os.path.join(model_dir, "*.safetensors"))
                 + glob.glob(os.path.join(model_dir, "*.bin")))
        total = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        if total > 0:
            # if user forces a different dtype than stored, we can't recompute
            # without param count, so report as-stored and note it.
            return total, f"on-disk weights ({len(files)} files)"
    # 4) analytic estimate of parameter count -> bytes
    p = analytic_param_count(layers, hidden, kv_heads, n_heads, head_dim, vocab, cfg, moe)
    if p:
        b = DTYPE_BYTES.get(args.weight_dtype, 2)
        return p * b, f"analytic ~{round(p/1e9,2)}B params x {args.weight_dtype} (APPROX)"
    return None, "unknown"


def analytic_param_count(layers, hidden, kv_heads, n_heads, head_dim, vocab,
                         cfg, moe) -> Optional[float]:
    if not (layers and hidden):
        return None
    n_heads = n_heads or 1
    head_dim = head_dim or (hidden // n_heads)
    kv_heads = kv_heads or n_heads
    # attention: q + o (hidden x n_heads*hd) + k + v (hidden x kv_heads*hd)
    attn = hidden * (n_heads * head_dim) * 2 + hidden * (kv_heads * head_dim) * 2
    # mlp: SwiGLU = 3 matrices hidden x intermediate
    if moe and moe.get("num_experts") and moe.get("moe_intermediate"):
        mlp = moe["num_experts"] * 3 * hidden * moe["moe_intermediate"]
    else:
        inter = cfg.get("intermediate_size") or 4 * hidden
        mlp = 3 * hidden * inter
    per_layer = attn + mlp
    total = per_layer * layers
    if vocab:
        tied = cfg.get("tie_word_embeddings", False)
        total += vocab * hidden * (1 if tied else 2)
    return total


# --------------------------------------------------------------------------- #
# memory terms
# --------------------------------------------------------------------------- #


def kv_cache_bytes(arch, batch, seq, kv_dtype) -> Optional[float]:
    """KV-cache for FULL-attention layers only (hybrid-aware)."""
    if not (arch["kv_heads"] and arch["head_dim"] and arch["full_attn_layers"]):
        return None
    b = DTYPE_BYTES.get(kv_dtype, 2)
    per_tok_per_layer = 2 * arch["kv_heads"] * arch["head_dim"] * b  # K and V
    return per_tok_per_layer * arch["full_attn_layers"] * batch * seq


def activation_bytes(arch, batch, seq, act_dtype, grad_ckpt) -> Optional[float]:
    """Training activation memory (approx). Dominant at large batch*seq.

    No-checkpoint approx (per layer): ~ batch*seq*hidden * factor, factor~16
    captures the several intermediate tensors of attn+MLP in mixed precision.
    Plus the attention score matrix batch*n_heads*seq^2 for full-attn layers.
    With gradient checkpointing: store only layer-boundary activations (~2x
    batch*seq*hidden per layer) and recompute the rest -> big reduction.
    """
    if not (arch["hidden"] and arch["layers"]):
        return None
    b = DTYPE_BYTES.get(act_dtype, 2)
    h, L = arch["hidden"], arch["layers"]
    bsh = batch * seq * h * b
    if grad_ckpt:
        per_layer = bsh * 2  # only checkpointed boundaries kept
        score = 0  # recomputed, not stored at peak (approx)
    else:
        per_layer = bsh * 16
        # attention score matrix grows with seq^2 (full-attn layers only)
        score_layers = arch["full_attn_layers"] or L
        nh = arch["n_heads"] or 1
        score = batch * nh * seq * seq * b * (score_layers / max(L, 1))
    return per_layer * L + score


def parallel_divide(term_bytes, kind, tp, pp, dp, zero) -> float:
    """Divide a memory term by the relevant parallel degree(s)."""
    if term_bytes is None:
        return 0.0
    out = term_bytes
    if kind in ("weights", "kv", "activations"):
        out /= max(tp, 1)            # tensor parallel shards these
    if kind in ("weights",):
        out /= max(pp, 1)            # pipeline parallel shards weights by layer
    if kind == "activations":
        # PP keeps ~all micro-batch activations in flight; treat as ~1/pp best case
        out /= max(pp, 1)
    # ZeRO / FSDP shard across the data-parallel group
    if zero >= 1 and kind == "optimizer":
        out /= max(dp, 1)
    if zero >= 2 and kind == "gradients":
        out /= max(dp, 1)
    if zero >= 3 and kind == "weights":
        out /= max(dp, 1)
    return out


# --------------------------------------------------------------------------- #
# estimate
# --------------------------------------------------------------------------- #


def estimate(args) -> dict[str, Any]:
    arch = load_arch(args)
    tp, pp, dp = max(args.tp, 1), max(args.pp, 1), max(args.dp, 1)
    zero = args.zero
    overhead_gb = args.overhead_gb

    terms_full: dict[str, Optional[float]] = {}   # before parallelism (model-total)
    terms_per_gpu: dict[str, float] = {}

    # weights
    w = arch["weight_bytes"]
    terms_full["weights"] = w
    terms_per_gpu["weights"] = parallel_divide(w, "weights", tp, pp, dp, zero)

    if args.mode == "inference":
        kv = kv_cache_bytes(arch, args.batch, args.seq, args.kv_dtype)
        terms_full["kv_cache"] = kv
        terms_per_gpu["kv_cache"] = parallel_divide(kv, "kv", tp, pp, dp, zero)
        # inference activations are small; approximate one-layer working set
        if arch["hidden"]:
            act = args.batch * args.seq * arch["hidden"] * DTYPE_BYTES.get(args.weight_dtype, 2) * 2
            terms_full["activations"] = act
            terms_per_gpu["activations"] = parallel_divide(act, "activations", tp, pp, dp, zero)
    else:  # training
        # gradients (same count as trainable params; bf16 grads = 2 B/param,
        # but we derive from weight *bytes* / weight-dtype-bytes to get param count)
        wb = DTYPE_BYTES.get(args.weight_dtype, 2)
        approx_params = (w / wb) if w else None
        grad = (approx_params * 2) if approx_params else None        # bf16 grads
        opt = (approx_params * OPT_BYTES.get(args.optimizer, 12)) if approx_params else None
        act = activation_bytes(arch, args.batch, args.seq, args.weight_dtype, args.grad_ckpt)
        terms_full["gradients"] = grad
        terms_full["optimizer"] = opt
        terms_full["activations"] = act
        terms_per_gpu["gradients"] = parallel_divide(grad, "gradients", tp, pp, dp, zero)
        terms_per_gpu["optimizer"] = parallel_divide(opt, "optimizer", tp, pp, dp, zero)
        terms_per_gpu["activations"] = parallel_divide(act, "activations", tp, pp, dp, zero)

    subtotal = sum(v for v in terms_per_gpu.values())
    overhead = overhead_gb * (1024 ** 3)
    # fragmentation / allocator headroom
    total_per_gpu = (subtotal + overhead) * args.margin

    result = {
        "schema": "ml-vram-estimator/1",
        "model": {
            "name": arch["name"], "layers": arch["layers"], "hidden": arch["hidden"],
            "n_heads": arch["n_heads"], "kv_heads": arch["kv_heads"],
            "head_dim": arch["head_dim"], "vocab": arch["vocab"],
            "full_attn_layers": arch["full_attn_layers"], "is_hybrid": arch["is_hybrid"],
            "moe": arch["moe"], "weight_source": arch["weight_source"],
        },
        "scenario": {
            "mode": args.mode, "batch": args.batch, "seq": args.seq,
            "weight_dtype": args.weight_dtype, "kv_dtype": args.kv_dtype,
            "optimizer": args.optimizer if args.mode == "training" else None,
            "grad_ckpt": bool(args.grad_ckpt) if args.mode == "training" else None,
            "parallelism": {"tp": tp, "pp": pp, "dp": dp, "zero": zero},
            "overhead_gb": overhead_gb, "safety_margin": args.margin,
        },
        "breakdown_per_gpu_gb": {k: _gb(v) for k, v in terms_per_gpu.items()},
        "model_total_gb": {k: _gb(v) for k, v in terms_full.items() if v},
        "overhead_gb": overhead_gb,
        "total_per_gpu_gb": _gb(total_per_gpu),
        "n_gpus_implied": tp * pp * dp,
        "notes": [],
    }

    # notes about assumptions / data quality
    if "APPROX" in (arch["weight_source"] or ""):
        result["notes"].append("Weights are an ANALYTIC estimate — give extra margin.")
    if arch["is_hybrid"]:
        result["notes"].append(
            f"Hybrid attention: KV-cache counted for {arch['full_attn_layers']} "
            f"full-attn layers only (of {arch['layers']}). Linear/SSM layers add a "
            "small constant state not modeled here.")
    if arch["moe"]:
        result["notes"].append(
            "MoE: all expert weights occupy VRAM (counted in weights). Compute uses "
            "only active experts, but memory holds them all.")
    if args.mode == "training" and not args.grad_ckpt:
        result["notes"].append(
            "Training without gradient checkpointing: activations dominate at long "
            "seq (seq^2 attention term). Add --grad-ckpt to cut activation memory.")

    # verdict against available VRAM
    avail = resolve_available_vram(args)
    if avail is not None:
        result["available_vram_gb_per_gpu"] = avail
        ratio = result["total_per_gpu_gb"] / avail if avail else 99
        if ratio <= 0.85:
            verdict, emoji = "fits", "✅"
        elif ratio <= 1.0:
            verdict, emoji = "tight", "⚠️"
        else:
            verdict, emoji = "oom", "❌"
        result["verdict"] = verdict
        result["verdict_label"] = f"{emoji} {verdict} ({result['total_per_gpu_gb']}GB / {avail}GB = {round(ratio*100)}%)"
    return result


def resolve_available_vram(args) -> Optional[float]:
    if args.available_vram:
        return args.available_vram
    if args.from_env and os.path.exists(args.from_env):
        try:
            env = json.load(open(args.from_env))
            v = env.get("accelerator", {}).get("vram_gb")
            if v:
                return float(v)
            ram = env.get("platform", {}).get("ram_gb")  # unified-memory hosts
            if ram:
                return float(ram)
        except Exception:
            return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate GPU memory before launch.")
    src = p.add_argument_group("model (one of)")
    src.add_argument("--model", help="path to a HF model dir (uses real weight size)")
    src.add_argument("--config", help="path to a config.json")
    src.add_argument("--num-params-b", type=float, help="param count in billions")
    src.add_argument("--weights-gb", type=float, help="override weight memory directly")
    man = p.add_argument_group("manual arch (override/supplement config)")
    man.add_argument("--layers", type=int)
    man.add_argument("--hidden", type=int)
    man.add_argument("--kv-heads", type=int)
    man.add_argument("--head-dim", type=int)
    sc = p.add_argument_group("scenario")
    sc.add_argument("--mode", choices=["inference", "training"], default="inference")
    sc.add_argument("--batch", type=int, default=1)
    sc.add_argument("--seq", type=int, default=2048)
    sc.add_argument("--weight-dtype", default="bf16")
    sc.add_argument("--kv-dtype", default="bf16")
    sc.add_argument("--optimizer", choices=list(OPT_BYTES), default="adamw")
    sc.add_argument("--grad-ckpt", action="store_true")
    par = p.add_argument_group("parallelism")
    par.add_argument("--tp", type=int, default=1, help="tensor parallel")
    par.add_argument("--pp", type=int, default=1, help="pipeline parallel")
    par.add_argument("--dp", type=int, default=1, help="data parallel (DDP/FSDP group)")
    par.add_argument("--zero", type=int, default=0, choices=[0, 1, 2, 3],
                     help="ZeRO/FSDP stage (3=FSDP full shard)")
    vd = p.add_argument_group("verdict")
    vd.add_argument("--available-vram", type=float, help="GB per GPU")
    vd.add_argument("--from-env", help="env_report.json from ml-env-probe")
    p.add_argument("--overhead-gb", type=float, default=1.0,
                   help="CUDA context + framework overhead per GPU")
    p.add_argument("--margin", type=float, default=1.1,
                   help="fragmentation/headroom multiplier (default 1.1)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    result = estimate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
