# Environment Capability Report — spark-0 (DGX Spark / GB10), 2026-05-22

> Produced by ml-env-probe (English query → English report). Machine-readable
> manifest in `env_report.json`. Version verdicts confirmed live via
> `pip index versions`, not asserted from a static matrix.

## Hardware
- Accelerator: **NVIDIA GB10** ×1 | compute cap **12.1 (Blackwell, sm_120)** | VRAM: unified memory (nvidia-smi reports N/A)
- Platform: **linux / aarch64 (ARM)** · Python **3.13.13** · CUDA **13.0** · driver **580.142** · RAM **121.7GB**
- Interconnect: single GPU (no NVLink) · multi-node: no
- Package managers: conda, pip (**no uv**) · currently in conda **base**

## Capability verdict (live-verified)
| Target | Verdict | Reason (verified live) | Action |
|--------|---------|------------------------|--------|
| **PyTorch** | ✅ | `cu128` stable wheels exist for aarch64+cp313: found 2.7.0→**2.11.0** | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| **flash-attention** | ⚠️→leans ❌ | PyPI ships sdist only (no prebuilt wheel) → source build; heavy on aarch64, and FA2 sm_120 kernel support is version-dependent and unverified | **Default to `F.scaled_dot_product_attention`**; only if truly needed try `MAX_JOBS=4 ... --no-build-isolation` |
| **xformers** | ❌ | cu128 index returns `No matching distribution` (no aarch64 wheel) | Fall back to SDPA |
| **bitsandbytes** | ⚠️ | weak aarch64 support | Use `torchao` quantization |
| **vLLM** | ⚠️ | needs a version that officially supports sm_120 + aarch64 | Verify version first, else infer directly via transformers |
| **Parallelism (TP/PP/DP)** | N/A | single GPU | N/A; if model exceeds memory use `device_map="auto"` / quantization / offload |

## Recommended install order (torch first, compiled packages last; prefer Python 3.11)
```bash
# base is py3.13 with numpy2; create a clean env
conda create -n proj python=3.11 -y && conda activate proj
# Blackwell needs cu128 (stable verified available on this host)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# verify the sm_120 kernels are present
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"
# upper-layer frameworks (attention via SDPA, do NOT install flash-attn)
pip install transformers accelerate datasets
```

## Known pitfalls (most relevant here)
1. **aarch64**: torch has wheels, but flash-attn must compile from source and xformers/bnb essentially have no wheel → default attention to SDPA.
2. **Blackwell sm_120**: CUDA must be ≥12.8 (this host is 13.0 ✅); don't rely on FA2 sm_120 kernels.
3. **Python 3.13 too new**: use 3.11–3.12 for new envs to avoid cp313 wheel gaps.
4. **conda base = py3.13 + numpy 2.4.4**: don't install project packages into base; on `_ARRAY_API not found`, pin `numpy<2` in the project env.
5. **No uv**: install it first (`pip install uv` or `conda install -c conda-forge uv`), otherwise use conda+pip.
