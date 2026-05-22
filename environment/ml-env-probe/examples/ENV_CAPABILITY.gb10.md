# 环境能力清单 — spark-0 (DGX Spark / GB10), 2026-05-22

> 这是 ml-env-probe 在一台真实机器上跑出来的示例输出(Step 1 探测 → Step 3 裁决)。
> 配套的机器可读清单见同目录 `env_report.gb10.json`。

## 硬件
- 加速器: **NVIDIA GB10** ×1 | compute cap **12.1 (Blackwell, sm_120)** | 显存: 统一内存(nvidia-smi 报 N/A)
- 架构: **linux / aarch64 (ARM!)** · Python **3.13.13** · CUDA **13.0** · 驱动 **580.142** · RAM **121.7GB**
- 互联: 单卡(无 NVLink) · 多机: 否
- 包管理: conda、pip(无 uv)· 当前在 conda **base** 环境

## 能力裁决
| 目标 | 能否 | 理由 | 行动 |
|------|------|------|------|
| **PyTorch** | ⚠️ | 需 aarch64 的 cu128 构建,稳定版可能缺 cp313 wheel | 用 cu128(必要时 nightly),并优先 Python 3.11 重建环境 |
| **flash-attention** | ❌ | sm_120 无 FA2 kernel **且** aarch64 无预编译 wheel | 用 `F.scaled_dot_product_attention` / `attn_implementation="sdpa"` |
| **xformers** | ❌/⚠️ | aarch64 无 wheel,源码编译在 sm_120 上未必有 kernel | 退回 SDPA |
| **bitsandbytes** | ⚠️ | aarch64 支持弱 | 改用 torchao 量化 |
| **vLLM** | ⚠️ | 需其官方标注支持 sm_120 + aarch64 的版本 | 核实版本后再装,否则用 transformers 直接推理 |
| **并行** | N/A | 单卡 | 不适用;模型超显存用 `device_map="auto"` / 量化 / offload |

## 推荐安装序(先 torch、后编译包;Python 降到 3.11 更稳)
```bash
# base 是 py3.13,新建 3.11 环境避开 cp313 wheel 缺失
conda create -n proj python=3.11 -y && conda activate proj
# Blackwell 走 cu128;稳定版没有就用 nightly
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
# 验证 sm_120 kernel 在位
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"
# 上层框架
pip install transformers accelerate datasets
# 不装 flash-attn —— 本机用 SDPA
```

## 已知坑(本机最相关的 4 条)
1. **aarch64**:除 torch 外,默认假设带 CUDA kernel 的包(flash-attn/xformers/bnb)都没预编译 wheel,逐个核实。
2. **Blackwell sm_120**:CUDA 必须 ≥12.8、torch ≥2.7;FA2 无 sm_120 kernel,注意力一律走 SDPA。
3. **Python 3.13 过新**:不少 ML 包还没出 cp313 wheel → 建议新环境用 3.11–3.12。
4. **conda base 已有 numpy 2.x**:遇到 `_ARRAY_API not found` 就在项目环境里 `pin numpy<2`;别在 base 装业务包。
