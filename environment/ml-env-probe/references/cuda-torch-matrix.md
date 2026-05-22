# CUDA / 驱动 / PyTorch 版本矩阵

> last updated: 2026-01 (覆盖到 torch 2.7/2.8、CUDA 13.0)。版本边界拿不准时用
> `pip index versions torch` 或 https://pytorch.org/get-started/locally/ 复核。

裁决永远从**核心三元组**推:`compute_cap` → 最低 CUDA;`arch` → 有没有预编译 wheel;
`cuda` → 选哪个 torch wheel tag。

## 1. compute capability → 架构 → 最低 CUDA toolkit

| compute cap | 架构代号 | 代表卡 | 最低 CUDA | 备注 |
|---|---|---|---|---|
| 6.0/6.1 | Pascal | P100, GTX 10xx | 9.0 | torch 2.x 仍支持但渐弃 |
| 7.0 | Volta | V100 | 9.0 | |
| 7.5 | Turing | T4, RTX 20xx | 10.0 | |
| 8.0 | Ampere | A100 | 11.0 | |
| 8.6 | Ampere | A40, RTX 30xx | 11.1 | |
| 8.9 | Ada Lovelace | L40, RTX 40xx | 11.8 | |
| 9.0 | Hopper | H100, H200, GH200 | 11.8(推荐 12.x) | FA3 仅此架构 |
| 10.0 | Blackwell | B100, B200, GB200 | **12.8** | sm_100 |
| 12.0 / 12.1 | Blackwell | RTX 50xx, GB10 (DGX Spark) | **12.8** | sm_120 |

**关键:** 卡的 cc 决定最低 CUDA。CUDA 比这低 → kernel 根本跑不起来(`no kernel
image is available for execution on the device`)。

## 2. torch wheel 的 CUDA 标签

torch 预编译 wheel 按 CUDA 运行时打 tag,**自带 CUDA 运行时**(不需要系统装同版本
CUDA,除非你要源码编译别的包)。

| wheel tag | 适配 CUDA | 适用架构 | 起始 torch |
|---|---|---|---|
| `cu118` | 11.8 | Volta–Hopper | 长期 |
| `cu121` | 12.1 | Ampere–Hopper | 2.1+ |
| `cu124` | 12.4 | Ampere–Hopper | 2.4+ |
| `cu126` | 12.6 | Ampere–Hopper | 2.6+ |
| `cu128` | 12.8 | **含 Blackwell sm_100/sm_120** | **2.7+** |

安装(指定 index-url 选 tag):
```bash
# Hopper/Ampere 通用
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Blackwell(B200/RTX50/GB10)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 还没进 stable 时用 nightly
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

**驱动**:`nvidia-smi` 顶部的 "CUDA Version" 是**驱动支持的最高** CUDA,不是装了的
toolkit。只要它 ≥ wheel 的 CUDA 即可(向后兼容)。cu128 wheel 需要驱动 ≥ 525(Linux)。

## 3. aarch64 (ARM) 专项 —— 最容易翻车

ARM 服务器/工作站(GH200、GB200、GB10/DGX Spark、部分云实例)是 `aarch64`,不是
`x86_64`。后果:

- **torch**:官方有 ARM 的 CUDA wheel(sbsa,linux aarch64),2.4+ 起较全;
  Blackwell 走 cu128(可能要 nightly)。装之前确认 wheel 存在:
  `pip index versions torch --index-url https://download.pytorch.org/whl/cu128`。
- **flash-attn / xformers / bitsandbytes / 多数带 CUDA kernel 的包**:历史上
  **只发 x86_64 wheel** → 在 ARM 上 `pip install` 会去**源码编译**(慢、易 OOM、可能
  编不过),或直接 `No matching distribution`。处理见 `compiled-packages.md`。
- 经验:在 aarch64 上,默认假设"非 torch 的 CUDA 编译包都没预编译 wheel",逐个核实。

## 4. Blackwell (sm_100 / sm_120) 专项

- 必须 CUDA ≥ 12.8 + torch ≥ 2.7(cu128),很多时候要 nightly。
- **FlashAttention-2 没有 sm_120 kernel**(写本文时):导入或运行报错/回退。
  → 用 `torch.nn.functional.scaled_dot_product_attention`(PyTorch 自带,Blackwell
  上有优化后端),或等 flash-attn 官方出 sm_120 支持。FA3 是 Hopper 专属,不适配。
- vLLM / SGLang 等对 Blackwell 的支持要看版本说明,优先用其官方标注支持 sm_120 的版本。

## 5. 验证装对了没

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, \
torch.cuda.is_available(), torch.cuda.get_device_capability())"
```
- `cuda_available=False` 但你有卡 → 装成了 CPU build,或 wheel 的 CUDA 高于驱动。重装对的 tag。
- `get_device_capability()` 返回 `(12, 1)` 而 torch 不支持 → 报 `no kernel image`,
  说明这个 torch 没编 sm_120,换 cu128 / nightly。

## 6. cuDNN

torch wheel 自带匹配的 cuDNN,**一般不用单独装**。只有源码编译(如 apex 某些路径、
自编 TensorRT)才需系统 cuDNN,且要与 CUDA 大版本对齐(cuDNN 9.x 配 CUDA 12.x)。
