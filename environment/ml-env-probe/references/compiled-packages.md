# 需编译的包:flash-attention / xformers / apex / bitsandbytes / deepspeed

> last updated: 2026-01。这些包带 CUDA kernel,版本错一个就编译失败或 ABI 崩。
> 核心是先判断"有没有预编译 wheel",没有再决定"编译还是降级"。

## 通用铁律(适用所有编译包)

1. **先装并锁定 torch,再装这些包。** 它们要按已装的 torch 编/选 wheel。
2. **关 build isolation**:`pip install <pkg> --no-build-isolation`。否则 pip 会在
   隔离环境里**重新拉一个 torch**(往往是错版本)→ 编出来的 kernel 和你运行时的
   torch ABI 不匹配,import 即崩。
3. **预编译 wheel 的匹配维度**:`torch 版本 × CUDA tag × python 版本 × cxx11abi ×
   平台(x86_64/aarch64)`。**任一不符就没有 wheel**,会回退到源码编译。
4. **源码编译要有 nvcc**,且 nvcc 的 CUDA 大版本要和 torch 的 CUDA 对齐。

## flash-attention

| 情况 | 裁决 | 行动 |
|---|---|---|
| x86_64 + Ampere/Hopper + 常见 torch/py | ✅ 有 wheel | 从 release 选对应 wheel,或 `pip install flash-attn --no-build-isolation` |
| aarch64 | ⚠️/❌ 通常无 wheel | 源码编译(见下)或降级 SDPA |
| Blackwell sm_120 (RTX50/GB10) | ❌ FA2 无 sm_120 kernel | **用 SDPA**,见降级 |
| Hopper 想要 FA3 | ✅ Hopper 专属 | 装 flash-attn v3 分支/接口 |

**源码编译(无 wheel 时):**
```bash
# 关键:限制并行编译数,否则每个 nvcc 进程吃几 GB,极易 OOM 把机器搞挂
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
```
- 耗时:30 分钟 ~ 2+ 小时,视 CPU 核数与 `MAX_JOBS`。
- 内存:每个编译 job 峰值 2–4GB;`MAX_JOBS=4` 约需 ~16GB 空闲内存。内存小就调到 2。
- 失败常见因:nvcc 缺失 / nvcc 与 torch CUDA 不一致 / python 头文件缺(装 `python3-dev`)。

**降级方案(❌ 或不想编时):** PyTorch 自带的
`torch.nn.functional.scaled_dot_product_attention`(SDPA),会自动选 flash /
mem-efficient / math 后端。transformers 里设 `attn_implementation="sdpa"`。Blackwell
上这是当前最稳的注意力路径。次选 xformers 的 `memory_efficient_attention`。

## xformers

- 版本与 torch **强绑定**,装错 torch 版本会触发重装 torch → 崩。优先用
  `pip install xformers --index-url https://download.pytorch.org/whl/cu128`(随 torch
  一起,版本自洽)。
- aarch64 / 新 CUDA 常无 wheel → 源码编译(同样 `--no-build-isolation` + `MAX_JOBS`)。
- 用途多数能被 SDPA 覆盖;非必需就别折腾。

## apex (NVIDIA)

- 几乎总是源码编译,且**极其版本敏感**。
```bash
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-build-isolation \
  --no-cache-dir --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./apex
```
- 现状:apex 的多数功能(fused optimizer、LayerNorm、混合精度)已被 **PyTorch 原生 +
  `torch.amp`** 取代。**优先用原生**,只有特定老代码硬依赖 apex 才编。

## bitsandbytes

- 8bit/4bit 量化。CUDA 版本相关,**aarch64 支持历史上很弱**(新版在改善,核实当前版本)。
- 装不上时:用 `torchao` 量化,或换支持目标架构的量化后端。

## deepspeed

- 多数算子运行时 JIT 编译(需 nvcc),装 wheel 即可:`pip install deepspeed`。
- 某些算子(如 `sparse_attn`)装不全是正常的,用到才需补。aarch64/Blackwell 上先验证
  `ds_report` 看哪些 op 可用。

## 排错速查

| 报错 | 病因 | 处理 |
|---|---|---|
| `No matching distribution` | 该平台/版本无 wheel | 源码编译或降级 |
| `no kernel image is available` | kernel 没编你这张卡的 cc | 换支持该 sm 的版本 / 重编指定 `TORCH_CUDA_ARCH_LIST` |
| import 时 `undefined symbol` | 与运行时 torch ABI 不匹配 | 重装,务必 `--no-build-isolation` |
| 编译中途被 kill / OOM | 并行 nvcc 吃爆内存 | 降 `MAX_JOBS` |
| `nvcc not found` | 没装 CUDA toolkit | 装匹配 torch CUDA 的 toolkit |

指定只编当前卡的架构能大幅缩短编译时间、避免无关 sm 报错:
```bash
export TORCH_CUDA_ARCH_LIST="12.0"   # 按 compute_cap 填,如 Blackwell sm_120 写 12.0
```
