# Apple Silicon (MPS) 与 CPU-only

> last updated: 2026-01。

## Apple Silicon (M 系列)

- 后端:**MPS** (Metal Performance Shaders)。`arch == arm64`,**无 CUDA**。
- 安装:`pip install torch`(官方 wheel 自带 MPS,无需特殊 index)。
- 设备:`torch.backends.mps.is_available()` → `tensor.to("mps")`。
- 统一内存:CPU/GPU 共享内存,大模型受总内存限制(不是独立显存)。

| 目标 | MPS 上 | 说明 |
|---|---|---|
| 中小模型推理 / 轻量训练 | ✅ | 适合开发、原型、小模型微调 |
| flash-attn / xformers / bitsandbytes | ❌ | CUDA 专属;用 SDPA(MPS 有支持) |
| 大模型训练 | ❌/⚠️ | 算力与内存带宽不足以做严肃训练 |
| 部分算子 | ⚠️ | 偶有未实现算子,设 `PYTORCH_ENABLE_MPS_FALLBACK=1` 回退 CPU |
| 量化推理 | ✅ | 用 llama.cpp / MLX(Apple 原生框架,大模型本地推理更优) |

**裁决**:适合开发和小规模推理;严肃训练 ❌。本地大模型推理优先考虑 **MLX** 或
llama.cpp(Metal),比 torch-MPS 更成熟。

## CPU-only

- 后端:CPU。`pip install torch`(默认即 CPU,或 `--index-url .../whl/cpu`)。

| 目标 | CPU 上 | 说明 |
|---|---|---|
| 小模型推理 / 调试 / 跑通流程 | ✅(慢) | 验证代码正确性够用 |
| 任何规模训练 | ❌ 实际不可行 | 太慢 |
| 大模型推理 | ⚠️ | 用 llama.cpp + GGUF 量化勉强可跑,看内存 |
| flash-attn 等 CUDA 包 | ❌ | 不适用 |

**裁决**:CPU 用于"能跑通逻辑",不用于"跑出结果"。给用户的报告里要明确这一点,避免
agent 在 CPU 上启动一个永远跑不完的训练。
