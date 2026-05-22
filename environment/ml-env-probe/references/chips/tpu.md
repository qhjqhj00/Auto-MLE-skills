# Google TPU

> last updated: 2026-01。

## 探测

- 无 smi 工具。靠信号判断:`/dev/accel0` 设备节点、`libtpu` 库、环境变量
  `TPU_NAME` / `COLAB_TPU_ADDR`、或在 GCP TPU VM 上。
- 形态:单 TPU、TPU Pod slice(多芯片,有自己的 mesh 拓扑)。

## 框架选择

| 框架 | TPU 上 | 说明 |
|---|---|---|
| **JAX** | ✅ 首选 | TPU 一等公民,XLA 编译,性能与生态最好 |
| **torch_xla** | ✅ | PyTorch on TPU,通过 XLA;比 JAX 略多坑但能用 PyTorch 代码 |
| 原生 CUDA torch | ❌ | TPU 不是 CUDA,完全不通 |
| flash-attn / xformers / 任何 CUDA kernel | ❌ | CUDA 专属;XLA 有自己的 attention 实现/融合 |

## 安装(torch_xla 路线)

```bash
pip install torch torch_xla   # 版本需与 TPU runtime / libtpu 匹配,按官方文档
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```
JAX 路线:`pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`。

## 并行

- 切分按 **mesh / SPMD**,与 GPU 的 TP/PP 概念不同。JAX 用 `shard_map`/`jit` +
  `Mesh`;torch_xla 用 SPMD API。
- TPU Pod 的拓扑(2D/3D torus)由 runtime 管理,不像 NVLink 那样手动判 topo。

## 裁决总则

TPU 上**优先 JAX**;有现成 PyTorch 代码且想复用就 torch_xla。任何 CUDA 专属包一律
❌,改用 XLA 原生路径。给报告时说明"这是 TPU,不要尝试装 CUDA/flash-attn"。
