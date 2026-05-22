# 寒武纪 (Cambricon MLU)

> last updated: 2026-01。生态成熟度低于昇腾,核实当前版本支持。

## 探测

- 工具:`cnmon`(看 MLU 卡型号/数量/利用率)。
- 栈:**Neuware (底层 SDK,含 CNRT/CNNL 等) + torch_mlu (PyTorch 适配)**。

## 安装与使用

```bash
# 装 Neuware SDK(厂商提供),再装匹配的 torch + torch_mlu
pip install torch_mlu   # 版本需与 torch、Neuware 配套(同昇腾的配套思路)
python -c "import torch, torch_mlu; print(torch.mlu.is_available())"
```
- 设备字符串 `"mlu"`:`tensor.to("mlu")`。
- 版本三件套:**驱动 ↔ Neuware ↔ torch_mlu(对应某 torch 版本)** 必须配套。

## 能与不能

| 目标 | MLU 上 | 说明 |
|---|---|---|
| 标准 PyTorch 模型 | ⚠️ | 常见算子可,冷门算子可能缺;用 Cambricon 适配的模型库 |
| flash-attn / xformers (CUDA) | ❌ | CUDA 专属;用厂商融合算子或退回标准 attention |
| vLLM / DeepSpeed 等 | ⚠️ | 看有无 Cambricon 适配分支,无则 ❌ |
| 多卡 | ✅ | 通信走 CNCL(对应 NCCL) |

**裁决总则**:与昇腾同理 —— 取决于有没有寒武纪适配版。主线 CUDA 生态默认不通。
没有适配就如实告知 ❌ 并给替代,**不要假装能跑**。
