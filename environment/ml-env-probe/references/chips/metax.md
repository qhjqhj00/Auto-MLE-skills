# 沐曦 (MetaX)

> last updated: 2026-01。较新的国产 GPU,生态仍在快速建设,务必核实当前版本能力。

## 探测

- 工具:`mx-smi`(看 GPU 型号/数量/显存)。
- 栈:**MACA (MetaX 计算架构,类 CUDA) + 适配版 PyTorch**。

## 安装与使用

- 用厂商提供的 MACA toolkit 与适配 torch wheel(通常厂商源)。
- MACA 设计上对标 CUDA,提供 CUDA 兼容层,意在降低迁移成本;实际兼容程度看版本与算子。

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## 能与不能

| 目标 | MetaX 上 | 说明 |
|---|---|---|
| 标准 PyTorch | ⚠️ | 常见路径可;算子覆盖与稳定性看 MACA 版本,需实测 |
| flash-attn / xformers | ⚠️/❌ | 看有无 MACA 适配;CUDA 专属二进制不通 |
| vLLM / 训练框架 | ⚠️ | 看厂商有无适配分支 |
| 多卡 | ✅(看版本) | 走厂商集合通信库 |

**裁决总则**:生态最年轻,**别假设**主线 CUDA 包能跑。以厂商适配版为准,拿不准就标
"需实测"并如实告知不确定性,而不是给一个可能错的肯定结论。
