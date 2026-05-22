# 海光 DCU (Hygon)

> last updated: 2026-01。基于 ROCm 类技术栈,迁移成本相对低。

## 探测

- 海光 DCU 用类 ROCm 栈,可能出现在 `rocm-smi` / `rocminfo`,或厂商自带工具
  (`hy-smi` 一类,以实际环境为准)。
- 栈:**DTK (DCU Toolkit,类 ROCm) + 适配版 PyTorch**。

## 安装与使用

```bash
# 装厂商提供的 DTK 与适配 torch wheel(通常厂商内部源)
# DCU 沿用 HIP/ROCm 编程模型,torch 里设备仍多用 "cuda" 别名(HIP 映射)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```
- 因 HIP 兼容,**很多 CUDA 代码改动很小甚至不改**就能跑(`hipify` 转译),这是 DCU
  相对昇腾/寒武纪的优势。

## 能与不能

| 目标 | DCU 上 | 说明 |
|---|---|---|
| 标准 PyTorch | ✅/⚠️ | HIP 兼容性好,多数能跑;性能与算子覆盖看 DTK 版本 |
| flash-attn / xformers | ⚠️ | 看是否有 ROCm/HIP 移植版;CUDA 专属二进制不通,但有 ROCm 分支可编 |
| vLLM / DeepSpeed | ⚠️ | ROCm 路线常有支持,核实版本 |
| 多卡 | ✅ | 走 RCCL(ROCm 版 NCCL) |

**裁决总则**:DCU 因 HIP/ROCm 兼容,迁移门槛低于昇腾/寒武纪。优先找 ROCm/HIP 版本的
包;CUDA 专属预编译二进制仍不通,需 ROCm 分支重编。
