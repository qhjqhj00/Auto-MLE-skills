# 华为昇腾 (Ascend NPU) — 深度指南

> last updated: 2026-01。国产里生态最成熟。核心栈:**CANN(底层) + torch_npu(PyTorch 适配)**。

## 探测

- 工具:`npu-smi info`(看卡型号、数量、显存、温度)。
- 环境变量:`ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 指向 CANN 安装路径。
- 常见型号:训练 910B/910C(310 系列偏推理)。

## 关键概念:版本三件套必须对齐

昇腾翻车几乎都因版本不匹配。三者必须配套(参考官方"版本配套表"):

```
固件+驱动 (firmware/driver)  ←→  CANN (toolkit)  ←→  torch_npu (要对应某个 torch 版本)
```

- `torch_npu` 的版本号对应一个具体的 `torch` 版本(如 `torch==2.1.0` ↔
  `torch_npu==2.1.0.postX`)。**先装 CPU 版 torch,再装同版本 torch_npu**。
- CANN 版本要与驱动版本匹配;torch_npu 又要与 CANN 匹配。任一错位 → import 即报错。

## 标准安装

```bash
# 1. 装 CPU 版 torch(注意:不是 CUDA wheel)
pip install torch==2.1.0   # 版本以 torch_npu 配套表为准
# 2. 装匹配的 torch_npu
pip install torch_npu==2.1.0.post*   # 按配套表
# 3. source CANN 环境
source ${ASCEND_HOME_PATH}/set_env.sh
# 4. 验证
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

## 写代码的差异

- 设备字符串用 `"npu"` 而非 `"cuda"`:`tensor.to("npu")`、`torch.npu.synchronize()`。
- 很多代码可用 `torch_npu` 的迁移工具/补丁让 `.cuda()` 自动映射到 npu,但显式写 `npu` 更稳。
- 算子覆盖:常见算子齐全;**自定义/冷门 CUDA 算子没有对应实现** → 报"算子不支持"。
  此时找 CANN 的等价算子或用昇腾版的库(如 `mindspeed` / `ModelLink` 训练框架)。

## 能与不能(裁决要点)

| 目标 | 昇腾上 | 说明 |
|---|---|---|
| 标准 transformers 训练/推理 | ✅ 多数可 | 用 torch_npu;部分算子需昇腾适配版库 |
| flash-attention(CUDA 版) | ❌ | 是 CUDA kernel;用昇腾的 FA 实现(CANN 内置 / mindspeed 的融合算子) |
| vLLM | ⚠️ | 用 `vllm-ascend` 插件项目,不是主线 vLLM |
| DeepSpeed / Megatron | ⚠️ | 用昇腾适配分支(如 MindSpeed),非上游原版 |
| 多卡并行 | ✅ | 通信走 **HCCL**(对应 NCCL),互联是 HCCS |
| bitsandbytes / 多数 CUDA-only 量化 | ❌ | 找昇腾原生量化方案 |

## 排错

- `import torch_npu` 失败:先 `source set_env.sh`;检查 torch ↔ torch_npu ↔ CANN ↔ 驱动四者配套。
- "算子不支持":换昇腾适配的模型库,或在 CANN 文档查等价算子。
- 性能差:确认用了昇腾融合算子(FA、RMSNorm 等),而非逐算子回退。

**裁决总则**:能不能跑取决于"有没有昇腾适配版"。主线 CUDA 生态默认不通,优先找
`xxx-ascend` / MindSpeed / ModelLink 这类适配实现;找不到适配就是 ❌,给替代或如实告知。
