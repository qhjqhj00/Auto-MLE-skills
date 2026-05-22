---
name: ml-env-probe
description: >-
  Probe the machine's ML hardware/software environment and tell the agent what
  it can and cannot run BEFORE attempting any ML setup. Produces a JSON manifest
  (env_report.json) plus a human-readable capability report with tiered ✅/⚠️/❌
  verdicts and concrete install commands. Use at the START of any MLE automation,
  or whenever the user asks about: CUDA / driver / PyTorch version compatibility,
  why flash-attention / xformers / apex / bitsandbytes / deepspeed won't install
  or build, which torch wheel to pick, how to set up an isolated env (uv/conda/venv),
  package conflicts (transformers↔tokenizers, numpy 2.x), or how to split a model
  across GPUs (tensor/pipeline/data parallel, FSDP). Covers NVIDIA, Apple MPS, CPU,
  Google TPU, and Chinese accelerators (昇腾 Ascend, 寒武纪 Cambricon, 海光 Hygon DCU,
  沐曦 MetaX). Triggers: "环境探测", "能跑什么", "装不上", "编译失败", "CUDA 版本",
  "torch 版本", "flash-attn", "环境清单", "环境隔离", "并行策略", "tensor parallel".
---

# ml-env-probe — ML 环境与能力探测

**第一层:环境与依赖管理。** ML 工程最容易翻车的地方,也是 agent 自动跑 MLE 前
必须先过的关。这个 skill 的唯一目的:**先探测真实环境,再告诉 agent 它的"能与不能"**,
避免照抄 README 式安装在某块卡 / 某个架构上撞墙而 agent 自己意识不到。

## 核心心法(先读这三条)

1. **探测是脚本的事,裁决是你的事。** `scripts/detect_env.py` 跑出确定性的事实
   清单 (`env_report.json`)。**绝不要凭印象猜 GPU 型号、显存、CUDA 版本** —— 一律
   以脚本输出为准。你的工作是拿事实 + reference 矩阵做判断。

2. **"能与不能"分三档,永远带退路:**

   | 档 | 含义 | 你要给出 |
   |----|------|---------|
   | ✅ **能跑** | 有官方预编译 wheel | 一行 `pip/uv install` |
   | ⚠️ **能跑但要编译/有坑** | 需源码编译或版本敏感 | 编译命令 + 预估耗时 + 防 OOM 的 `MAX_JOBS` + 常见坑 **并给降级方案** |
   | ❌ **跑不了** | 架构/版本不支持 | **等价替代**(如 flash-attn → PyTorch SDPA) |

   只说"行/不行"是没用的;agent 需要的是带退路的判断。

3. **输出语种跟随用户的提问语种 (match the user's query language).** 判断用户这次
   请求用的是哪种语言,**所有给人看的产物都用那种语言**:你对用户说的话、`ENV_CAPABILITY.md`
   的标题与正文、裁决表里的"理由/行动"。用户用英文问("help me inspect the env")就
   全英文,用中文问就中文,其他语种同理。
   - **例外:`env_report.json` 永远保持原样**(英文键名 + 英文 `flags` 消息)——它是机器
     可读清单,语种固定才好被下游程序消费。你向用户复述 `flags` 时再翻译成其语种。
   - reference 文件是你自己读的内部知识,什么语种不影响你产出何种语种的报告。

## 工作流(按顺序)

### Step 1 — 探测

```bash
python3 scripts/detect_env.py --out env_report.json
```

纯 stdlib,任何机器都能跑、不会崩。覆盖 NVIDIA / Apple MPS / CPU / TPU / 昇腾 /
寒武纪 / 海光 / 沐曦,缺什么工具就把对应字段标 null。输出的 `flags` 字段是**风险旗标**
(`error` / `warn` / `info`),是你裁决的起点。

> 如果 `env_report.json` 已存在且机器没变,可直接复用,不必重跑。

### Step 2 — 按旗标加载对应 reference

**不要一次读完所有 reference。** 看 `flags` 和 `accelerator.vendor`,只加载相关的:

| 触发条件 | 加载 |
|---------|------|
| 任何 NVIDIA 卡 / 选 torch 版本 / `cuda-too-old` `blackwell` `arm-cuda` 旗标 | `references/cuda-torch-matrix.md` |
| 要装 flash-attn / xformers / apex / bitsandbytes / deepspeed | `references/compiled-packages.md` |
| 要建新环境 / `transformers-tokenizers` `numpy2` 旗标 / 选 uv-conda-venv | `references/env-isolation.md` |
| `accelerator.count > 1` 或多机 / 问怎么切模型 | `references/parallelism-decision.md` |
| `vendor` 是 ascend/cambricon/metax/amd-or-hygon-rocm/apple/google-tpu | `references/chips/<vendor>.md` |

### Step 3 — 产出裁决 + 安装方案

把事实 + 旗标 + reference 合成。**核心三元组**永远是
`compute_cap + arch(x86_64/aarch64) + cuda` —— 一切从它推。给出:

- 一张**能力裁决表**(每个目标包/能力一行,✅/⚠️/❌ + 一句理由 + 行动)
- 选定的 **torch / CUDA 组合** + 精确安装命令(指明 index-url / wheel tag)
- 推荐的**环境隔离**方案(给出可直接执行的建环境命令)
- 如多卡:推荐的**并行策略**(TP/PP/DP/FSDP)+ 理由

### Step 4 — 落盘两份产物

1. `env_report.json` —— 机器可读,下游 agent 程序化读取(`flags`、`accelerator` 可逐项查)。**保持英文,不翻译。**
2. `ENV_CAPABILITY.md` —— 人类可读摘要,用下面的模板,**语种跟随用户的提问语种**(见心法 3)。

## 输出模板:ENV_CAPABILITY.md

> 下面是**中文 query 时**的渲染示例。英文 query 时把所有标题/正文/"理由·行动"译成英文
> (Hardware / Capability verdict / Recommended install order / Known pitfalls …),
> emoji 与表格结构不变。其他语种同理。

```markdown
# 环境能力清单 — <主机名/日期>

## 硬件
- 加速器: <vendor> <name> ×<count>  | compute cap <cc> (<codename>) | 显存 <vram>GB
- 架构: <os>/<arch>  Python <py>  CUDA <cuda> 驱动 <driver>  RAM <ram>GB
- 互联: <topology>(NVLink: 有/无)  多机: <是/否>

## 能力裁决
| 目标 | 能否 | 理由 | 行动 |
|------|------|------|------|
| PyTorch | ✅ | cu128 有 aarch64 wheel | `uv pip install torch --index-url ...` |
| flash-attention | ❌ | sm_120+aarch64 无预编译且 FA2 无 sm_120 kernel | 用 `torch.nn.functional.scaled_dot_product_attention` |
| xformers | ⚠️ | 无 aarch64 wheel,需源码编译 ~40min | `pip install -v --no-build-isolation ...` 或退回 SDPA |
| 并行 | N/A | 单卡 | 不适用;模型超显存用 device_map/offload |

## 推荐安装序(按依赖顺序,先 torch 后编译包)
1. 建环境: <命令>
2. 装 torch(锁版本): <命令>
3. 装编译包(关 build isolation): <命令>

## 已知坑
- <从 flags + reference 提炼的 2-4 条最相关的>
```

## 关键裁决规则速查(细节在 reference 里)

- **核心三元组**:`compute_cap` 定最低 CUDA;`arch` 定有没有预编译 wheel(aarch64
  常常没有);`cuda` 定 torch wheel tag(cu118/cu121/cu124/cu126/cu128)。
- **Blackwell (sm_100/sm_120)**:CUDA ≥ 12.8,torch ≥ 2.7 / nightly cu128。
  FlashAttention-2 无 sm_120 kernel → 用 SDPA。
- **aarch64**:torch 有官方 ARM CUDA wheel(2.4+),但 flash-attn/xformers/bnb 多半
  没有 → 源码编译或降级。
- **顺序铁律**:先装并锁定 torch,再 `--no-build-isolation` 编译 flash-attn/apex/
  xformers —— 否则 pip 会拉错 torch 版本导致 ABI 崩。
- **transformers ↔ tokenizers**:每个 transformers 锁定一段 tokenizers 区间,单独
  升级其一是头号静默 bug。
- **并行**:单卡免谈;多卡有 NVLink → TP 友好;只有 PCIe → TP 受带宽限,优先
  DP/FSDP 或 PP;模型 > 单卡显存 → 必须 TP/PP/offload。决策树见 reference。

## 边界

- 这个 skill **只探测和建议**,不擅自改系统(不动驱动、不全局装包)。建环境/装包的
  命令产出给用户或下游 agent 执行,除非用户明确让你执行。
- 兼容性矩阵会过期。版本类断言以 reference 文件为准,reference 标了 "last updated";
  拿不准的版本边界提示用户用 `pip index versions <pkg>` 或官方 matrix 复核。
