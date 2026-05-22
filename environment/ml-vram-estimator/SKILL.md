---
name: ml-vram-estimator
description: >-
  Estimate GPU memory (VRAM) BEFORE launching, so a run doesn't OOM hours in.
  Given a model (HF config.json / model dir, or manual params) and a scenario
  (inference vs training, batch size, sequence length, dtype, parallelism), it
  computes a per-GPU breakdown — weights + KV-cache + gradients + optimizer +
  activations + overhead — and returns a fits ✅ / tight ⚠️ / OOM ❌ verdict with
  concrete fallbacks (smaller batch, shorter seq, quantize, grad checkpointing,
  more GPUs, FSDP/ZeRO, offload). Correctly handles GQA, MoE (all experts in
  VRAM), hybrid attention (KV-cache only for full-attention layers), and reads
  real on-disk weight size for quantized models. Use when the user asks: will
  this model fit, how much VRAM does X need, why did it OOM, what batch/seq can
  I run, how many GPUs do I need, KV cache size, activation memory, optimizer
  memory. Triggers: "显存", "会不会 OOM", "放得下吗", "显存估算", "VRAM", "OOM",
  "batch size 能开多大", "需要几张卡", "KV cache", "fit in memory", "memory footprint".
---

# ml-vram-estimator — 事前显存估算

事前算清楚显存,别等跑起来几小时后才 OOM。给定模型 + 场景(推理/训练、batch、seq、
dtype、并行),算出每卡显存分项,给 ✅/⚠️/❌ 裁决和退路。

## 核心心法

1. **算是脚本的事,裁决和退路是你的事。** `scripts/estimate_vram.py` 跑确定性算术,
   产出分项 breakdown + 总量。你的工作是结合可用显存给裁决、并在 OOM 时给**可执行的退路**。
   **优先用真实权重大小**(给 `--model <dir>` 时脚本读磁盘 safetensors,自动含量化开销)。

2. **显存 = 这几项之和,别只算权重:**

   | 场景 | 分项 |
   |------|------|
   | 推理 | 权重 + **KV cache**(随 batch×seq 增长) + 少量激活 + 开销 |
   | 训练 | 权重 + 梯度 + **优化器状态(最大头,Adam ≈ 12B/参)** + **激活(长 seq 时爆炸)** + 开销 |

   OOM 八成栽在被忽略的项上:推理是 KV cache,训练是优化器状态和激活。

3. **裁决三档,OOM 必带退路:**

   | 档 | 判据(占可用显存) | 给什么 |
   |----|------|--------|
   | ✅ fits | ≤ 85% | 可以跑;可顺带说还能开多大 batch/seq |
   | ⚠️ tight | 85–100% | 能跑但危险;给降一档的具体参数 |
   | ❌ oom | > 100% | **按性价比排序给退路**(见下),并标出哪一项是元凶 |

4. **输出语种跟随用户的提问语种 (match the user's query language).** 给人看的产物
   (你的结论、报告正文、breakdown 解读)用用户提问的语种;脚本输出的 JSON 保持英文
   (机器可读,复述时再翻译)。

## 工作流

### Step 1 — 拿到模型架构与权重大小

- 有本地权重目录 → `--model <dir>`(最准,直接读磁盘大小,量化也算对)。
- 只有 config → `--config config.json`。
- 都没有,只知道规模 → `--num-params-b 7 --hidden 4096 --layers 32 --kv-heads 8 --head-dim 128`。

脚本会自动识别 **GQA(num_key_value_heads)**、**MoE(num_experts)**、**混合注意力
(layer_types 里的 full vs linear/SSM)**——这些是 naive 估算最常算错的地方。

### Step 2 — 算

```bash
# 推理:能不能放下 + 直接对比本机可用显存(吃 ml-env-probe 的产物)
python3 scripts/estimate_vram.py --model <dir> --mode inference \
    --batch 1 --seq 8192 --weight-dtype fp8 \
    --from-env ../ml-env-probe/env_report.json

# 训练:全量微调
python3 scripts/estimate_vram.py --config config.json --mode training \
    --batch 4 --seq 4096 --weight-dtype bf16 --optimizer adamw \
    --grad-ckpt --zero 3 --dp 8 --available-vram 80
```

关键参数:`--mode` · `--batch` · `--seq` · `--weight-dtype`(bf16/fp16/fp8/int4/nvfp4)
· `--kv-dtype` · `--optimizer`(adamw/sgd/adafactor/none)· `--grad-ckpt` ·
并行 `--tp/--pp/--dp/--zero`(zero 3 = FSDP 全分片)· `--available-vram` 或 `--from-env`。

> `--available-vram` 是**每卡** GB。统一内存机器(如 GB10/Apple)没有独立显存时,
> `--from-env` 会回退到总 RAM —— 提醒用户这与独立显存语义不同,留更多余量。

### Step 3 — 裁决 + 退路

读脚本输出的 `verdict_label`、`breakdown_per_gpu_gb`、`notes`。**OOM 时按下面性价比
顺序给退路**(细节和数量级在 references 里):

1. **降 batch / seq**(线性降 KV cache 与激活;最便宜)
2. **梯度检查点 `--grad-ckpt`**(训练激活砍到零头,代价 ~30% 算力)
3. **量化权重 / KV**(fp8/int4 砍权重;kv-dtype fp8 砍 KV cache)
4. **LoRA/QLoRA**(训练时几乎消掉优化器状态那一大块)
5. **分片到多卡**:FSDP/ZeRO-3(`--zero 3 --dp N`)切权重+梯度+优化器;或 TP(`--tp N`,需 NVLink)
6. **CPU/NVMe offload**(最后手段,慢)

给结论时**点名元凶**:"OOM 主因是优化器状态 78GB → 上 ZeRO-3 或 LoRA"。

## 输出模板:VRAM_ESTIMATE.md(语种跟随 query)

```markdown
# 显存估算 — <模型> / <场景一句话>

- 模型: <name> | <layers>L hidden<h> | GQA kv<kv> | <MoE: 总参/激活> | 混合注意力: full <x>/<L>
- 场景: <mode> batch<b> seq<s> dtype<dt> | 并行 tp<>/pp<>/dp<>/zero<>
- 可用显存: <avail>GB/卡 (来源: 独立显存 / 统一内存)

## 裁决: <✅/⚠️/❌> <总量>GB / <avail>GB = <%>

| 分项 | GB/卡 | 备注 |
|------|------|------|
| 权重 | | <来源:磁盘真实大小/解析估算> |
| KV cache | | 仅 <full_attn_layers> 层全注意力 |
| 梯度 | | 训练 |
| 优化器 | | Adam≈12B/参 |
| 激活 | | <grad-ckpt 与否> |

## 退路(若 ⚠️/❌,按性价比)
1. <最相关的 2-4 条,带具体参数,如 "seq 4096→2048 省 ~32GB">
```

## 边界

- 估算是**数量级 + 安全余量**,不是字节级精确(默认 `--margin 1.1` + 1GB 开销)。
  目的是不 OOM;留 15% 余量再下结论。
- 激活内存随实现/框架差异大(flash-attn 省、长 seq 的 seq² 项主导);脚本给的是
  保守近似,reference 里说明假设。拿不准就让用户先用小 batch 实测一步再外推。
- 权重用 `--weight-dtype` 改 dtype 时,若给的是 `--model`(磁盘已是某量化),脚本按
  磁盘大小报、不会重算到别的 dtype;要换 dtype 估算请用 `--num-params-b`。
