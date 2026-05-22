---
name: ml-data-prep
description: >-
  Prepare SFT/chat datasets for training: convert between alpaca / sharegpt /
  openai-messages formats, check data quality, and verify tokenization & packing
  correctness — the framework-neutral data layer that serves any downstream
  trainer. Catches the silent failures that don't error but quietly hurt the
  model: empty/duplicate/malformed samples, answers truncated at max_len, wrong
  label masking (training on the prompt), missing EOS (model won't stop),
  double-BOS, and packing without a block-diagonal mask (cross-document
  contamination). Use when the user mentions: dataset format conversion, alpaca,
  sharegpt, openai messages / jsonl, data quality, dedup, length distribution,
  truncation, chat template, tokenize, packing, padding, label mask, loss mask,
  attention mask, preparing data for SFT/fine-tuning, LLaMA-Factory / axolotl /
  trl data format. Triggers: "数据格式", "格式转换", "alpaca", "sharegpt", "messages",
  "数据质量", "去重", "截断", "chat template", "packing", "label mask", "tokenize",
  "数据清洗", "微调数据", "训练数据准备".
---

# ml-data-prep — 训练数据准备(格式 / 质量 / tokenize 打包)

框架中立的数据层:一份做好的数据能服务所有下游 trainer。专治那些**不报错但悄悄伤效果**
的问题——空/重复/坏样本、答案被 max_len 截断、label mask 错(把 prompt 也训了)、缺 EOS
(模型不停)、双 BOS、packing 不做块对角(跨文档污染)。

## 核心心法

1. **脚本干活,你做裁决和解释。** 三个纯 stdlib 脚本(tokenize 那个可选用 transformers)
   产出确定性结果;你把 flags 解读成"改哪里、怎么改"。
2. **三档裁决,带退路:** ✅ clean / ⚠️ review / ❌ will-harm-training。❌ 必须指出哪些样本、
   怎么修(丢弃/补全/去重/调 max_len)。
3. **重点防"静默失败":** loss 照样下降但模型学坏的那些坑(截断答案、label mask、packing
   污染)是这个 skill 的主战场,优先排查。
4. **输出语种跟随用户的提问语种。** 给人看的结论用 query 语种;脚本 JSON 保持英文。

## 三个能力 = 三个脚本

| 能力 | 脚本 | 何时用 |
|------|------|--------|
| **格式互转** | `scripts/convert_format.py` | alpaca↔sharegpt↔openai;统一成下游框架要的格式 |
| **质量检查** | `scripts/check_quality.py` | 训练前体检:空/重/坏/超长/标签对齐 |
| **tokenize 打包** | `scripts/inspect_tokenization.py` | 套模板后到底长啥样:模板/EOS/BOS/label mask/packing |

## 工作流(推荐顺序)

### Step 1 — 统一格式
```bash
python3 scripts/convert_format.py --input raw.json --to openai --out data.jsonl
```
- `--from auto` 自动识别;`--to openai|sharegpt|alpaca`;`--out-format jsonl|json`。
- 不确定转哪个 → **openai messages 的 jsonl**,兼容面最广(各框架隐性要求见 references/formats.md)。
- stderr 的报告会提示多轮转 alpaca 这类有损操作。

### Step 2 — 质量体检
```bash
python3 scripts/check_quality.py --input data.jsonl --max-len 4096 --tokenizer /path/to/model
```
- 读 `verdict` + `flags` + `issues`。**❌ 时按 references/quality-checks.md 给出每类问题的修法**
  并点名数量(如"3 条空回复 → 丢弃")。
- 有 tokenizer 就传,token 长度比字符数准得多(截断判断靠它)。
- 长度分布(p50/p90/p99/max)用来选 max_len、判断要不要 packing(联动 ml-vram-estimator 看显存)。

### Step 3 — tokenize / 打包体检
```bash
python3 scripts/inspect_tokenization.py --tokenizer /path/to/model --input data.jsonl
```
- 逐条核对 flags,**重点看 `label-mask-unknown` 和 `packing-attention`**——静默掉点的常客。
- 没装 transformers 会打印手动清单并指回 references/tokenize-packing.md(优雅降级)。
- 把"套模板后的样子"(rendered_preview)念给用户看,确认和推理时一致。

## 输出模板:DATA_REPORT.md(语种跟随 query)

```markdown
# 数据准备报告 — <数据集>

- 来源格式: <detected> → 目标格式: <to>(<n> 条,<multiturn> 条多轮)
- 质量裁决: <✅/⚠️/❌>  长度 p50/p99/max = <…>(单位 token/char)
- tokenize: 模板 <ok/手拼>,EOS <ok/缺>,label mask <自动/需手动>,packing <注意块对角>

## 必修(❌/⚠️ 项,带数量与修法)
| 问题 | 数量 | 修法 |
|------|------|------|
| 空回复 | 3 | 丢弃 |
| 超长截断(答案在尾) | 12% | max_len 4096→8192 或截 prompt 侧 |
| label mask 模板不支持 | — | 用 DataCollatorForCompletionOnlyLM |

## 建议
- max_len 取 <p99 附近的值>;p50≪max_len → 开 packing(注意块对角注意力 + position_ids 重置)
```

## 边界

- v1 IR 只建模纯文本对话轮;**工具调用 / 多模态 parts 会被丢弃并告警**,需要时单独处理。
- 多轮 → alpaca 是有损的(用非标准 `history` 字段);优先转 sharegpt / openai 保多轮。
- 质量阈值是经验默认(`--dup-warn/--trunc-warn/--trunc-error`),按数据规模与任务调整。
- packing 的块对角注意力**正确性依赖训练框架与 attn 实现版本**;脚本能指出风险点,但最终
  要在你的框架/版本里核实(老版 trl 的裸 concat 会污染)。
