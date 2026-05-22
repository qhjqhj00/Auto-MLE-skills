# ml-data-prep

训练数据准备:格式互转 + 质量检查 + tokenize/打包体检。框架中立——一份做好的数据
服务所有下游 trainer(LLaMA-Factory / axolotl / trl / …)。

## 它解决什么

SFT 数据的坑大多**不报错**:空/重复样本、答案被 max_len 截断、label mask 错(把 prompt
也训了)、缺 EOS(模型不停)、双 BOS、packing 不做块对角(跨文档污染)。loss 照样下降,
但模型学坏。这个 skill 把这些静默失败摊到明面上。

## 三个能力

```bash
# 1. 格式互转(alpaca / sharegpt / openai-messages,走规范化中间表示)
python3 scripts/convert_format.py --input raw.json --to openai --out data.jsonl

# 2. 质量检查(空/重/坏/超长/标签对齐 → ✅/⚠️/❌ 裁决)
python3 scripts/check_quality.py --input data.jsonl --max-len 4096 --tokenizer /path/to/model

# 3. tokenize/打包体检(模板/EOS/BOS/label mask/packing,需 transformers 否则给清单)
python3 scripts/inspect_tokenization.py --tokenizer /path/to/model --input data.jsonl
```

## 算得对、查得出的地方

- **格式互转无损多轮**:经 openai→sharegpt→alpaca 往返,多轮 `history` 完整保留。
- **截断答案风险**:专门统计"超 max_len 且以 assistant 结尾"的样本(答案会被切 → 模型学不写完)。
- **label mask**:识别模板带不带 `{% generation %}`;不带就明确告诉你必须自己 mask prompt。
- **packing 块对角**:把边界处 position_ids 打出来对比——correct `[37,38,0,1,2]`(重置)
  vs naive `[37,38,39,40,41]`(继续累加,错)。

## 设计

与 ml-env-probe / ml-vram-estimator 同源:脚本产事实 → SKILL+reference 给裁决与修法;
三档裁决带退路;输出语种跟随 query;JSON 保持英文。

## 结构

```
ml-data-prep/
├── SKILL.md
├── scripts/
│   ├── convert_format.py        # alpaca/sharegpt/openai 互转(规范化 IR)
│   ├── check_quality.py         # 质量扫描 → flags + 裁决
│   └── inspect_tokenization.py  # tokenize/packing 体检(transformers 可选)
├── references/
│   ├── formats.md               # 三格式规范 + 互转坑 + 各框架隐性要求
│   ├── quality-checks.md        # 每项检查的含义/阈值/修法
│   └── tokenize-packing.md      # 静默杀手:模板/EOS/BOS/label mask/packing
└── examples/                    # 真实运行输出
```

> 纯 stdlib;只有 tokenize 体检在你需要真实 token 信息时用 `transformers`(+`jinja2`),
> 没装则优雅降级为手动清单。
