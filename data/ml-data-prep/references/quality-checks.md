# 数据质量检查:每项的含义、阈值、修法

> last updated: 2026-01。`scripts/check_quality.py` 扫数据,产出分档 flags + 总裁决
> ✅ clean / ⚠️ review / ❌ will-harm-training。下面解释每项为什么重要、怎么修。

裁决 = flags 里最严重的那档(error→❌,warn→⚠️,否则 ✅),与 ml-env-probe 同套路。

## 致命项(error → ❌,必须修否则伤训练)

| 检查 | 为什么致命 | 修法 |
|------|-----------|------|
| **no-assistant**(无回复轮) | 没有可学习的目标,纯噪声 | 丢弃这些样本 |
| **empty-assistant**(空回复) | 教模型"输出空" → 推理时提前停 / 输出空串 | 丢弃或补全回复 |
| **parse-errors**(解析失败) | 格式损坏,训练时可能整批崩或被静默跳过 | 修复 JSON / 统一格式 |

## 警告项(warn → ⚠️,看任务决定)

| 检查 | 影响 | 修法 |
|------|------|------|
| **exact-duplicates** | 重复样本被反复学 → 过拟合/记忆;默认 >1% 告警 | 精确去重(按内容哈希) |
| **near-duplicates** | 仅大小写/空白不同的近重复 | 归一化后去重 |
| **not-ending-on-assistant** | SFT 里最后一轮应是 assistant,否则没有可学的结尾 | 截到最后一个 assistant 轮 |
| **non-alternating-roles** | 连续两个同角色,多数 chat 模板假设严格交替,会套错 | 合并或修正轮次 |
| **truncation**(超 max_len) | 超长样本被截断;**若答案在尾部被切 → 模型学会不写完**;默认 >5% warn,>20% error | 提高 max_len / 截 prompt 侧保留答案 / 过滤超长 |
| **empty-user** | 空提问轮,语义可疑 | 检查来源 |
| **mixed-formats** | 一份数据里混了多种格式 | 先用 convert_format 统一 |

## 长度分布(info,但驱动很多决策)

报告 p50/p90/p99/max(有 tokenizer 用 token 数,否则字符数)。用途:
- **选 max_len**:看 p99/p95,在"覆盖绝大多数样本"与"显存可承受"间权衡(显存见 ml-vram-estimator)。
- **长尾**:max 远大于 p99 → 少数超长样本,要么单独处理要么过滤,别为它们把 max_len 拉太高浪费显存。
- **packing 收益**:若 p50 远小于 max_len,padding 浪费大 → 用 packing(见 tokenize-packing.md)。

## 截断为什么是隐形杀手

> 截断不报错,但若把 assistant 的答案截掉,模型在那些样本上看到的是"问题 + 半截答案",
> 等于训练它**不要写完整答案**。脚本的 `answer_at_risk` 专门统计"超长且以 assistant 结尾"
> 的样本数 —— 这些是真正危险的。

## 用法与阈值

```bash
python3 scripts/check_quality.py --input data.jsonl --max-len 4096
# 有 tokenizer 时用真实 token 长度(强烈建议,字符数只是粗略代理)
python3 scripts/check_quality.py --input data.jsonl --max-len 4096 --tokenizer /path/to/model
```
阈值可调:`--dup-warn 0.01`、`--trunc-warn 0.05`、`--trunc-error 0.20`。这些是经验默认,
按数据规模和任务调整;小数据集对重复更敏感,长上下文任务对截断更敏感。

## 检查通过之后

质量裁决 ✅/⚠️ 不代表 tokenize 也对。务必再跑 `inspect_tokenization.py` 确认套模板、
EOS、label mask、packing 这些**不报错但伤效果**的环节(见 tokenize-packing.md)。
