# 数据集格式:alpaca / sharegpt / openai-messages

> last updated: 2026-01。三种主流 SFT/chat 格式的规范、互转坑、以及各框架的隐性要求。

互转用 `scripts/convert_format.py`(走规范化中间表示:任意格式 → IR → 任意格式)。

## 三种格式

### Alpaca(单轮为主)
```json
{"instruction": "...", "input": "(可选)", "output": "...", "system": "(可选)"}
```
- `input` 是给 instruction 的补充材料,常拼成 `instruction + "\n\n" + input` 作为用户内容。
- **天生表达不了多轮对话**。LLaMA-Factory 扩展了非标准的 `history` 字段:
  `"history": [["user1","assistant1"], ["user2","assistant2"]]`,最后的 instruction/output
  是要训练的那一轮。别的框架不一定认 `history`。

### ShareGPT(多轮)
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
 "system": "(可选,也可放进 conversations 里 from=system)"}
```
- 角色名是 `human` / `gpt`(不是 user/assistant!),system 有时在顶层、有时在数组里。
- 也有变体用 `from: function_call/observation/tool` 表示工具调用(v1 转换会丢弃并告警)。

### OpenAI messages(事实标准)
```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```
- 角色 `system/user/assistant/tool`;支持 `tool_calls`、多模态 `content` 为 parts 列表。
- 现在是最通用的格式,新框架几乎都直接吃它。**不确定转哪个就转它。**

## 互转的坑(脚本已处理,但要知道)

| 坑 | 说明 |
|----|------|
| **角色名映射** | human↔user,gpt↔assistant,system↔system。sharegpt 用 human/gpt,转错会让模板对不上。 |
| **多轮 → alpaca 有损** | alpaca 原生单轮;脚本用 LLaMA-Factory 的 `history` 承载,但这是非标准字段,告警提示。 |
| **system 位置** | messages 放在 messages[0];sharegpt 可顶层或数组内;alpaca 单独字段。统一收进 IR 的 system。 |
| **轮次顺序与交替** | 必须保持 user/assistant 严格交替、顺序不乱(见 quality-checks.md 的交替检查)。 |
| **工具调用 / 多模态** | v1 IR 只建模纯文本轮;tool_calls / 图文 parts 会被丢弃并告警,需要时单独处理。 |
| **JSON 数组 vs JSONL** | 大数据用 JSONL(逐行,流式友好);小数据用 JSON 数组。脚本读时自动识别,写用 `--out-format`。 |

## 各框架的隐性要求(最容易踩)

| 框架 | 吃什么格式 | 隐性要求 |
|------|-----------|---------|
| **LLaMA-Factory** | alpaca / sharegpt | 必须在 `dataset_info.json` 里注册数据集,声明列名与角色 tag(`role_tag`/`content_tag`);sharegpt 的 from 值要和配置里的 tag 对上 |
| **axolotl** | sharegpt / alpaca / chat_template | 配置里指定 `type`(如 `sharegpt`、`chat_template`)和字段映射;新版推荐 `chat_template` + messages |
| **trl `SFTTrainer`** | messages(conversational)或 `text`(已套模板) | 给 messages 会**自动套 chat template**;给 text 则按原样,不再套模板(别重复套) |
| **OpenAI / 多数 API 微调** | messages 的 **.jsonl** | 必须 JSONL;每行一个 `{"messages":[...]}` |
| **unsloth** | messages / alpaca | 配合其 `apply_chat_template`/`to_sharegpt` 工具,注意和你的模板一致 |

**通用建议:转成 OpenAI messages 的 JSONL** 兼容面最广;再按目标框架的列名要求做最后微调。
转换后务必用 `check_quality.py` 复查,再用 `inspect_tokenization.py` 确认套模板后的样子。
