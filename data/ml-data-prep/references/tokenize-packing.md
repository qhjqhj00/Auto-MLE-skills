# Tokenize 与打包:不报错但悄悄伤效果的坑

> last updated: 2026-01。这一层最危险——错了**不会报错**,loss 照常下降,但模型学坏。
> `scripts/inspect_tokenization.py` 给一个 tokenizer + 样本,逐项体检。

## 1. Chat template 必须一致

- 训练时用 `tokenizer.apply_chat_template(...)` 套**模型自己的模板**,推理时用**同一个**。
  训练套模板 A、推理套模板 B(或手拼)→ 静默掉点,且很难查。
- 自己手拼 prompt 而不用官方模板,是常见的隐性不一致来源。
- 体检:看 `inspect_tokenization.py` 的 `rendered_preview`,确认特殊 token(如
  `<|im_start|>`、`<think>`)和你推理时一致。

## 2. EOS / BOS

- **缺 EOS**:assistant 轮末尾没有 EOS → 模型**永远学不会停**,生成时停不下来。
  确认模板在 assistant 内容后追加了 EOS。
- **双 BOS**:模板已经输出了 BOS,你又用 `tokenizer(text, add_special_tokens=True)`
  再加一个 → 序列开头两个 BOS,分布偏移。**对已套模板的文本用 `add_special_tokens=False`**。
- 体检:`special_tokens` 字段 + `double-bos-*` / `missing-eos` flag。

## 3. Label mask(loss 只算回复,别算 prompt)

- SFT 默认只对 **assistant 回复** token 算 loss,prompt/user token 设为 **-100**(忽略)。
  不 mask → 模型也在"学习生成用户的问题",通常不是你要的(除非刻意 train on inputs)。
- 怎么拿到 mask:
  - 模板带 `{% generation %}` 标记 → `apply_chat_template(..., return_assistant_tokens_mask=True)`
    能自动给出 assistant 区间。
  - **模板不带**(很多模型如此,Qwen 系当前就不带)→ transformers 无法自动 mask,你必须
    自己处理:`trl` 的 `DataCollatorForCompletionOnlyLM`、或 LLaMA-Factory `train_on_inputs=false`、
    或手动按 assistant 起始位置切。
- 体检:`label-mask` / `label-mask-unknown` flag 会告诉你模板支不支持。

## 4. Padding side

- **训练右 padding**(右补),**批量生成左 padding**(左补,否则 decode 错位)。
- pad token 缺失时常用 EOS 代替,但要确保 pad 位置的 label 也被 mask 成 -100,别算进 loss。
- 用 packing 时通常不需要 padding(见下)。

## 5. Packing —— 最隐蔽的杀手

把多条短样本拼进一个 `max_len` 块以省 padding、提吞吐。但拼接后:

- **注意力会跨样本**:朴素 concat 下,后一条样本能 attend 到前一条 → **跨文档污染**,
  模型学到本不该看到的上下文关联。**不报错**,只是悄悄变差。
  - 正解:**块对角注意力**——每条样本只能看自己。实现靠 FlashAttention 的 varlen
    (`cu_seqlens`)、或传入正确的 4D attention mask、或框架的 `position_ids` + 内部处理。
- **position_ids 必须按文档重置**:RoPE 等位置编码靠 position_ids。拼接后若位置继续累加
  (`...,37,38,39,40,...`),第二条样本的位置就错了;必须每条重置为 0
  (`...,37,38,0,1,...`)。`inspect_tokenization.py` 的 `packing-attention` flag 直接把这两种
  position_ids 在边界处打出来对比。
- 框架现状:`trl` 新版 `packing=True` 配合支持的 attention 实现会处理块对角;**老版本只是
  裸 concat,会污染**。务必核实你的版本与 `attn_implementation`(flash_attention_2 / 支持
  varlen 的实现)。LLaMA-Factory 的 `neat_packing` / `packing` 选项同理要确认。

## 6. 截断方向

- 截断要砍 **prompt 侧**,保住 assistant 答案。从右边一刀切常把答案截掉(见 quality-checks.md
  的 `answer_at_risk`)。长样本优先考虑提高 max_len 或丢弃,而不是盲目右截。

## 体检流程

```bash
python3 scripts/inspect_tokenization.py --tokenizer /path/to/model --input data.jsonl
# 没数据时用内置样本看模板与特殊 token
python3 scripts/inspect_tokenization.py --tokenizer /path/to/model --demo
```
需要 `transformers`(+ `jinja2` 套模板)。没装时脚本打印手动检查清单并指回本文件。
逐条核对 flag,尤其 `label-mask-unknown` 和 `packing-attention` —— 这两个是静默掉点的常客。
