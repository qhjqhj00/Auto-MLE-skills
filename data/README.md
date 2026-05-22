# data/ — dataset preparation

数据准备层。框架中立(framework-neutral):一份做好的数据服务所有下游 trainer。
The data layer is largely trainer-agnostic, so one well-prepared copy serves everything
downstream (LLaMA-Factory / axolotl / trl / …).

| Skill | 作用 / What |
|-------|------------|
| [ml-data-prep](ml-data-prep/) | 格式互转(alpaca/sharegpt/openai)+ 质量检查(空/重/坏/超长/标签对齐)+ tokenize 打包体检(模板/EOS/BOS/label mask/packing)。Format conversion + quality checks + tokenization/packing inspection. |

**重点 / Focus:** 那些**不报错但悄悄伤效果**的静默失败——答案被 max_len 截断、label mask
错(把 prompt 也训了)、缺 EOS、双 BOS、packing 跨文档污染。Silent failures that don't
raise an error but quietly hurt the model.
