# Auto-MLE-skills

一套持续扩充的 **机器学习工程(MLE)自动化 Agent Skills 集合**。

每个 skill 教 AI agent 搞定 ML 流程里一个又难又容易出错的环节——那种 agent 一旦"凭印象
猜"就会悄悄浪费几小时的地方。做法统一:agent 先跑一个确定性脚本拿到**事实**,再用内置的
对照矩阵把事实变成**带具体下一步的裁决**。

[English →](README.md)

## 设计心法

repo 里每个 skill 都遵循同样四条:

1. **探测是脚本的事,裁决是 agent 的事。** 纯 stdlib 脚本产出确定性、机器可读的清单
   (JSON)。agent **绝不猜**事实(GPU 型号、显存、CUDA 版本、模型架构)——它只读。
2. **知识放在 `references/`,不焊死在正文里。** 兼容性矩阵和经验值过期很快(PyTorch 几个
   月一版、新卡不断),放进带日期的 reference 文件才好更新。
3. **裁决分档,永远带退路。** 不是"行/不行",而是 ✅ 能跑 / ⚠️ 能跑但有坑 / ❌ 跑不了;
   遇到 ⚠️/❌ 时给等价替代或具体修法,而不是只丢一句拒绝。
4. **输出语种跟随用户提问语种。** 中文问→中文报告,英文问→英文报告;JSON 清单保持英文,
   方便下游程序消费。

skill 之间还能**串联**:一个 skill 的 JSON 输出可喂给下一个(比如显存估算器读环境探测的
`env_report.json` 来知道有多少显存可用)。

## 已有 skills

### 🧰 `environment/` — 第一层:环境与依赖

ML 工程最大的痛点,也是 agent 最容易翻车的地方。先探测机器,再告诉 agent 它的能与不能。

| Skill | 做什么 |
|-------|--------|
| [**ml-env-probe**](environment/ml-env-probe/) | 探测软硬件(NVIDIA、Apple MPS、CPU、TPU,以及国产芯片——昇腾/寒武纪/海光/沐曦),产出 ✅/⚠️/❌ 能力清单:该用哪套 torch/CUDA、flash-attn/xformers 为啥编不过、怎么隔离环境、多卡怎么切。 |
| [**ml-vram-estimator**](environment/ml-vram-estimator/) | 启动前估算显存,避免跑几小时才 OOM。拆解 权重+KV cache+梯度+优化器+激活,正确处理 GQA/MoE/混合注意力/量化,给 fits/tight/OOM 裁决与退路。 |

### 📦 `data/` — 数据准备

框架中立:一份做好的数据服务所有下游 trainer。

| Skill | 做什么 |
|-------|--------|
| [**ml-data-prep**](data/ml-data-prep/) | alpaca / sharegpt / openai-messages 互转,质量检查(空/重复/坏样本、长度分布、截断风险、标签对齐),tokenize 与打包体检——专抓那些**不报错但悄悄伤效果**的坑(答案被截断、label mask 错、缺 EOS、双 BOS、packing 跨文档污染)。 |

## 路线图

| 类别 | 状态 | 范围 |
|------|------|------|
| `environment/` | ✅ 已有 | 环境与依赖探测、显存估算 |
| `data/` | ✅ 已有 | 数据集格式转换、质量检查、tokenize/打包 |
| `training/` | 计划中 | 训练配置、启动、分布式编排、超参合理性 |
| `harness/` | 计划中 | 实验脚手架:日志、checkpoint/续训、监控、run 管理 |
| `evaluation/` | 计划中 | 评测基准搭建、指标流水线 |
| `serving/` | 计划中 | 部署与推理服务 |

## 怎么用

每个 skill 是自包含目录:`SKILL.md`(给 agent 的指令)、`scripts/`(也可手动跑的入口)、
`references/`。

**配合 Claude Code / agent:** 把 skill 目录拷到 skills 位置(如 `~/.claude/skills/` 或项目的
`.claude/skills/`),按名调用或让 agent 凭 description 自动触发。

**手动跑**(脚本纯 stdlib,无需安装):

```bash
# 探测环境
python3 environment/ml-env-probe/scripts/detect_env.py --out env_report.json

# 针对模型+场景估算显存,并对照刚探到的环境裁决
python3 environment/ml-vram-estimator/scripts/estimate_vram.py \
    --model /path/to/model --mode inference --batch 1 --seq 8192 \
    --from-env env_report.json
```

## 贡献新 skill

遵循上面四条心法和目录约定。一个好 skill:
- 有一个产出**事实**(JSON)的脚本,绝不猜;
- 把版本敏感的知识放进带 `last updated:` 的 `references/*.md`;
- 给**分档裁决 + 退路**,不是二元答案;
- 输出跟随用户提问语种;
- 在 `examples/` 里至少带一份真实示例。

## 许可证

[Apache License 2.0](LICENSE)。
