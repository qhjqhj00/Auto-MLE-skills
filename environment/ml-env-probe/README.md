# ml-env-probe

**第一层:环境与依赖管理。** ML 自动化跑之前,先探测真实环境,告诉 agent 它的"能与不能"。

## 它解决什么

照抄 README 式安装在某块卡 / 某个架构上撞墙,而 agent 自己意识不到。这个 skill 让
agent **先探测、再裁决**:给定 GPU 型号、架构、CUDA、Python,推导出能用哪套
torch/CUDA、哪些编译包(flash-attn/xformers/apex)能装、装不上时退到什么、多卡怎么切。

## 怎么用

```bash
python3 scripts/detect_env.py --out env_report.json
```

`detect_env.py` 是纯 stdlib、不会崩、覆盖 NVIDIA / Apple MPS / CPU / TPU / 昇腾 /
寒武纪 / 海光 / 沐曦 的探测脚本,输出一份带**风险旗标**的 JSON 清单。之后 `SKILL.md`
指导 agent 按旗标加载对应 reference,产出 ✅/⚠️/❌ 三档能力裁决 + 精确安装命令。

## 设计要点

- **探测(脚本,确定性) 与 裁决(agent + reference)分离** —— 事实不靠猜,知识好更新。
- **"能与不能"分三档,永远带退路** —— ✅ 一行装 / ⚠️ 编译命令+耗时+防 OOM+降级 / ❌ 给等价替代。
- **核心三元组** `compute_cap + arch + cuda` 驱动一切判断。

## 结构

```
ml-env-probe/
├── SKILL.md                      # 主编排:探测→裁决→隔离→并行
├── scripts/detect_env.py         # 唯一探测入口 → env_report.json
├── references/
│   ├── cuda-torch-matrix.md      # cc→CUDA→torch wheel + ARM/Blackwell 特例
│   ├── compiled-packages.md      # flash-attn/xformers/apex/bnb 编译条件+降级
│   ├── env-isolation.md          # uv/conda/venv 流程 + 冲突对
│   ├── parallelism-decision.md   # GPU数/NVLink/显存/模型 → TP/PP/DP/FSDP 决策树
│   └── chips/                    # ascend(深) / tpu / apple-mps / cambricon / hygon-dcu / metax
└── examples/                     # 真实机器(GB10)的样例输出
```

> 兼容性矩阵会过期,reference 文件标了 `last updated`。版本边界拿不准时用
> `pip index versions <pkg>` 或官方 matrix 复核。
