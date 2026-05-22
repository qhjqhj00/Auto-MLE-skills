# ml-vram-estimator

事前估算 GPU 显存,避免跑起来几小时后才 OOM。给定模型 + 场景(推理/训练、batch、
seq、dtype、并行),算出每卡显存分项,给 ✅ fits / ⚠️ tight / ❌ oom 裁决和退路。

## 它解决什么

OOM 几乎总是栽在被忽略的项上 —— 推理是 **KV cache**,训练是**优化器状态**和**激活**。
naive"模型多大就要多少显存"会错得离谱。这个 skill 把每一项算清楚。

## 怎么用

```bash
# 推理:这模型在本机放得下吗(吃 ml-env-probe 的 env_report.json)
python3 scripts/estimate_vram.py --model /path/to/model --mode inference \
    --batch 1 --seq 8192 --from-env ../ml-env-probe/env_report.json

# 训练:全量微调要几张卡
python3 scripts/estimate_vram.py --config config.json --mode training \
    --batch 4 --seq 4096 --optimizer adamw --grad-ckpt --zero 3 --dp 8 \
    --available-vram 80
```

没有权重/config 时可纯手动:`--num-params-b 7 --hidden 4096 --layers 32 --kv-heads 8 --head-dim 128`。

## 算得对的地方(naive 估算最常翻车处)

- **GQA**:用 `num_key_value_heads`,不是 attention heads(可差几倍)。
- **混合注意力**:读 `layer_types`,KV cache 只算 full-attention 层。
  实测 Qwen3.6-35B(40 层仅 10 层全注意力)128k 上下文 KV cache 仅 **1.25GB**,
  naive 按 40 层 MHA 会算成 **~80GB**,差 64 倍。
- **MoE**:所有专家权重常驻显存(总参),不是只算激活参。
- **量化**:给 `--model` 时直接读磁盘 safetensors 大小,fp8/nvfp4 自动算对。

## 设计

与 ml-env-probe 同源:
- **脚本算(确定性算术) ↔ SKILL+reference 给裁决与退路**。
- **三档裁决,OOM 必带可执行退路**(降 batch/seq → grad-ckpt → 量化 → LoRA → FSDP → offload)。
- **输出语种跟随用户提问语种**;JSON 保持英文(机器读)。
- 能吃 ml-env-probe 的 `env_report.json` 直接对比可用显存,两个 skill 串成一条线。

## 结构

```
ml-vram-estimator/
├── SKILL.md
├── scripts/estimate_vram.py        # 唯一入口 → JSON breakdown + verdict
├── references/
│   ├── memory-breakdown.md         # 主公式:每一项 + 字节表 + 余量
│   ├── inference-and-kv.md         # KV cache 深入:GQA/混合注意力/MoE/量化/vLLM
│   ├── training-memory.md          # 优化器表 + 激活 + grad-ckpt + LoRA/QLoRA
│   └── parallelism-memory.md       # TP/PP/DP/ZeRO/FSDP 各切哪一项
└── examples/                       # Qwen35B 推理 + 7B 全量微调 OOM 的真实输出
```

> 估算是数量级 + 安全余量(默认 margin 1.1),不是字节级精确。临界场景先实测一步再外推。
