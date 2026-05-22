# 推理显存 & KV cache 深入

> last updated: 2026-01。推理 = 权重 + KV cache + 少量激活。OOM 几乎总是 KV cache。

## KV cache 公式(最关键)

```
KV bytes = 2 × num_kv_heads × head_dim × num_layers × batch × seq_len × dtype_bytes
           └K和V┘                                    └────随这两个线性增长────┘
```

三个最常算错的点:

### 1. GQA / MQA —— 用 kv_heads,不是 attention_heads
分组查询注意力下,K/V 头数 `num_key_value_heads` 远小于 `num_attention_heads`。
用错会把 KV cache 高估几倍。例:Qwen3.6-35B 是 16 个 attn 头但只有 **2 个 kv 头** → KV 小 8 倍。

### 2. 混合注意力 —— 只算 full-attention 层
有些模型(Qwen3.5/3.6 MoE、Jamba、部分 Mamba 混合)的 `layer_types` 里大部分是
**linear_attention / mamba / ssm**:它们是**常数大小状态**,不随 seq 增长!只有
`full_attention` 层有随 seq 增长的 KV cache。
- 例:Qwen3.6-35B 40 层里仅 **10 层 full attention** → KV cache 只按 10 层算。
  按 40 层算会高估 4 倍。脚本读 `layer_types` 自动处理。
- 线性/SSM 层另有一份小的常数状态(conv_kernel × ssm dims),量级很小,脚本未计入,
  长序列下可忽略,但要知道它存在。

### 3. 长上下文下 KV cache 会超过权重
KV 随 seq 线性涨。长 context(128k+)时,KV cache 可能比权重还大。
- 估算口诀:先看权重,再算 KV;问"我要的 batch×seq 下 KV 是多少",别只看模型大小。

## 砍 KV cache 的手段(OOM 退路)

| 手段 | 效果 | 代价 |
|------|------|------|
| 降 batch / seq | 线性降 | 吞吐/上下文变小 |
| `--kv-dtype fp8` | KV 减半 | 轻微精度损失 |
| GQA/MQA 模型 | 天生小 | 选模型时就定了 |
| PagedAttention (vLLM) | 减碎片浪费、按需分配 | 用 vLLM 而非裸 transformers |
| 滑动窗口 / 局部注意力 | 上界封顶 | 模型需支持 |

## 权重(推理)

- 取真实磁盘大小最准(脚本读 safetensors)。量化模型(fp8/int4)的磁盘大小已含 scale 开销。
- 量化推理:fp8 ≈ 参数量×1B;int4/nvfp4 ≈ 参数量×0.5B + scale,实际看磁盘。
- **MoE**:所有专家权重都要常驻显存(总参,如 35B),即使每 token 只激活少数(3B)。
  显存按总参算,算力按激活参算 —— 别把"激活 3B"误当显存只要 3B。

## 推理激活

相对小:解码阶段每步只过一个 token,激活 ≈ batch×hidden 量级;prefill 阶段过整个
prompt,激活 ≈ batch×seq×hidden。脚本给保守近似;真正吃显存的是权重和 KV。

## 框架差异

- **vLLM / SGLang / TGI**:会预分配大块显存做 KV cache 池(`gpu_memory_utilization`
  默认 0.9)。所以"启动就占 90%"是正常的池化,不是真用满。估算这类服务时关注
  KV cache 池能容纳的总 token 数,而非单请求。
- **裸 transformers `generate`**:按需分配,峰值 = 权重 + 当前 batch 的 KV + 激活。
