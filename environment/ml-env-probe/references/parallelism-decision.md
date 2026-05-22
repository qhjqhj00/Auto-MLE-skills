# 分布式 / 并行策略决策树

> last updated: 2026-01。输入:`accelerator.count`、`interconnect`(NVLink/PCIe/多机)、
> 单卡显存、模型参数量。输出:推荐 DP/FSDP/TP/PP 的切法。

## 先建立直觉:四种并行

| 切法 | 切什么 | 通信量 | 何时用 |
|---|---|---|---|
| **DP / DDP** | 复制整个模型,切数据 | 每步 all-reduce 梯度 | 模型能塞进单卡 |
| **FSDP / ZeRO** | 切参数+梯度+优化器状态(数据并行的省显存版) | all-gather/reduce-scatter | 模型勉强塞不下单卡,但全机能放下 |
| **TP (张量并行)** | 切单层内的矩阵 | **每层多次 all-reduce**,极吃带宽 | 单层都放不下 / 要降单卡显存,**需高带宽互联** |
| **PP (流水线并行)** | 按层切成 stage | stage 间点对点传 activation | 跨节点扩展;有 bubble,需 micro-batch |

## 决策树

```
单卡 (count == 1)?
├─ 模型放得下显存 → 直接单卡,无需并行
└─ 放不下 → 量化(4/8bit)/ device_map="auto" 分层到 CPU / offload(慢)
            真不行才上多卡

多卡同机 (count > 1, 单机)?
├─ 模型能塞进单卡 → 优先 DDP(吞吐最高);要省显存上 FSDP
└─ 模型塞不进单卡:
   ├─ 有 NVLink (interconnect.nvlink == true) → TP(卡内带宽够,TP 才划算)
   │     张量并行度 = 刚好放下模型的最小卡数(常 2/4/8)
   └─ 只有 PCIe (topology == multi-gpu-pcie) → TP 会被带宽拖死
         → 优先 FSDP(通信更稀);或 PP(按层切,通信量小)

多机 (multi_node)?
└─ 分层组合:节点内 TP(走 NVLink)× 节点间 PP 或 DP/FSDP(走网络 IB/RoCE)
   先确认机间网络(InfiniBand/RoCE vs 普通以太网);普通以太网别做跨机 TP
```

## 关键判断点

- **NVLink 是 TP 的前提**。`nvidia-smi topo -m` 里有 `NV#` 才是 NVLink;全是 `SYS`/`PHB`/
  `PXB` 就是 PCIe,此时 TP 的 all-reduce 会把训练拖到很慢 → 改 FSDP/PP。
  (探测脚本已在 `interconnect.nvlink` 给出结论。)
- **模型放得下吗?** 粗算:`参数量 × 每参字节`。推理 fp16≈2B/参;训练还要梯度+优化器
  状态(Adam ≈ 16B/参 with fp32 states,或混合精度下约 12–16B/参)。
  例:7B 模型 fp16 推理 ≈ 14GB;全量 fp16+Adam 训练 ≈ 100GB+ → 必须切。
- **TP 度数**:取"刚好放下"的最小 2 的幂,且 ≤ 单节点卡数(别跨节点做 TP,除非有
  超高速互联)。
- **多机网络**:`ibstat` / `ip link` 看有没有 InfiniBand / RoCE。没有高速网就别指望
  跨机 TP/大规模 FSDP,优先把并行度压在节点内。

## 框架到策略的对应

| 想要 | 用什么 |
|---|---|
| DDP / 单机多卡数据并行 | `torchrun` + `DistributedDataParallel` |
| FSDP / ZeRO 省显存 | PyTorch FSDP、DeepSpeed ZeRO-2/3、`accelerate` |
| TP / PP 大模型训练 | Megatron-LM、`torch.distributed` 的 device mesh、nanotron |
| 大模型推理切分 | vLLM(`tensor_parallel_size` / `pipeline_parallel_size`)、TGI、SGLang |
| 只想快速跑起来 | `accelerate launch`(自动配 DDP/FSDP) |

## 非 NVIDIA 平台

- **昇腾多卡**:HCCL(对应 NCCL),`torch_npu` + DeepSpeed/Megatron 的昇腾适配分支。互联看 HCCS。
- **TPU**:JAX `pmap`/`shard_map` 或 torch_xla SPMD,TPU pod 切分逻辑不同(看 mesh)。
- **Apple/CPU**:无多卡并行可言;Apple 单芯片,CPU 走 `gloo` 后端做多进程 DDP 意义有限。

判定原则同上:**先看能不能塞进单设备,再看互联带宽决定切法。**
