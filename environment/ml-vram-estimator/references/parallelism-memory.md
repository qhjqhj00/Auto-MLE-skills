# 并行如何分摊显存

> last updated: 2026-01。每种并行切的项不同。脚本据此对各分项做除法。
> 切法该选哪种,见 ml-env-probe 的 parallelism-decision.md(看 NVLink/显存/模型大小)。

## 各并行对各显存项的分摊

| 并行 | 权重 | 梯度 | 优化器 | KV/激活 | 通信 | 前提 |
|------|------|------|--------|---------|------|------|
| **DP / DDP** | ✗ 每卡全复制 | ✗ | ✗ | ✓ 切 batch | 梯度 all-reduce | 模型放得下单卡 |
| **TP 张量并行** | ✓ /tp | ✓ /tp | ✓ /tp | ✓ /tp | 每层多次 all-reduce(吃带宽) | **需 NVLink** |
| **PP 流水线** | ✓ /pp(按层) | ✓ /pp | ✓ /pp | 部分 | stage 间点对点 | 有 bubble,需 micro-batch |
| **ZeRO-1** | ✗ | ✗ | ✓ /dp | ✗ | | 省优化器 |
| **ZeRO-2** | ✗ | ✓ /dp | ✓ /dp | ✗ | | +省梯度 |
| **ZeRO-3 / FSDP** | ✓ /dp | ✓ /dp | ✓ /dp | ✗ | all-gather 权重 | 省最多 |

记法:✓ = 这项被该并行切小;✗ = 不切(每卡仍是全量或按 batch)。

## 关键直觉

- **DDP 不省单卡权重显存** —— 每卡一份完整模型副本,只是数据分到不同卡。模型本身放不下
  单卡时,DDP 没用,要上 FSDP/TP/PP。
- **FSDP / ZeRO-3 是省显存王牌**:权重+梯度+优化器全部按数据并行组 `dp` 切。
  全量微调放不下时的标准答案。`--zero 3 --dp N`。
- **TP 切得彻底但吃带宽**:权重/梯度/优化器/激活都按 `tp` 切,但每层多次 all-reduce
  → **没有 NVLink 就别用 TP**(PCIe 会被拖死)。适合单层都放不下、且卡间高带宽。
- **PP 通信最省**:按层切成 stage,只在 stage 边界传 activation。适合跨节点。代价是
  流水线 bubble + 要调 micro-batch 数。

## 组合(大规模训练常见)

`总卡数 = tp × pp × dp`。典型:
- 单机 8 卡有 NVLink:`tp=8`(模型切 8 份),或 `zero=3 dp=8`(FSDP)。
- 多机:节点内 `tp=8` 走 NVLink × 节点间 `pp` 或 `dp/zero3` 走 IB。
- 脚本的 `n_gpus_implied = tp×pp×dp`,核对是否与实际卡数一致。

## 用脚本验证切法

```bash
# 全量微调放不下单卡?试 FSDP across 8
--mode training ... --zero 3 --dp 8 --available-vram 80
# 模型本身就超单卡?试 TP(确认有 NVLink)
--mode inference ... --tp 4 --available-vram 80
```
看 `total_per_gpu_gb` 是否落进 ✅;不行就加并行度或叠加 grad-ckpt / 量化 / LoRA。

## 注意

- 估算对并行的建模是**一阶近似**:TP 通信缓冲、ZeRO 的临时 all-gather 峰值、PP 的
  micro-batch 在途激活,都会额外占一些,已包含在默认余量里。临界场景留更大 margin。
- offload(ZeRO-Offload / FSDP CPU offload)把状态挪到 CPU/NVMe,显存进一步降但变慢,
  脚本未单独建模 —— 作为最后退路在结论里提示。
