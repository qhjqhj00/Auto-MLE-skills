# environment/ — Layer 1: environment & dependencies

第一层:环境与依赖管理。Probe the machine *before* doing any ML work, then tell the
agent what it can and can't run. This is the biggest pain point in ML engineering and
where agents trip most often (照抄 README 式安装在某块卡/某个架构上撞墙).

| Skill | 作用 / What |
|-------|------------|
| [ml-env-probe](ml-env-probe/) | 探测软硬件 → ✅/⚠️/❌ 能力清单(torch/CUDA 组合、编译包能否装、环境隔离、多卡切分)。Detect hardware/software → capability report. |
| [ml-vram-estimator](ml-vram-estimator/) | 启动前估算显存,避免 OOM。Estimate VRAM before launch to avoid OOM. |

**串联用法 / Composed:** `ml-env-probe` 产出的 `env_report.json` 可直接喂给
`ml-vram-estimator --from-env`,形成"探测环境 → 估算能否跑"的闭环。
