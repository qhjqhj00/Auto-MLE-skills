# Auto-MLE-skills

A growing collection of **Agent Skills for automating machine-learning engineering (MLE)**.

Each skill teaches an AI agent one hard, error-prone part of the ML workflow — the kind
of thing that silently wastes hours when an agent guesses instead of checking. The agent
runs a deterministic script to gather **facts**, then uses bundled reference matrices to
turn those facts into a **verdict with concrete next steps**.

[中文说明 →](README.zh-CN.md)

## Design philosophy

Every skill in this repo follows the same four rules:

1. **Detection is a script's job; judgment is the agent's job.** A pure-stdlib script
   produces a deterministic, machine-readable manifest (JSON). The agent never *guesses*
   facts (GPU model, VRAM, CUDA version, model architecture) — it reads them.
2. **Knowledge lives in `references/`, not in prose.** Compatibility matrices and
   heuristics go stale fast (new PyTorch every few months, new GPUs constantly). Keeping
   them in dated reference files makes them cheap to update.
3. **Verdicts are tiered and always carry a fallback.** Not "yes/no" but
   ✅ works / ⚠️ works-but-watch-out / ❌ won't work — and when it's ⚠️/❌, the skill gives
   an equivalent substitute or a concrete fix, not just a refusal.
4. **Output language follows the user's query language.** Chinese question → Chinese
   report; English question → English report. The JSON manifest stays English so
   downstream programs can consume it.

The skills are also **composable**: one skill's JSON output can feed the next (e.g. the
VRAM estimator reads the env probe's `env_report.json` to know how much memory it's
working with).

## Skills

### 🧰 `environment/` — Layer 1: environment & dependencies

The biggest pain point in ML engineering, and where agents trip most often. Probe the
machine *first*, then tell the agent what it can and can't run.

| Skill | What it does |
|-------|--------------|
| [**ml-env-probe**](environment/ml-env-probe/) | Probe hardware/software (NVIDIA, Apple MPS, CPU, TPU, and Chinese accelerators — 昇腾/寒武纪/海光/沐曦) and produce a ✅/⚠️/❌ capability report: which torch/CUDA combo to use, why flash-attn/xformers won't build, how to isolate the env, how to split across GPUs. |
| [**ml-vram-estimator**](environment/ml-vram-estimator/) | Estimate VRAM *before* launch so a run doesn't OOM hours in. Breaks down weights + KV-cache + gradients + optimizer + activations, handles GQA / MoE / hybrid attention / quantization, and returns a fits/tight/OOM verdict with fallbacks. |

### 📦 `data/` — dataset preparation

Framework-neutral: one well-prepared dataset serves any downstream trainer.

| Skill | What it does |
|-------|--------------|
| [**ml-data-prep**](data/ml-data-prep/) | Convert between alpaca / sharegpt / openai-messages, check data quality (empty/duplicate/malformed samples, length distribution, truncation risk, label alignment), and inspect tokenization & packing — catching the *silent* failures (truncated answers, wrong label masking, missing EOS, double-BOS, cross-document packing contamination) that don't error but quietly hurt the model. |

## Roadmap

| Category | Status | Scope |
|----------|--------|-------|
| `environment/` | ✅ available | env & dependency detection, VRAM estimation |
| `data/` | ✅ available | dataset format conversion, quality checks, tokenization/packing |
| `training/` | planned | training config, launch, distributed orchestration, hyperparameter sanity |
| `harness/` | planned | experiment scaffolding: logging, checkpoint/resume, monitoring, run management |
| `evaluation/` | planned | benchmark setup, metric pipelines |
| `serving/` | planned | deployment & inference serving |

## Using a skill

Each skill is a self-contained directory with a `SKILL.md` (the agent-facing instructions),
a `scripts/` entry point you can also run by hand, and `references/`.

**With Claude Code / an agent:** copy the skill directory into your skills location
(e.g. `~/.claude/skills/` or a project's `.claude/skills/`), then invoke it by name or
let the agent auto-trigger from the description.

**By hand** (the scripts are pure Python stdlib, no install needed):

```bash
# detect the environment
python3 environment/ml-env-probe/scripts/detect_env.py --out env_report.json

# estimate VRAM for a model + scenario, judged against that environment
python3 environment/ml-vram-estimator/scripts/estimate_vram.py \
    --model /path/to/model --mode inference --batch 1 --seq 8192 \
    --from-env env_report.json
```

## Repo layout

```
Auto-MLE-skills/
├── README.md  README.zh-CN.md
├── LICENSE
└── <category>/
    └── <skill-name>/
        ├── SKILL.md        # agent-facing instructions (frontmatter: name + description)
        ├── README.md       # human overview
        ├── scripts/        # deterministic entry point(s), pure stdlib
        ├── references/     # compatibility matrices & heuristics (dated)
        └── examples/       # real sample outputs
```

## Contributing a new skill

Follow the four design rules above and the layout convention. A good skill:
- has a script that produces **facts** (JSON), never guesses;
- puts version-sensitive knowledge in `references/*.md` with a `last updated:` line;
- gives **tiered verdicts with fallbacks**, not binary answers;
- adapts output to the user's query language;
- ships at least one real example in `examples/`.

## License

[Apache License 2.0](LICENSE).
