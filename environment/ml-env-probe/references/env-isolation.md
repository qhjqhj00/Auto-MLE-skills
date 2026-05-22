# 环境隔离与依赖冲突

> last updated: 2026-01。选哪个工具 + 怎么建 + ML 包之间的常见冲突对。

## 选工具:uv / conda / venv

| 工具 | 何时用 | 优点 | 注意 |
|---|---|---|---|
| **uv** | 默认首选(纯 pip 生态) | 极快、锁文件可复现、`uv pip` 兼容 | 不管理非 Python 系统库(CUDA toolkit/编译器) |
| **conda/mamba** | 需要系统级非 Python 依赖(CUDA toolkit、cuDNN、MKL、GDAL 等) | 能装编译器/系统库 | 慢、易把 channel 搞乱;`base` 别装业务包 |
| **venv** | 最小依赖、不想装额外工具 | stdlib 自带 | 功能最少 |

经验法则:**纯 Python+torch wheel → uv;需要系统 CUDA toolkit 来源码编译 → conda
装 toolkit/编译器,再在其中用 pip/uv 装 Python 包。** 别把 conda 装的 CUDA 和 pip
wheel 自带的 CUDA 混着依赖。

## 标准建环境流程

### uv(推荐)
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # 先 torch
uv pip install transformers accelerate datasets                          # 再上层
# 编译包最后,关 isolation:
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```
> Python 版本选 **3.10–3.12** 最稳(3.13 还有包没出 wheel,见探测旗标 `python-too-new`)。

### conda(需系统 CUDA toolkit 时)
```bash
conda create -n proj python=3.11 -y && conda activate proj
# 只在确需源码编译时装 toolkit;否则 torch wheel 自带运行时就够
conda install -c nvidia cuda-toolkit=12.8 -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### venv
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 安装顺序(铁律)

1. **先 torch**(锁定版本与 CUDA tag)。
2. 再装上层框架(transformers / accelerate / datasets / peft …)。
3. **最后**装需编译的 kernel 包(flash-attn / apex / xformers),`--no-build-isolation`。

颠倒顺序 → 上层包或 build isolation 会拉一个不同的 torch,ABI 崩。

## 常见冲突对(ML 生态高频静默 bug)

| 冲突 | 症状 | 处理 |
|---|---|---|
| **transformers ↔ tokenizers** | 每个 transformers 锁一段 tokenizers 区间;单独升级其一报 `tokenizers>=x,<y` 或运行时怪错 | 让 pip 解析 transformers 的依赖,不要手动钉死 tokenizers |
| **numpy 2.x ↔ 老编译包** | import 报 `_ARRAY_API not found` / `numpy.dtype size changed` | 老包未重编时 `pip install "numpy<2"` |
| **torch ↔ xformers/flash-attn** | `undefined symbol`、import 崩 | 三者用同一 CUDA tag;编译包 `--no-build-isolation` |
| **CUDA wheel ↔ conda CUDA** | 运行时找错 libcudart | 二选一来源,别混依赖 |
| **transformers ↔ accelerate/peft/trl** | 训练 API 变动报错 | 同一时期版本一起升,看各自 release 的兼容声明 |
| **protobuf / grpcio 过新** | sentencepiece、TF 衍生工具报错 | 按报错钉 protobuf 版本区间 |

## 可复现

- uv:`uv pip compile` / `uv.lock` 产出锁文件,提交进仓库。
- conda:`conda env export --no-builds > environment.yml`(去掉 build 串更可移植)。
- 始终记录 `pip freeze` 快照,出问题能回滚。

## 给下游 agent 的建议

把"建环境命令"作为**可执行块**写进 `ENV_CAPABILITY.md`,但**不要擅自执行全局安装或
改 base 环境**,除非用户明确同意。优先建独立 env,避免污染 conda base。
