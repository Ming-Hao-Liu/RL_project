# 代码使用与复现

本文集中维护安装、运行、验证、分析和版本对应。项目背景与主要结论见[README](../README.md)，接口与依赖见[架构说明](architecture.md)。**以下命令均在项目根目录执行**，代码块中的路径相对项目根目录，不相对本文件。正式研究已经冻结，复现使用新输出目录，不覆盖现有原始记录。

## 环境与测试

需要 uv 和 Python 3.11。在项目根目录执行，环境、缓存和临时文件均留在项目内：

```sh
mkdir -p .cache/uv .cache/python .cache/matplotlib .tmp
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv sync --locked --group dev
uv run pytest
```

项目的 `.venv` 可能依赖 `.cache/python` 中的解释器，清理缓存时保留该目录。

## 执行与复现

生成新任务、执行或续跑、验证策略、生成单组分析：

```sh
uv run rl-retest prepare --config configs/study.json --suite dev --out artifacts/dev
uv run rl-retest run --run-dir artifacts/dev --workers 1
uv run rl-retest verify --run-dir artifacts/dev
uv run rl-retest analyze --run-dir artifacts/dev
```

正式运行使用 `main`、`large` 或 `stress` suite，并为每次新运行指定新的输出目录。`run` 检查冻结源码和配置，续跑只补齐未完成的作业；不要覆盖已有正式记录。`pruning` 是补齐原Proposal剪枝成本对照的独立运行组，复用large中的20任务，不并入原三组基准统计；命令见下节。

旧三组数据的严格复算使用其生成代码版本，避免改写来源摘要：

```sh
mkdir -p .tmp/original
git archive 3382745 | tar -x -C .tmp/original
PYTHONPATH="$PWD/.tmp/original/src" uv run python -m rl_retest.cli verify --run-dir data/main
PYTHONPATH="$PWD/.tmp/original/src" uv run python -m rl_retest.cli verify --run-dir data/large
PYTHONPATH="$PWD/.tmp/original/src" uv run python -m rl_retest.cli verify --run-dir data/stress
```

新增初始化确认实验使用独立的配置与数据目录：

```sh
uv run python -m scripts.confirm_planning --out artifacts/planning-reproduction
uv run python -m scripts.analyze_planning --source data/planning --out results/planning
```

已有确认数据的复算：`uv run python -m scripts.verify_planning`。上述运行命令创建复现副本，不把重复运行计为新样本。

确认实验的配置见 `configs/planning.json`。保存清单严格核对源代码与生成提交；从零重建可先检出该清单记录的提交并使用新的输出目录，不覆盖已有数据。分析脚本可以在当前版本读取冻结记录。

重新生成当前统计与机制结果：

```sh
uv run python -m scripts.analyze_study --source data --out results/study
uv run python -m scripts.analyze_history --source data --out results/history
uv run python -m scripts.analyze_sequential --source data --out results/sequential
uv run python -m scripts.diagnose_tolerance --source data --out results/search
uv run python -m scripts.analyze_search --source data --out results/search
uv run python -m scripts.review_evidence --source data --out results/validation
```

前三项读取正式记录；序贯分析还检查固定解析例子的成本边界。容差诊断在既有任务上重新规划，搜索分析读取该诊断结果。重复复现不增加独立样本数。

## 剪枝成本消融

本组使用既有20任务、原搜索和固定M=10000、ε=0.01，仅切换显式剪枝。详细计时、统计和停止规则见[实验协议](protocol.md)。保持环境与线程设置同上；策略运行和计时均使用一个进程，受控计时期间不同时运行本项目其他实验。

以下用于首次建立正式补充数据；已有目录时跳过prepare，用run/time续跑缺失作业和完整计时配对：

```sh
uv run rl-retest prepare --config configs/study.json --suite pruning --out data/pruning
uv run rl-retest run --run-dir data/pruning --workers 1
uv run rl-retest verify --run-dir data/pruning
uv run python -m scripts.pruning time --source data/pruning --out data/pruning/timing.json
uv run python -m scripts.pruning analyze --source data/pruning --out results/pruning
```

本组已完成；现有结果直接读取或重建分析，不重复运行。代码冻结版本为`b39a11c`。prepare/run/verify复用原CLI；time先核对与原large的任务及开启剪枝结果，再保存冻结顺序和120条测量；analyze只读取保存记录，不重新规划。复现分析可以把最后一条的输出改为`artifacts/pruning-analysis-reproduction`。从零复现使用新目录，如`artifacts/pruning-reproduction/pruning`，并在其父目录建立指向原`data/large`的`large`符号链接，供输入与旧结果核对。新一轮计时应使用独立输出文件，不能覆盖正式测量；同一文件的续跑只补缺失配对。完整策略、来源清单和原始计时位于data/pruning，配对表、统计、诊断图和核验输出位于results/pruning。它们不增加原任务数，也不替换原large的计时字段。

## 数据与计算契约

测试预算限制环境交互次数；规划预算统计实际后继信念更新。规划器返回完整不可变策略，评价器执行其中保存的终止路径，不重新优化。真实条件模型与规划模型分别传入。

正式种子、bootstrap 设置和方法配置以 `configs/study.json` 及每组数据的冻结 manifest 为准。清单中的源码和 Git 摘要描述数据生成时的来源，不会随文档整理而改写。模块边界与数据流见架构文档。

原基准与planning确认按任务配对、设置等权；校准按独立校准重复聚类，范围限定于固定任务集。逐比较95%区间不代表全体结论的联合保证。精确计算和探索性诊断分别解释。运行时间为单工作进程下的观测值，不能据此认定机器没有其他负载。

## 机制与校准实验

从[统一研究说明](report.md)开始阅读；第5.1–5.3节解释重测边界及机制，第6.5节完成搜索分账，第7.1节报告有限样本校准。数据字典见[research数据](../data/README.md)，判断、效应范围与停止理由见[证据台账](evidence.md)。研究范围保持冻结。

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# 完整重建新实验到独立目录；原始数据不覆盖
uv run python -m scripts.run_research --source data --out artifacts/research-reproduction --workers 2
# 以下读取本地正式数据；verify不重新优化保存策略
uv run python -m scripts.verify_research --source data --out results/research
uv run python -m scripts.analyze_mechanisms --source data --out results/research
uv run python -m scripts.analyze_calibration --source data --out results/research
```

`run_research`按作业原子保存、支持续跑，`--group grid|deep|calibration`可选固定子组；冻结清单要求配置和数值源码相同，运行顺序不改变任务或校准表。上述分析入口读取`<source>/main`、`<source>/planning`和`<source>/research`。重建副本的分析可将三个组放在同一新的source根目录下（main/planning复用原始目录的符号链接即可），不改写正式目录。

阶段计时可独立执行`uv run python -m scripts.time_initialization --out artifacts/phase-timing-reproduction.json`。它只重测8个规定任务、三次重复，核对保存策略和计数，不增加统计样本。受控计时请在本项目批处理结束后运行。原始正式A–D源码版本为`75ed79b`；分析与复核脚本随当前HEAD保存，完整来源由`data/research/manifest.json`和`results/research/release.json`关联。数据继续保持本地未压缩、不进入Git。

## 研究冻结与版本对应

研究基准版本`db37d68`与研究冻结提交`53659a9`保留；`b39a11c`增加原承诺内的剪枝消融脚本与协议，生产规划算法和既有实验记录不变。[统一研究说明](report.md)第5.3节与[完整同任务表](../results/research/deep_summary.csv)是深任务最终比较，`deep.csv`提供每个任务的五方法记录和匹配计时。

| 证据批次 | 范围 | 来源与复算 |
|---|---|---|
| 原基准main/large/stress | 3240主任务、160大任务及40复用压力任务 | manifest记录生成时HEAD `4672e95`与实际源码哈希；可恢复的同源码快照`3382745`用于严格复算 |
| 独立planning确认 | 新种子20261124的160任务 | `6413033`，与旧large分开汇总 |
| research机制与校准 | 1329网格、24深任务、36复用任务上的120校准重复 | 数值源码`75ed79b`，研究说明基准`db37d68`，冻结汇总见`53659a9` |
| 剪枝消融 | 同20个large任务、40条策略及120条匹配计时 | `b39a11c`；`scripts.pruning`，只新增显式剪枝成本对照 |
| 深任务匹配计时 | 同24任务×5方法×3重复 | `scripts/time_deep.py`及`data/research/deep_timing.json`；不是新增任务 |

已有正式计时直接由分析器读取，不必重跑。需要复现计时时，批处理结束后单独执行：

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run python -m scripts.time_deep --source data --out artifacts/deep-timing-reproduction.json
```

完整研究材料以`results/research/release.json`关联最终源码提交、配置、数据和结果哈希；`SHA256SUMS`校验该目录输出。Git不包含大结果，交接需同时保留本地data/results目录。数据生成版本不被当前文档版本覆盖，旧批次的均值不混入新批次排名。
