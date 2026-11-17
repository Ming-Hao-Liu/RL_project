# RL Project

有限时域贝叶斯决策研究：能力改变后如何解释历史测试，并在测试成本与规划计算预算下决定是否重测。

## 项目入口

- [研究正文](docs/report.md)：问题、方法、结果及讨论。
- [实验协议](docs/protocol.md)：模型、配置、评价与统计口径。
- [理论说明](docs/theory.md)：推导、假设与边界。
- [证据索引](docs/evidence.md)：研究结论对应的数据和图表。
- [验证说明](docs/validation.md)：检查范围、实际结果及计时口径。
- [项目架构](docs/architecture.md)：模块职责、接口和依赖方向。
- [交接材料](docs/handoff.md)：选图、问答和实际贡献说明。
- [原始提案](docs/references/Proposal.md)与[课程要求](docs/references/Project_Requirements.md)：原文及译文，作为来源材料保留。

## 目录

```text
src/rl_retest/    模型、策略、规划、评价和实验流程
tests/           独立枚举、算法、数据流程和架构检查
configs/         可执行研究配置
scripts/         正式数据分析、机制诊断和独立复核入口
docs/            当前研究与工程说明
data/            一份正式任务和完整策略记录（本地，不纳入 Git）
results/         一份统计表、图表、诊断与验证结果（本地，不纳入 Git）
artifacts/       新运行的临时输出，默认忽略
```

[数据说明](data/README.md)解释任务、作业、记录及来源摘要；[结果说明](results/README.md)说明各输出的用途。Git 只跟踪这两个目录的 README，不包含实验数据或生成图表。新克隆需要获得数据目录，或按冻结配置重新运行实验。

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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

正式运行使用 `main`、`large` 或 `stress` suite，并为每次新运行指定新的输出目录。`run` 检查冻结源码和配置，续跑只补齐未完成的作业；不要覆盖已有正式记录。`pruning` 是可执行的剪枝诊断配置，不属于已报告的三组正式实验。

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

## 数据与计算契约

测试预算限制环境交互次数；规划预算统计实际后继信念更新。规划器返回完整不可变策略，评价器执行其中保存的终止路径，不重新优化。真实条件模型与规划模型分别传入。

正式种子、bootstrap 设置和方法配置以 `configs/study.json` 及每组数据的冻结 manifest 为准。清单中的源码和 Git 摘要描述数据生成时的来源，不会随文档整理而改写。模块边界与数据流见架构文档。

统计按任务配对、设置等权；逐比较 95% 区间不代表全体结论的联合保证。精确计算和探索性诊断分别解释。运行时间为单工作进程下的观测值，不能据此认定机器没有其他负载。
