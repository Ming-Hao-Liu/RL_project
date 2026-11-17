# 项目架构

本项目实现有限时域贝叶斯决策实验。模型和研究约定见[实验协议](protocol.md)，数学推导见[理论说明](theory.md)，运行命令见[代码使用与复现](usage.md)。

## 模块与数据边界

```text
src/rl_retest/
  model.py                 模型、历史、信念和纯数学运算
  policy.py                不可变、完整的 Stop/Test 策略
  bounds.py                完美信息上界与停止判据
  evaluation.py            固定策略的真实条件评价
  planners/
    common.py              预算、转移缓存和规划结果
    dynamic.py             精确 DP、滚动前瞻和迭代加深
    blinkered.py           单路径限制规划
    bounded.py             上下界搜索及检查点
  experiments/
    tasks.py               实验配置、任务和记忆条件
    calibration.py         合成校准与固定plug-in规划输入
    runner.py              规划、评价、并行、续跑和验证
    records.py             策略与结果编解码
  analysis.py              保存记录的统计与图表
  cli.py                   命令行流程组合
scripts/                   完整基准分析、机制诊断和定向复核
tests/                     独立联合枚举验证器与行为测试
data/                      正式任务和完整策略记录
results/                   统计表、图表、诊断和验证输出
```

数学模块不读配置、不写文件、不启动进程。策略可以独立于规划器保存、读取和评价。规划器只接收允许使用的模型；真实条件模型由实验层单独传给评价器。分析与绘图读取持久化记录，不重新优化已保存策略；诊断脚本中明确指定的规划干预单独保存。

## 依赖方向

```text
cli → experiment runner → tasks / planners / evaluation / records
cli → analysis → records
planners → common / bounds / policy / model
evaluation → policy / model
calibration → tasks / model
tasks → model
records → policy
bounds → model
```

`model` 和 `policy` 不依赖规划器或实验模块。规划器不能读取任务生成器、真实评价器或结果文件；评价器不得导入规划器或重新选择终止路径。生产分析模块不调用规划器。模块导入不运行实验或创建输出目录。[独立验证器](../tests/oracle.py) 不复用生产后验、上界或 Bellman 实现。

## 公共契约

| 对象 | 归属 | 契约 |
|---|---|---|
| `ModelSpec`、`History`、`Belief` | model | 类型顺序、能力概率表、历史计数及奖励语义固定 |
| `PlanningProblem` | model | 只含规划信念和规划模型，不含隐藏类型 |
| `EvaluationContext` | model | 含真实条件信念与评价模型 |
| `Stop`、`Test` | policy | 不可变，终止路径明确，两个观测分支完整 |
| `TransitionCache`、`PlanResult` | planners.common | 每次规划独占缓存和预算；结果包含完整策略、模型价值、界和统计 |
| `Task`、`ExperimentSpec` | experiments.tasks | 固定任务与实验条件；不向规划器泄漏生成类型 |

`evaluate(policy, context)` 返回期望净效用、期望测试数和终止成功概率；它执行保存的 `Stop.path`。转移预算集中在 `planners.common`，后验数学由 `model` 提供，评价计算不消耗规划预算。Bellman 递推集中在 `dynamic`，上界集中在 `bounds`。

## 运行与持久化

`rl-retest prepare` 冻结配置、任务与作业；`run` 规划并评价；`verify` 从完整策略复算；`analyze` 读取结果生成表图。每个作业独占输出分片，原子写入；续跑保留成功记录。多预算轨迹保存不可变检查点，理论等价的方法标记来源后复用计算。

[study.json](../configs/study.json)、[planning.json](../configs/planning.json)和[research.json](../configs/research.json)分别定义原基准、独立初始化确认、机制与校准实验；正式运行以各自 manifest 内的配置快照为准。manifest 保存任务、作业、配置、依赖锁和规划源码摘要。核心源码摘要按字节计算，覆盖模型、策略、上界、评价、规划器和实验模块；改变这些文件后须建立新运行，不能改写已有来源摘要。分析脚本版本单独记录。

`data/` 和 `results/` 各保留一份规范内容，只有目录说明纳入 Git；原始数据、派生输出、环境和缓存均留在项目内。本地数据格式与目录说明见[数据入口](../data/README.md)及[结果入口](../results/README.md)。

根节点初始化接口`solve_bounded_warm`返回普通PlanResult和初始化检查点。它复用同一TransitionCache调用滚动myopic；只允许单预算调用，避免较大预算初始化被较小预算检查点免费复用。原搜索默认接口保持不变。独立确认脚本使用带研究名和种子的任务ID，数据目录为data/planning，不与旧三组运行混存。

## 有限校准与机制扩展

`experiments/calibration.py`只负责合成已知类型校准计数、固定估计器与plug-in规划输入，不改变规划器接口。历史/未来单独误差条件在输入构造层完成。共享估计表的统计依赖保存在实验记录的size/repeat中；规划算法仍只接收固定PlanningProblem。

`scripts/run_research.py`冻结机制/校准清单、独立作业与原子输出，复用现有DP和搜索；`analyze_mechanisms.py`/`analyze_calibration.py`从保存记录生成统计与分歧案例；`verify_research.py`固定策略复算并调用测试目录独立验证器。解析阈值案例由分析脚本调用已有DP验算，统计部分不触发重规划。`time_initialization.py`仅补测既有warm搜索；新增计时字段不改变动作选择、计数或预算。`time_deep.py`补测既有24任务的五方法，保存独立匹配计时；`analyze_mechanisms.py`将其并入同任务质量/成本汇总。
