# 项目架构

## 目的与约束

本项目是一个可复现的有限时域贝叶斯决策实验。研究协议和验收标准以 [Guideline.md](../Guideline.md) 为准；本文是代码边界和运行契约的依据。原始提案与课程要求保留在项目根目录。

数学模型和实验流程分离；策略独立于规划器；规划模型和真实评价模型分别传递；统计分析只读取原始记录。采用 Python 3.11、NumPy、pytest、Matplotlib 与 uv。通用 POMDP 框架提供的求解器需要额外适配因子化信念、完整策略证书与后继信念计算预算，因此当前用项目专用实现表达这些研究对象。

## 目标结构

```text
src/rl_retest/
  model.py                 模型、历史、信念和纯计算
  policy.py                不可变、完整的 Stop/Test 策略
  bounds.py                完美信息上界与停止判据
  evaluation.py            在真实条件模型下评价固定策略
  planners/
    common.py              预算、转移缓存和规划结果
    dynamic.py             精确 DP、滚动前瞻和迭代加深
    blinkered.py           单路径限制规划
    bounded.py             上下界搜索及检查点
  experiments/
    tasks.py               实验配置、任务和记忆条件
    runner.py              规划、评价、并行和续跑
    records.py             策略与结果读写
  analysis.py              统计、图表和覆盖检查
  cli.py                   命令行组合入口
tests/                     独立联合枚举验证器与回归测试
configs/study.json         唯一可执行研究配置
artifacts/<run_id>/        清单、任务、作业、原始结果和分析输出
docs/                      架构、理论、结果与贡献记录
```

文件随功能实施时创建；不预先添加空模块。

## 依赖方向

```text
cli → experiment runner → task construction / planners / evaluation / records
cli → analysis → records
planners → common / bounds / policy / model
evaluation → policy / model
task construction → model
records → policy
bounds → model
```

`model` 和 `policy` 不依赖规划器或实验；规划器不能读取任务生成器、真值评价器或文件；评价器不得导入规划器或替策略重新选择终止路径；分析不得调用规划器。模块导入不执行实验或文件操作。`tests/oracle.py` 不复用生产后验、上界或规划实现。

## 重要接口与责任

- `ModelSpec`、`History`、`Belief`、`PlanningProblem`、`EvaluationContext` 描述模型；规划输入不含隐藏类型或真实评价对象。
- `Stop(path_id)` 和 `Test(path_id, failure_policy, success_policy)` 是不可变完整策略。fallback 用固定编号；零概率观测分支也有可执行策略。
- 规划器返回 `PlanResult`，包括策略、模型价值、可用界、预算与检查点。每次规划独占预算和缓存。
- `evaluate(policy, context)` 返回真实期望效用、期望测试数和成功概率，执行策略所保存的终止路径。
- `Task`、`ExperimentSpec` 定义任务和实验条件；序列化记录含格式版本、完整策略、模型值、真实评价及运行来源。
- 转移和预算集中在 `planners.common`；评价的后验更新不消耗规划预算。Bellman 递推集中在 `planners.dynamic`；上界计算集中在 `bounds`。

主入口为 `rl-retest prepare/run/verify/analyze`。`prepare` 冻结配置、任务与作业；`run` 规划并评价；`verify` 从保存策略独立复算；`analyze` 只读取记录。每个作业拥有独立缓存与结果分片，成功后原子写入；续跑保留成功结果。大规模搜索和迭代加深各在一次执行中保存多个预算检查点；严格等价的方法与记忆条件复用结果，并在记录中标记来源。

研究配置以 `configs/study.json` 为唯一可执行来源。每次运行记录配置、随机种子、任务哈希、源码哈希、Git 版本、锁文件哈希和依赖环境；规划或评价源码变化后需建立新运行目录。分析结果单独记录分析源码哈希。所有生成物、环境、缓存和临时目录位于项目内；原始结果默认不纳入 Git，正式证据包应归档并提供校验值。

## 实施顺序

1. 项目说明、可安装骨架、策略和结果的往返序列化。
2. 模型运算、独立联合枚举、精确 DP 与固定策略评价；先打通一个解析任务。
3. 基线、统一预算和上下界搜索。
4. 正式配置展开、批处理、续跑、覆盖检查、统计和图表。

各层随实现添加行为测试；架构测试约束依赖方向。详细数学、实验和验收要求见 Guideline。
