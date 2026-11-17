# 当前结果

本目录保存当前研究使用的一份未压缩结果。除本说明外，表格、图表、诊断策略和机器验证输出不纳入 Git。

| 目录 | 内容 | 生成入口 |
|---|---|---|
| study | 完整基准的配对统计、压力测试、表格和图表 | `scripts.analyze_study` |
| history | 历史收益的终止奖励／测试成本分解 | `scripts.analyze_history` |
| sequential | 序贯收益频率分解、解析后验及成本边界 | `scripts.analyze_sequential` |
| search | 策略质量、自然成本、证书、容差诊断记录及完整策略 | `scripts.diagnose_tolerance`、`scripts.analyze_search` |
| validation | 原基准独立评价、更新计数、数据完整性及流程验证 | `scripts.review_evidence`、测试与 CLI 验证 |
| planning | 独立初始化确认及匹配计时 | `scripts.analyze_planning`、`scripts.verify_planning` |
| research | A–D机制/校准分析、深任务完整比较、研究冻结清单 | `scripts.analyze_mechanisms`、`scripts.analyze_calibration`、`scripts.verify_research` |

CSV 是可读取的数值表；PDF 是矢量图，PNG 是同图预览，两种格式不是两个实验。JSON／JSONL 保存协议、来源和完整策略。各分析目录的 README 解释对应输出，研究结论统一见 `../docs/report.md` 和 `../docs/evidence.md`。

输入来自 `../data/`。固定任务上的成本和容差诊断单独标记，不能当作新的随机确认样本。重建命令见项目 README；容差诊断先于搜索分析执行。

`validation/data_integrity.json` 列出正式原始文件的 SHA256；`validation/workspace_checks.json` 记录原基准整理时的测试、数据复算、数值一致性和独立目录检查范围；最新工作区与研究冻结状态以`research/release.json`为准。文件存在性和摘要检查不代替数学证明或算法验证。

`planning/` 是独立初始化确认分析：effects保存预定主比较和次要区间，paired_tasks分解上下界与策略增益，costs保留全部结束状态，timing汇总匹配任务的重复计时。两幅PDF/PNG展示质量/成本与初始化后检查点。使用`python -m scripts.analyze_planning`重建，固定策略验证入口为`scripts.verify_planning`。

`research/` 保存当前A–D扩展：`search_attribution*`为已有独立批次的初始化/搜索分账；`threshold_cases*`为精确重测案例；`difficulty*`为原3240任务的H/D分类；`grid*`、`deep*`为固定机制任务；`calibration*`为有限校准原始估计表的整理、聚类统计、误差归因和动作案例；`phase_timing.csv`为8固定任务的分阶段计时。`verification.json`记录保存策略及独立枚举检查，`input_sha256.json`记录本轮全部原始文件摘要，`release.json`与`reproduction.json`记录版本及隔离复现范围。生成脚本分别为`analyze_mechanisms`、`analyze_calibration`、`verify_research`。

新基础诊断图为PNG：`difficulty.png`、`mechanism_grid.png`、`calibration.png`，均由同目录完整数值表生成。校准区间按30个独立校准重复聚类，只描述固定36任务上的校准随机性；机制网格及24个深任务不附虚构的随机总体区间。

`research/deep_summary.csv`为24深任务的最终五方法汇总：最优任务数、平均/最大/非零条件平均损失、更新成本、匹配规划时间中位数/P90及搜索证书闭合数分别列出。条件正损失子群为空时字段留空；不把DP的相同上下界字段计作搜索证书。逐任务`deep.csv`的`planning_median_seconds`来自同任务三次重复中位数，汇总再跨24任务取中位数/P90。原始匹配计时在`data/research/deep_timing.json`，现有分析入口自动读取；没有该文件时不生成耗时结论。
