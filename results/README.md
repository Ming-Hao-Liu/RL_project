# 当前结果

本目录保存当前研究使用的一份未压缩结果。除本说明外，表格、图表、诊断策略和机器验证输出不纳入 Git。

| 目录 | 内容 | 生成入口 |
|---|---|---|
| study | 完整基准的配对统计、压力测试、表格和图表 | `scripts.analyze_study` |
| history | 历史收益的终止奖励／测试成本分解 | `scripts.analyze_history` |
| sequential | 序贯收益频率分解、解析后验及成本边界 | `scripts.analyze_sequential` |
| search | 策略质量、自然成本、证书、容差诊断记录及完整策略 | `scripts.diagnose_tolerance`、`scripts.analyze_search` |
| validation | 独立评价、更新计数、数据完整性及流程验证 | `scripts.review_evidence`、测试与 CLI 验证 |

CSV 是可读取的数值表；PDF 是矢量图，PNG 是同图预览，两种格式不是两个实验。JSON／JSONL 保存协议、来源和完整策略。各分析目录的 README 解释对应输出，研究结论统一见 `../docs/report.md` 和 `../docs/evidence.md`。

输入来自 `../data/`。固定任务上的成本和容差诊断单独标记，不能当作新的随机确认样本。重建命令见项目 README；容差诊断先于搜索分析执行。

`validation/data_integrity.json` 列出正式原始文件的 SHA256；`validation/workspace_checks.json` 记录当前代码测试、数据复算、数值一致性和独立目录检查的范围。文件存在性和摘要检查不代替数学证明或算法验证。
