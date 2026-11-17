# 第二版结果：从平均现象到决策机制

完整报告见REPORT.md，置信度与研究决策见EVIDENCE_LEDGER.md，原始数据及原参数网格全部保留。没有新增随机确认样本；新增证据是原数据重分析、解析边界、固定任务上的诊断干预。95%区间均逐比较解释。

| 研究问题 | 第二版回答 | 证据 |
|---|---|---|
| 历史何时有价值？ | reset与misread机制不同。n3的平均差.081429与.006387，正收益频率81.42%与7.22%。pC=.5、cost=.1的reset比较双方都立即停止，.131968全来自终止选路；不能笼统归为省测试。 | history/overall.csv、capability_cost.csv、task_mechanisms.csv；REPORT §4 |
| 误读只会过早放弃吗？ | 否。pC=.5、cost=.005时误读诱发无益重测；pC=.9时正确历史反而值得支付额外测试成本。 | history/SUMMARY.md、history_decomposition.pdf |
| 为什么连续测试均值小？ | 161/3240任务获益，频率4.9691%×条件收益.00362550=.00018016。161中118个首测相同后续不同、34个首测不同，仅9个是myopic不启动。 | sequential/task_decomposition.csv、decomposition.csv；REPORT §5 |
| 决策边界可否手算？ | 单次成功预测11/28<.5，SS后823/1100>.5；c<91/11600时两步有净价值。先验和fallback边界另有解析表达，11点成本诊断通过。 | sequential/analytic_decision_table.csv、cost_diagnostic.csv、protocol.json |
| 搜索为什么在同M下赢ID？ | 完整深度2的更新门槛1640/6480，M1000仍只交付深度1；搜索在部分树逐步改善。只针对当前large条件，不推广到已求精确值的主网格ID。 | search/depth_completion.csv、iterative_plateau.csv |
| 为什么自然成本基线更好？ | myopic完整滚动策略平均540次更新、效用.817001；搜索10000次、.812329，并承担界与扫描工作。成本约束不同，保留该负面结果。 | search/natural_cost.csv、quality_cost.pdf |
| 小任务搜索是否因预算不足损失？ | 原199个非精确输出全是正容差停止。同任务同M只改ε=0后199个精确，平均更新27.96→253.88。属于探索性机制干预，不替换原主表。 | search/tolerance_protocol.json、tolerance_diagnostic.csv、tolerance_policies.jsonl |
| 证书有什么价值？ | n3零更新即可证明STOP；n0 M10000平均宽度.029484，真实最优未知。B3有308个已精确策略却宽度正，说明证明进展不等于实际损失。 | search/main_quality.csv、certificate_convergence.csv；REPORT §6 |
| 零压力损失是否鲁棒？ | 部分条件策略保持不变，模型值可有误而动作不变；不宣称任意失配无害。重采样组真实最优未知，不报告regret。 | search/stress_policy_stability.csv；baseline/tables/probability_shift.csv、resampling.csv |

## 结论的强化、收窄与否定

强化：历史收益恒等分解、序贯收益频率分解、迭代深度计数门槛及正容差干预给出了可核验机制。
收窄：历史收益不全由少测产生；连续测试的存在性不意味着大平均收益；同M不等于同CPU成本；耗时是单worker观测；压力零结果不证明总体等效。
否定：搜索普遍优于自然成本myopic/blinkered的主张不成立。原基准所有结果仍被保留。

## 旧版差异与未决范围

新增REPORT、证据台账、技术审阅、报告审阅、答辩问答及机制图。未修改生产规划器或真值评价器，旧ZIP摘要保持不变。只更正分析模块的计时说明措辞。

没有主张large真实最优已求得、先验子群有独立因果作用、方法对任意模型失配鲁棒，或实际DPO效应已验证。这些是模型/研究范围边界，不是用测试通过代替研究问题。候选额外工作与停止理由逐项见台账末表。
