# 结论与证据索引

本文件定位当前报告的证据，不替代[报告正文](report.md)或重复维护实验配置。统计口径见[实验协议](protocol.md)，计算检查见[验证说明](validation.md)。链接指向本地规范数据；数据与生成结果尚未纳入Git。

## H1：历史价值的条件

**限定条件下支持：** 有历史主网格中，correct相对reset及misread的平均净效用差分别为.081429与.006387，95%区间分别为[.074886,.088053]与[.004643,.008231]；正收益任务比例81.42%和7.22%。所有策略均在相同真实历史条件模型下评价。

逐任务奖励/成本分解残差小于1e-10。p_C=.5、κ=.1时correct和reset都停止，.131968差值全部来自终止选择；p_C=.5、κ=.005对misread主要节省无益测试；p_C=.9对misread的成本节省项为负。先验构成与K等因素有混杂，子群关联不作为独立因果解释，n仅覆盖0/3。

证据：[总体表](../results/history/overall.csv)、[能力和成本分层](../results/history/capability_cost.csv)、[逐任务分解](../results/history/task_mechanisms.csv)、[组说明](../results/history/README.md)。

## H2：充分统计量与信息价值

**已支持：** 独立先验、已知观测机制且无共享未知参数时，完整历史与按能力分组的计数定义同一决策问题。事前信息层级与同一历史下真实最优策略不劣于reset是不同命题；不把不同信念的价值逐历史排序，不推广为任意文本摘要充分或有限计算保持信息优势。

证据：[T1证明](theory.md)、[独立验证测试](../tests/test_independent.py)、[充分性与边界测试](../tests/test_core.py)、[独立策略评价](../results/validation/independent_policy_values.json)。

## S1：连续测试的存在性与边界

**已支持：** 在P(C)=.05、p_C=.8、p_A=.05、q=.5、B=2的解析例子中，一次成功预测11/28，两次成功预测823/1100；V₂=.5+max(0,273/32000−87c/80)。非负成本c<91/11600时两步优于停止，c=.005时净改善.00309375。11个固定成本点与DP及评价误差小于1e-10。该例证明存在性，不代表随机主分布中的普遍量级。

证据：[后验决策表](../results/sequential/analytic_decision_table.csv)、[成本诊断](../results/sequential/cost_diagnostic.csv)、[固定协议](../results/sequential/protocol.json)、[T2证明](theory.md)。

## S2：分布上的连续测试收益

**限定条件下支持：** 全3240任务的DP−滚动myopic均值.00018016等于正收益频率161/3240乘以条件均值.00362550。B3均值.00051771，95%区间[.00046313,.00057478]。1980任务双方根停止；161个获益任务中9个myopic根停止、34个首测不同、118个首测相同后续不同。

条件均值是按观察结果划分的描述，不替代总体。固定κ=.02时DP/myopic平均测试数.656351/.660503；跨κ的成本差不能直接解释为次数差。数值零阈值1e-10不是实用效应阈值。

证据：[分组表](../results/sequential/decomposition.csv)、[逐任务表](../results/sequential/task_decomposition.csv)、[组说明](../results/sequential/README.md)。

## P1：同后继更新预算与完整层门槛

**限定条件下支持：** large n=0、ε=.01、M=100/1000/10000时，搜索相对迭代加深的平均差为.001514/.010897/.001341，对应逐比较区间均大于0。所有n0任务的迭代加深在M100/1000返回相同策略。当前概率表下根和第一层后的两个观测均为正概率，完整深度二需要2K+(2K)²次更新，K20/40对应1640/6480；两个固定任务的计数及保存策略与推导一致。

同M不等于同CPU成本。主网格的迭代加深已达到精确值，不将large现象推广为搜索普遍优于迭代加深。

证据：[预算配对表](../results/study/tables/budget_pairs.csv)、[完成深度计数](../results/search/depth_completion.csv)、[策略平台](../results/search/iterative_plateau.csv)、[主网格质量](../results/search/main_quality.csv)。

## P2：自然成本基线与搜索

**不支持搜索普遍占优：** large n=0中，myopic/blinkered平均效用均为.817001，平均更新数540/3540；M10000、ε=.01搜索为.812329及10000更新。搜索减基线均值−.004673，95%区间[−.004999,−.004349]。

myopic用一步判断构造完整时域滚动策略；迭代加深交付截断完成层；搜索另维护界和前沿。动作扫描数不是统一FLOP计量。large没有真实最优值，不能将基线称为最优。

证据：[自然成本表](../results/search/natural_cost.csv)、[基线配对表](../results/study/tables/natural_baselines.csv)、[效用成本图](../results/search/quality_cost.pdf)。

## P3：策略质量与证书宽度

**限定条件下支持：** 主实验全部199个搜索非精确输出因正容差停止；相同M只改ε为0后199个达到exact，平均更新数27.96→253.88。B3另有308个已达到精确值的策略仍有正界宽。large n0 M10000平均界宽.029484，U−J(myopic)=.024811只是上界，并需额外搜索成本。

199任务干预属于探索性诊断，不替换原排名，也不证明任意规模下取消容差即可在同M求精确值。large真实regret未知；新的下界初始化未测试。

证据：[干预协议](../results/search/tolerance_protocol.json)、[全部诊断记录](../results/search/tolerance_diagnostic.csv)、[完整策略](../results/search/tolerance_policies.jsonl)、[证书收敛](../results/search/certificate_convergence.csv)、[独立诊断复核](../results/validation/tolerance_review.json)。

## R1/R2：模型失配

**限定条件下支持：** 指定n0、假定p_C=.4的20任务策略全部保持相同；假定.6时18/20改变，平均真实regret约.000598。有历史的两个偏移条件均20/20策略相同。恰好一路径重采样时，有历史nominal DP−reset均值.080298，95%区间[.040662,.123482]。

40个压力任务复用主网格，未增加独立样本。全零观察不证明总体等效；重采样组没有相关真实最优参照，只报告策略差。仅初始信念误设的TV界不能直接用于未来概率表误设。

证据：[策略稳定性](../results/search/stress_policy_stability.csv)、[概率表压力](../results/study/tables/probability_shift.csv)、[重采样压力](../results/study/tables/resampling.csv)、[T3证明](theory.md)。

## V1：评价、计算和来源

规划器不接触隐藏类型，评价器执行完整冻结策略，M统计全部实际后继更新。18个预定设置的51策略独立有理数评价最大误差3.33e-16，六种方法的实际更新调用数与记录一致。计时包含上下文初始化；单worker不证明无争抢；别名排除后不同方法的计时任务覆盖可能不同，不据此作跨记忆或混合网格纯速度排名。

证据：[验证说明](validation.md)、[独立检查](../results/validation/checks.json)、[更新计数](../results/validation/transition_accounting.json)。
