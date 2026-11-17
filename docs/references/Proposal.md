# When Should Failed Paths Be Retested After DPO Training?
# DPO 训练后，何时应重新测试失败路径？

**Course focus:** Algorithms and theoretical analysis; finite-horizon decision making, dynamic programming, and value of information.  
**课程重点：** 算法与理论分析；有限时域决策、动态规划和信息价值。

**Team members:** Minghao Liu (刘明昊) / 260010126; Haoyang Qin (秦浩洋) / 260010154; Shinan Yin (尹施南) / 260010270; Xuhao Liang (梁栩豪) / 250010046; Xi Qin (秦曦) / 260010138.  
**团队成员：** 刘明昊 / 260010126；秦浩洋 / 260010154；尹施南 / 260010270；梁栩豪 / 250010046；秦曦 / 260010138。

**Proposal submission:** November 7, 2026; **Final report:** December 5, 2026.  
**提案提交：** 2026 年 11 月 7 日；**最终报告：** 2026 年 12 月 5 日。

## 1. Research Problem and Expected Outcomes / 研究问题与预期成果

An agent may benefit from revisiting a path that failed when its capabilities were more limited. We study a stationary environment in which the agent's execution capability changes. Our objective is to determine how historical failures should be interpreted in light of the capability under which they occurred, when paid retesting is worthwhile, and how to obtain near-optimal decisions with a quantified optimality gap under a limited computation budget.

智能体可能从重新尝试一条曾在其能力较弱时失败的路径中获益。我们研究一个环境保持稳定、但智能体执行能力发生变化的情形。我们的目标是确定：如何结合失败发生时的能力来解释历史失败，何时付费重新测试是值得的，以及如何在计算预算有限的情况下获得具有量化最优性差距的近优决策。

We first compare the expected utility of retaining the complete history, retaining sufficient statistics, and discarding memory. We also quantify the loss incurred by treating historical outcomes as observations generated under the current capability. We then examine cases in which a single test is unprofitable but a sequence of tests is valuable. Finally, we investigate whether search can identify a high-value policy within a finite computation budget while providing an upper bound on its suboptimality.

首先，我们比较保留完整历史、保留充分统计量和丢弃记忆三种方式的期望效用，并量化将历史结果误认为当前能力下产生的观测所造成的损失。接着，我们考察单次测试不划算、但连续测试有价值的情形。最后，我们研究搜索能否在有限计算预算内找到高价值策略，同时给出其次优程度的上界。

We will develop a benchmark with explicit capability changes and analyze how historical information affects utility, including the losses caused by its misinterpretation. Analytically tractable examples will illustrate when historical experience misleads decisions and when sequential testing improves utility. We will then implement search with upper and lower bounds and evaluate its decision quality and computational cost.

我们将构建一个明确包含能力变化的基准，并分析历史信息如何影响效用，包括错误解释历史信息造成的损失。可进行解析处理的例子将说明历史经验何时会误导决策，以及连续测试何时能提高效用。随后，我们将实现带上下界的搜索，评估其决策质量和计算成本。

## 2. Related Work / 相关工作

Problems involving costly information acquisition followed by a terminal decision have been formulated as Bayesian selection problems and metalevel Markov decision processes. Their reward is concentrated in the terminal choice, whereas standard multi-armed bandit problems accumulate rewards through repeated actions. The blinkered approximation restricts each planning calculation to further evaluation of a single alternative [1]. BMPS approximates the value of computation using features such as myopic and perfect-information values; its experiments include Bernoulli selection tasks and comparisons with optimal policies, myopic policies, and deep Q-networks (DQN) [2]. We build on these formulations to study the reuse of historical observations after capability changes.

涉及有成本的信息获取并在之后做出终止决策的问题，已被表述为贝叶斯选择问题和元层马尔可夫决策过程。这类问题的奖励集中于最终选择，而标准多臂老虎机问题通过重复行动累积奖励。Blinkered 近似将每次规划计算限制为只进一步评估单个备选项 [1]。BMPS 利用短视价值和完美信息价值等特征来近似计算的价值；其实验包括伯努利选择任务，以及与最优策略、短视策略和深度 Q 网络（DQN）的比较 [2]。我们以这些表述为基础，研究能力变化后历史观测的再利用。

HIRO addresses the changing meaning of historical high-level experience as low-level behavior evolves, using off-policy correction during training [3]. We represent capability changes through capability-dependent success and failure probabilities and analyze when historical observations remain informative for current decisions.

HIRO 针对底层行为演化时历史高层经验含义发生变化的问题，在训练期间使用离策略校正 [3]。我们通过依赖能力的成功和失败概率表示能力变化，并分析历史观测何时仍能为当前决策提供信息。

ERRAND studies paid memory revalidation, budget allocation, and memory maintenance, and compares different capability models in Appendix E.10 [4]. It focuses on changes in external facts while agent capabilities remain fixed. We hold path types fixed and allow execution capability to change in a known manner, examining posterior updates, sequential testing, and bounds on decision error. The Memory Trust Gap studies capability-dependent overreliance on stale memories [5], while Faulty Memory examines performance degradation under repeated updates to textual memory [6]. These studies provide application context. Our simulations explicitly record paths, capabilities, and outcomes to isolate how historical observations affect probabilistic inference and decision utility.

ERRAND 研究有成本的记忆重新验证、预算分配和记忆维护，并在附录 E.10 中比较不同的能力模型 [4]。它关注智能体能力固定时外部事实的变化。我们固定路径类型，允许执行能力以已知方式变化，考察后验更新、连续测试和决策误差界。The Memory Trust Gap 研究依赖能力的对过时记忆的过度依赖 [5]，Faulty Memory 则考察反复更新文本记忆时的性能下降 [6]。这些研究提供了应用背景。我们的仿真明确记录路径、能力和结果，以单独考察历史观测如何影响概率推断和决策效用。

HSVI provides upper and lower bounds on the utility gap between a policy and an optimal policy [7], while AEMS improves existing policies through anytime online search [8]. We adopt this bounding framework. Free access to all path types defines an upper bound, and the value of an executable policy defines a lower bound. The focus is on explaining how capability changes alter the relevance of historical observations and comparing decision quality and computational cost under controlled conditions.

HSVI 给出策略与最优策略之间效用差距的上界和下界 [7]，AEMS 则通过可随时终止的在线搜索改进已有策略 [8]。我们采用这一界定框架：免费获知所有路径类型所得到的价值构成上界，可执行策略的价值构成下界。重点是解释能力变化如何改变历史观测的相关性，并在受控条件下比较决策质量和计算成本。

## 3. Task Formulation and Objective / 任务形式化与目标

Each task contains $K$ candidate paths and may include a fallback path with known success probability $q_0$. The type of each path, $Z_i \in \{I, C, A\}$, remains fixed throughout the task. A permanently infeasible path of type $I$ has success probability $p_I(s)=0$. A capability-dependent path of type $C$ has a success probability that varies with capability $s$. A capability-independent path of type $A$ has a fixed success probability, which may be below 1. Prior distributions over path types are known and mutually independent. The policy knows the success-probability table $p_z(s)$ for each type and capability, while the true type of each path must be inferred from observations. Conditional on type and capability, trial outcomes are independent.

每个任务包含 $K$ 条候选路径，还可以包含一条成功概率 $q_0$ 已知的备选路径。每条路径的类型 $Z_i \in \{I, C, A\}$ 在整个任务期间保持固定。永久不可行的 $I$ 类型路径成功概率为 $p_I(s)=0$；能力依赖的 $C$ 类型路径，其成功概率随能力 $s$ 变化；能力无关的 $A$ 类型路径具有固定成功概率，该概率可以小于 1。路径类型的先验分布已知且相互独立。策略知道各类型在各能力下的成功概率表 $p_z(s)$，但每条路径的真实类型必须根据观测推断。在给定类型和能力的条件下，各次试验结果相互独立。

Each historical record contains the path, the capability at the time of testing, and the outcome. Success counts $n^+_{i,s}$ and failure counts $n^-_{i,s}$, grouped by path and capability, retain all historical information required for type inference and therefore constitute sufficient statistics. Let $b_i^0(z)$ denote the prior type probability and $b_i(z)$ the posterior probability after incorporating historical observations. Then

每条历史记录包含路径、测试时的能力以及结果。按路径和能力分组的成功次数 $n^+_{i,s}$ 和失败次数 $n^-_{i,s}$ 保留了类型推断所需的全部历史信息，因此构成充分统计量。令 $b_i^0(z)$ 表示类型的先验概率，$b_i(z)$ 表示纳入历史观测后的后验概率，则

$$b_i(z) \propto b_i^0(z)\prod_s p_z(s)^{n^+_{i,s}}[1-p_z(s)]^{n^-_{i,s}}.$$

After capability changes to $s_1$, historical likelihoods are still evaluated at the capabilities under which the observations were generated. The resulting type posterior is combined with $p_z(s_1)$ to predict current success probabilities. Two additional conditions serve as controls: a misspecified-memory condition treats historical trials as observations generated at $s_1$, and a memory-reset condition discards the history and restarts from the prior.

能力变化为 $s_1$ 后，历史似然仍按产生观测时的能力计算。所得类型后验与 $p_z(s_1)$ 结合，用于预测当前成功概率。另外设置两个对照条件：记忆模型误设条件将历史试验视为在 $s_1$ 下产生的观测；记忆重置条件丢弃历史，从先验重新开始。

In the new stage, testing path $i$ incurs a cost $\kappa > 0$ and yields a binary success or failure observation. Tests leave path types unchanged, and task rewards are obtained only from terminal execution. At most $B$ tests are permitted, including repeated tests of the same path. After choosing STOP, the policy selects one path for a single execution and receives $R\cdot\operatorname{Bernoulli}(p_{Z_i}(s_1))$: a success yields $R$, and a failure yields 0. The task then terminates. Conditional on path type, test outcomes and the terminal execution outcome are independent. Historical testing costs are sunk. The objective is to maximize expected terminal reward minus the cost of new tests:

在新阶段，测试路径 $i$ 产生 $\kappa > 0$ 的成本，并得到成功或失败的二元观测。测试不改变路径类型，任务奖励只在最终执行时获得。最多允许 $B$ 次测试，包括对同一路径的重复测试。选择 STOP 后，策略选择一条路径执行一次，获得 $R\cdot\operatorname{Bernoulli}(p_{Z_i}(s_1))$：成功获得 $R$，失败获得 0，任务随即结束。给定路径类型时，测试结果和最终执行结果相互独立。历史测试成本为沉没成本。目标是最大化期望最终奖励减去新测试成本：

$$\max_\pi J(\pi)=\mathbb E_\pi[R\mathbf 1\{\text{terminal success}\}-\kappa N],\quad N\le B.$$

Here, $\pi$ denotes a policy and $N$ the number of tests performed in the new stage. Let $b$ denote the current beliefs over path types. The posterior predictive success probability of path $i$ is $\mu_i(b)=\sum_z b_i(z)p_z(s_1)$, and the value of stopping immediately and selecting the best path is $S(b)=R\max\{q_0,\mu_1(b),\ldots,\mu_K(b)\}$. With $h$ tests remaining, let $\tau(b,i,o)$ denote the Bayesian belief update after testing path $i$ and observing outcome $o$. The optimal value satisfies

其中，$\pi$ 表示策略，$N$ 表示新阶段执行的测试次数。令 $b$ 表示当前对路径类型的信念。路径 $i$ 的后验预测成功概率为 $\mu_i(b)=\sum_z b_i(z)p_z(s_1)$，立即停止并选择最佳路径的价值为 $S(b)=R\max\{q_0,\mu_1(b),\ldots,\mu_K(b)\}$。在剩余 $h$ 次测试时，令 $\tau(b,i,o)$ 表示测试路径 $i$ 并观测到结果 $o$ 后的贝叶斯信念更新。最优价值满足

$$V_0(b)=S(b),$$
$$V_h(b)=\max\left\{S(b),\ \max_i\left[-\kappa+\sum_{o\in\{0,1\}}P(o\mid b,i)V_{h-1}(\tau(b,i,o))\right]\right\}.$$

This Bellman recurrence compares immediate stopping with further testing. The test budget $B$ limits actual interactions, whereas the search budget introduced below limits planning computation. These budgets are accounted for separately.

这一贝尔曼递推比较立即停止与继续测试。测试预算 $B$ 限制实际交互次数，下文引入的搜索预算则限制规划计算量；两者分别计量。

## 4. Theoretical Analysis and Anytime Planning / 理论分析与可随时终止的规划

We will first establish that, under a correct model, free access to history, and the option to ignore historical information, the ex ante expected value satisfies the ordering: complete history ≥ any correctly interpreted historical summary ≥ memory reset. Retaining sufficient statistics yields the same value as retaining the complete history. This separates information loss due to compression from utility loss due to misinterpretation. When only the initial belief is misspecified and the subsequent observation model is correct, let $\delta$ denote the total variation distance between the true and assumed joint distributions over path types. We will derive the utility-loss bound $2(R+B\kappa)\delta$ for a policy optimized under the misspecified initial belief and examine its conditions using analytically tractable examples.

我们首先证明：在模型正确、可免费获取历史且可以选择忽略历史信息的条件下，事前期望价值满足如下顺序：完整历史 ≥ 任何被正确解释的历史摘要 ≥ 记忆重置。保留充分统计量所获得的价值与保留完整历史相同。这区分了压缩造成的信息损失与错误解释造成的效用损失。当只有初始信念被误设且后续观测模型正确时，令 $\delta$ 表示路径类型的真实联合分布与假定联合分布之间的全变差距离。我们将推导在误设初始信念下优化所得策略的效用损失界 $2(R+B\kappa)\delta$，并通过可解析的例子考察其适用条件。

One example directly illustrates the effect of misinterpreting historical failures. Consider one path with unknown type, prior $(0.2, 0.6, 0.2)$, historical success probabilities $(0, 0, 1)$, current success probabilities $(0, 1, 1)$, fallback success probability $0.5$, and $R=1, B=1, \kappa=0.11$. A single historical trial is retained. Under memory reset, the current predictive success probability is $0.8$. One additional test improves expected terminal reward by $0.1$, which is below its cost, so immediate execution yields expected net utility $0.8$. With correctly interpreted history, the historical trial succeeds with probability $0.2$, yielding terminal value $1$. Following a historical failure, the current predictive success probability is $0.75$, and an additional test yields expected terminal reward $0.875$. After subtracting the test cost, the unconditional expected net utility is $0.2+0.8(0.875-0.11)=0.812$. Treating the historical failure as a failure under the current capability assigns zero probability to path feasibility and causes the policy to select the fallback. Average net utility then falls to $0.2+0.8\times 0.5=0.6$. All values are evaluated under the same true model.

一个例子直接说明错误解释历史失败的影响。考虑一条类型未知的路径，先验为 $(0.2, 0.6, 0.2)$，历史成功概率为 $(0, 0, 1)$，当前成功概率为 $(0, 1, 1)$，备选路径成功概率为 $0.5$，且 $R=1, B=1, \kappa=0.11$。保留一次历史试验。在记忆重置下，当前预测成功概率为 $0.8$。增加一次测试能使期望最终奖励提高 $0.1$，低于测试成本，因此立即执行的期望净效用为 $0.8$。正确解释历史时，历史试验以 $0.2$ 的概率成功，得到最终价值 $1$。若历史试验失败，当前预测成功概率为 $0.75$，增加一次测试可得到 $0.875$ 的期望最终奖励。扣除测试成本后，无条件期望净效用为 $0.2+0.8(0.875-0.11)=0.812$。若将历史失败当作当前能力下的失败，则会将路径可行性的概率判为零，使策略选择备选路径。平均净效用因而降至 $0.2+0.8\times 0.5=0.6$。所有数值均在同一个真实模型下评估。

Another example illustrates the value of sequential testing. Let the capability-independent type have success probability $0.05$, the capability-dependent type have current success probability $0.8$ and prior probability $0.05$, and the fallback have success probability $0.5$. The permanently infeasible type has zero prior probability. Even after one successful test, the posterior predictive success probability remains below that of the fallback, so a myopic policy stops. A policy that tests once and performs a second test only after success achieves an expected net utility improvement of $0.00309375$ over immediate fallback selection when $\kappa=0.005$.

另一个例子说明连续测试的价值。设能力无关类型的成功概率为 $0.05$，能力依赖类型的当前成功概率为 $0.8$、先验概率为 $0.05$，备选路径的成功概率为 $0.5$；永久不可行类型的先验概率为零。即使一次测试成功，后验预测成功概率仍低于备选路径，因此短视策略会停止。当 $\kappa=0.005$ 时，先测试一次、仅在成功后进行第二次测试的策略，相比立即选择备选路径，期望净效用提高 $0.00309375$。

Each search node represents a belief over path types and maintains lower and upper value bounds. The initial lower bound is the value of stopping immediately. The initial upper bound is the value obtained if all path types were revealed at no cost:

每个搜索节点表示对路径类型的一个信念，并维护价值下界和上界。初始下界为立即停止的价值；初始上界为免费获知所有路径类型时获得的价值：

$$L_h^{(0)}(b)=S(b),$$
$$U_h^{(0)}(b)=R\mathbb E_b[\max\{q_0,p_{Z_1}(s_1),\ldots,p_{Z_K}(s_1)\}].$$

For $h=0$, both bounds are set to $S(b)$. Bounds are maintained for every candidate action, with unexpanded branches using the initial bounds. After expansion, the bounds of the success and failure branches are weighted by their observation probabilities, the test cost is subtracted, and the result is compared with the value of stopping. During backup, lower bounds increase monotonically and upper bounds decrease monotonically.

当 $h=0$ 时，两个界均设为 $S(b)$。每个候选动作都维护上下界，未展开分支使用初始界。展开后，成功和失败分支的界按各自观测概率加权，减去测试成本，再与停止价值比较。在回传过程中，下界单调增加，上界单调减少。

The upper bound can be computed directly from the distinct success probabilities, avoiding enumeration of all $3^K$ type assignments. Let $v_0=q_0 < v_1 < \cdots < v_m$ list the distinct success probabilities above the fallback, and let $F_i(t)=\sum_z b_i(z)\mathbf 1\{p_z(s_1)\le t\}$ denote the probability that path $i$ has success probability at most $t$. Then

上界可以根据不同的成功概率直接计算，无须枚举全部 $3^K$ 种类型组合。令 $v_0=q_0 < v_1 < \cdots < v_m$ 列出高于备选路径成功概率的各个不同成功概率，并令 $F_i(t)=\sum_z b_i(z)\mathbf 1\{p_z(s_1)\le t\}$ 表示路径 $i$ 的成功概率至多为 $t$ 的概率，则

$$U_h^{(0)}(b)/R=q_0+\sum_{j=1}^{m}(v_j-v_{j-1})\left[1-\prod_i F_i(v_{j-1})\right],\quad h>0.$$

When all paths share the same success-probability table for the three types, $m\le 3$, and the computation requires $O(Km)$ time. This product formula relies on posterior independence across path types; correlated types require evaluation under their joint distribution.

当所有路径对三种类型共用同一成功概率表时，$m\le 3$，计算所需时间为 $O(Km)$。这一乘积公式依赖路径类型后验之间的独立性；若类型相关，则需要在其联合分布下计算。

Each lower bound is stored with a complete executable policy that attains it. Unexpanded leaves stop immediately, while expanded nodes retain the selected action and the continuation policy for each outcome. Branches assigned zero probability by the model also stop immediately. Search first selects the action with the highest current upper bound, then expands the leaf within that action with the largest probability-weighted bound width. Ties are resolved by path index. An action is pruned when its upper bound is no greater than the best lower bound at the current node, while the complete policy realizing the lower bound is retained. When the root satisfies $U-L\le \varepsilon R$, or the computation budget is exhausted, the algorithm returns this policy and its value interval. Execution follows the recorded outcome-contingent continuation policies. By the monotonicity of the Bellman recurrence and the executability of the lower-bound policy,

每个下界都与达到该值的完整可执行策略一同保存。未展开的叶节点立即停止；已展开节点保留所选动作和每个结果对应的后续策略。模型赋予零概率的分支也立即停止。搜索首先选择当前上界最高的动作，再展开该动作下概率加权界宽最大的叶节点；相同时按路径索引决定。当某动作的上界不大于当前节点的最佳下界时，该动作被剪枝，同时保留实现该下界的完整策略。当根节点满足 $U-L\le \varepsilon R$ 或计算预算耗尽时，算法返回该策略及其价值区间。执行时遵循记录的、依观测结果而定的后续策略。根据贝尔曼递推的单调性和下界策略的可执行性，

$$0\le V^*(b)-J(\pi_L\mid b)\le U(b)-L(b).$$

Here, $V^*(b)$ is the optimal value and $\pi_L$ is the stored lower-bound policy. Thus, $U-L$ bounds the policy's suboptimality under the specified model. In model-misspecification experiments, we will report the planner's model-based interval and the utility gap under the true environment separately. The initial upper bound may be loose; experiments will quantify the resulting computational cost and the effect of pruning on search speed.

其中，$V^*(b)$ 为最优价值，$\pi_L$ 为保存的下界策略。因此，$U-L$ 给出指定模型下该策略的次优程度界。在模型误设实验中，我们将分别报告规划器基于模型的区间和真实环境下的效用差距。初始上界可能较松；实验将量化由此产生的计算成本及剪枝对搜索速度的影响。

## 5. Experimental Design and Evaluation Metrics / 实验设计与评估指标

The main experiment fixes $R=1, q_0=0.5$, with $K\in\{3,6,10\}$, $B\in\{1,2,3\}$, $n\in\{0,3\}$ historical trials per path, $p_C(s_0)=0.1$, $p_C(s_1)\in\{0.1,0.5,0.9\}$, $p_A=0.9$, and $\kappa\in\{0.005,0.02,0.1\}$. This gives $3\times3\times2\times3\times3=162$ parameter settings. For each path, a prior is sampled from $(0.2,0.6,0.2)$, $(0.6,0.3,0.1)$, $(0.1,0.2,0.7)$ using fixed random seeds. Each setting contains 20 independent tasks with historical observations, giving 3240 tasks in total. Priors are sampled within tasks, so the number of parameter settings remains 162. Development and final evaluation use disjoint seed sets, and results will be reported for all parameter settings.

主要实验固定 $R=1, q_0=0.5$，并设置 $K\in\{3,6,10\}$、$B\in\{1,2,3\}$、每条路径的历史试验次数 $n\in\{0,3\}$、$p_C(s_0)=0.1$、$p_C(s_1)\in\{0.1,0.5,0.9\}$、$p_A=0.9$ 和 $\kappa\in\{0.005,0.02,0.1\}$，共得到 $3\times3\times2\times3\times3=162$ 组参数设置。对每条路径，使用固定随机种子从 $(0.2,0.6,0.2)$、$(0.6,0.3,0.1)$、$(0.1,0.2,0.7)$ 中采样一个先验。每组设置包含 20 个带历史观测的独立任务，总计 3240 个任务。先验在任务内部采样，因此参数设置数仍为 162。开发和最终评估使用互不重叠的种子集，并报告所有参数设置的结果。

We will compare exact dynamic programming (DP), a myopic policy using the correct model, depth-2 and depth-3 lookahead, blinkered planning, iterative deepening under a matched computation budget, and our upper- and lower-bound search. For each path, blinkered planning solves the subproblem in which further tests are restricted to that path, selects the best initial action, and replans after each observation [1,2]. Memory-reset and misspecified-memory conditions provide additional controls for the effects of memory on decisions. All methods use the same terminal selection rule, test budget, and test cost. Computational comparisons include belief updates, bound computation, and initialization.

我们将比较精确动态规划（DP）、使用正确模型的短视策略、深度为 2 和 3 的前瞻、blinkered 规划、计算预算匹配的迭代加深，以及我们的上下界搜索。对于每条路径，blinkered 规划求解将进一步测试限制在该路径上的子问题，选择最佳初始动作，并在每次观测后重新规划 [1,2]。记忆重置和记忆模型误设条件为考察记忆对决策的影响提供额外对照。所有方法使用相同的最终选择规则、测试预算和测试成本。计算量比较包括信念更新、界的计算和初始化。

Larger-scale experiments use $K\in\{20,40\}$, $B\in\{4,6\}$, with $n=3$, $p_C(s_1)=0.5$, $\kappa=0.02$. Each computation of a successor belief for an outcome branch counts as one evaluation. Search budgets are $10^2,10^3,10^4$ such evaluations, with stopping tolerances $\varepsilon\in\{0.01,0.05\}$. Two stress tests will examine deviations from the modeling assumptions: shifting the planner's assumed current success probability for the capability-dependent type by $\pm 0.1$, and resampling the types of 10% of paths at the capability transition. These tests assess decisions under inaccurate success-probability estimates and changes in path types.

较大规模实验使用 $K\in\{20,40\}$、$B\in\{4,6\}$，并设 $n=3$、$p_C(s_1)=0.5$、$\kappa=0.02$。每次为一个结果分支计算后继信念计为一次评估。搜索预算分别为 $10^2,10^3,10^4$ 次此类评估，停止容差为 $\varepsilon\in\{0.01,0.05\}$。两个压力测试将考察偏离建模假设的情形：将规划器假定的能力依赖类型当前成功概率偏移 $\pm 0.1$；以及在能力转换时重新采样 10% 路径的类型。这些测试评估成功概率估计不准确及路径类型变化时的决策。

The primary metrics are exact expected net utility, regret relative to the optimal policy, number of tests, number of search expansions, single-core runtime, and bound width. The expected net utility of a complete policy can be computed by recursively enumerating its observation branches. Policies obtained under misspecified models are also evaluated under the true task-generating model. Small-scale experiments will verify $L\le V^*\le U$ for each task; larger-scale experiments will report the value of the available executable policy and the upper bound. Comparisons use paired utility differences on the same tasks, with confidence intervals across tasks. Runtime summaries include the median and upper quantiles.

主要指标包括精确期望净效用、相对于最优策略的遗憾、测试次数、搜索展开次数、单核运行时间和界宽。完整策略的期望净效用可通过递归枚举其观测分支计算。在误设模型下得到的策略也将在真实任务生成模型下评估。小规模实验将对每个任务验证 $L\le V^*\le U$；较大规模实验将报告可获得的可执行策略的价值及上界。比较采用同一任务上的配对效用差异，并给出跨任务的置信区间。运行时间汇总包括中位数和高分位数。

## 6. Team Responsibilities and Schedule / 团队分工与进度

| Member / 成员 | Responsibilities and deliverables / 职责与交付内容 |
| --- | --- |
| Minghao Liu (刘明昊), 260010126 | Establish the sufficiency of capability-specific success and failure counts; derive value relations between memory conditions and utility-loss bounds for misinterpreted history; verify counterexamples and proofs. <br>证明按能力区分的成功和失败次数的充分性；推导不同记忆条件之间的价值关系及错误解释历史的效用损失界；验证反例和证明。 |
| Haoyang Qin (秦浩洋), 260010154 | Specify the task-generating model and Bayesian updates; implement an independent enumeration-based verifier and exact dynamic programming. <br>明确任务生成模型和贝叶斯更新；实现独立的枚举验证器和精确动态规划。 |
| Shinan Yin (尹施南), 260010270 | Derive and implement bound updates; retain complete executable policies; verify the reported optimality-gap bounds. <br>推导并实现界的更新；保留完整可执行策略；验证所报告的最优性差距界。 |
| Xuhao Liang (梁栩豪), 250010046 | Implement myopic, blinkered, and lookahead baselines; conduct comparisons under matched computation budgets; analyze the utility gained from additional computation. <br>实现短视、blinkered 和前瞻基线；在匹配的计算预算下进行比较；分析额外计算带来的效用。 |
| Xi Qin (秦曦), 260010138 | Conduct the prespecified stress tests, statistical analysis, and analysis of applicability; compile the materials required to reproduce the experiments. <br>开展预先设定的压力测试、统计分析和适用性分析；整理复现实验所需的材料。 |

The proposal will be submitted on November 7. During November 8-14, we will finalize the model, complete the proofs, and verify the analytic examples. During November 15-21, we will complete exact dynamic programming and search with optimality-gap bounds. The main and larger-scale experiments will run during November 22-28. Stress tests, cross-checks, and report preparation will take place from November 29 to December 3. During December 4-5, we will consolidate the results, submit the report, and prepare all members for the oral defense. Each member will write the corresponding results section and cross-check another member's work.

提案将于 11 月 7 日提交。11 月 8—14 日确定模型、完成证明并验证解析例子。11 月 15—21 日完成精确动态规划和带最优性差距界的搜索。11 月 22—28 日运行主要实验和较大规模实验。11 月 29 日至 12 月 3 日进行压力测试、交叉检查和报告准备。12 月 4—5 日汇总结果、提交报告，并为所有成员准备口头答辩。每位成员撰写相应的结果部分，并交叉检查另一位成员的工作。

Independent tasks will run in parallel on multicore CPUs. Hardware and thread counts will be fixed when evaluating exact expected utility and search runtime. Implementation and debugging will be concentrated in the first three project days. Once checked, the task-generating model and analytic tests will be held fixed, followed by completion of proofs, batch computation, and analysis. Final deliverables will include the model specification, proofs, source code, fixed configurations, raw results, complete executable policies, and bound records.

独立任务将在多核 CPU 上并行运行。评估精确期望效用和搜索运行时间时，将固定硬件和线程数。实现和调试将集中在项目开始后的前三天。任务生成模型和解析测试经检查后将保持固定，随后完成证明、批量计算和分析。最终交付物包括模型说明、证明、源代码、固定配置、原始结果、完整可执行策略和界的记录。

## References / 参考文献

1. Nicholas Hay, Stuart Russell, David Tolpin, Solomon Eyal Shimony. *Selecting Computations: Theory and Applications.* UAI, 2012. [Official proceedings page / 官方会议论文页面](https://proceedings.mlr.press/r10/hay12a.html).
2. Frederick Callaway, Sayan Gul, Paul M. Krueger, Thomas L. Griffiths, Falk Lieder. *Learning to Select Computations.* UAI, 2018. [Institutional publication record / 机构出版记录](https://is.mpg.de/re/publications/callaway2018learning); [Full text / 全文](https://arxiv.org/html/1711.06892).
3. Ofir Nachum, Shixiang Gu, Honglak Lee, Sergey Levine. *Data-Efficient Hierarchical Reinforcement Learning.* NeurIPS, 2018. [Paper / 论文](https://arxiv.org/abs/1805.08296); [Authors' implementation / 作者实现](https://github.com/tensorflow/models/tree/master/research/efficient-hrl).
4. Beining Wu, Zihao Ding, Jun Huang. *ERRAND: Budgeted Maintenance of Agent Memory.* arXiv:2609.29545, 2026. [Full text, including Appendix E.10 / 全文，包括附录 E.10](https://arxiv.org/html/2609.29545v1).
5. Jundong Hu, Shekar Ramachandran. *The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents.* arXiv:2609.01852, 2026. [Paper / 论文](https://arxiv.org/abs/2609.01852).
6. Dylan Zhang, Yanshan Lin, Zhengkun Wu, Yihang Sun, Bingxuan Li, Dianqi Li, Hao Peng. *Useful Memories Become Faulty When Continuously Updated by LLMs.* arXiv:2605.12978, 2026. [Paper / 论文](https://arxiv.org/abs/2605.12978); [Authors' code and data / 作者的代码和数据](https://github.com/DylanZSZ/Memory-Collapse-Eval).
7. Trey Smith, Reid Simmons. *Heuristic Search Value Iteration for POMDPs.* UAI, 2004. [Author-hosted paper / 作者提供的论文](https://longhorizon.org/trey/papers/smith04_hsvi.pdf).
8. Stéphane Ross, Brahim Chaib-draa. *AEMS: An Anytime Online Search Algorithm for Approximate Policy Refinement in Large POMDPs.* IJCAI, 2007, pp. 2592-2597. [Official proceedings page / 官方会议论文页面](https://ijcai.org/proceedings/2007/417).
