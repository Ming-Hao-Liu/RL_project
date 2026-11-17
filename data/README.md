# 正式数据

本目录只保留一份未压缩的正式数据。除本说明外，内容不纳入 Git。

| 目录 | 基础任务数 | 作业数 | 方法／检查点记录数 |
|---|---:|---:|---:|
| main | 3240 | 22140 | 32400 |
| large | 160 | 800 | 1920 |
| stress | 40，复用 main | 320 | 320 |

三组共 3400 个不同基础任务、23260 个作业、34640 条记录。large 的历史条件共享规定的先验抽样，stress 复用 main 的任务；方法、预算检查点和压力变体不是新的独立样本。

## 每组文件

- `manifest.json`：冻结的配置、种子、任务／作业摘要、源码摘要、依赖锁摘要和生成环境。
- `tasks.jsonl`：任务 ID、先验、历史计数及参数；不包含生成时的隐藏类型。
- `jobs.jsonl`：应执行的方法、记忆条件及预算组合。
- `records/<job_id>.jsonl`：结果、完整策略、模型值、真实效用、计算量、状态和检查点。
- `verification.json`：该组保存策略的复算结果及同模型精确参照检查数量。

每条 JSONL 行是一个 JSON 对象。策略中的 STOP 保存具体路径，TEST 保存成功和失败两个后续策略。`complete`、`tolerance`、`budget_exhausted` 分别表示计算完成、达到容差或耗尽规划预算；后两种状态仍返回可执行策略，不能据此当作无效数据删除。

方法别名和预算检查点保留在原作业格式内。它们有明确的实验语义，与整份目录的重复副本不同。分析时由方法、记忆条件、预算和任务 ID 选择对应记录。

## 来源与验证

原始 manifest、任务、作业和策略记录保持生成时字节内容；其中历史 Git/source 摘要是来源信息，不是当前仓库版本声明。文件摘要索引见 `../results/validation/data_integrity.json`，保存策略复算命令见项目 README。

统计表与图表统一位于 `../results/`，不在本目录保存第二份分析输出。新增实验使用新的运行目录；不要覆盖这些正式数据。

## planning：独立初始化确认

新种子20261124，160个基础任务，2240条方法记录（每任务14条），另有32个任务×4方法×3次重复的384条计时。任务名包含研究名与种子；与原large组分开。manifest冻结新配置、源码和生成提交；records每任务一个JSON数组，timing.jsonl保存原始重复。此组不使用旧CLI作业格式；用`python -m scripts.verify_planning`复算，不能把其记录混入旧large汇总。

## research：解析机制与有限校准

`configs/research.json`冻结范围，manifest记录配置、任务清单哈希、逐源码文件哈希、生成提交75ed79b和线程环境。所有记录未压缩、保留完整策略，不与旧主网格混算。

| 文件/组 | 行数/数量 | 数据含义 |
|---|---:|---|
| tasks.jsonl | 36 | 从main按参数和index选择的固定校准评价任务；复用历史，不新增任务总体样本 |
| jobs.jsonl | 1473 | grid1329、deep24、独立校准重复120 |
| truth.json | 36策略 | 固定任务的真实表DP参照，各只计算一次 |
| records/grid-*.json | 1329作业×2策略 | 完整21×21网格加2锚点，B1/2/3；exact/myopic |
| records/deep-*.json | 24作业×5策略 | exact/myopic/blinkered/cold/warm，完整问题定义和策略 |
| records/calibration-*.json | 120作业，8880策略 | 8640主要方法策略及240预定误差分离策略 |
| phase_timing.json | 24 | 已有planning中8任务各3次，单进程补测阶段耗时 |

每个结果JSON有`job`和`records`。机制组额外有`problem`（belief/spec）；校准组有`calibration`（样本量、重复、成功计数、估计值、有效样本数）。每个校准重复为五个概率参数估计表：C_old和三个C_current各m次，A两批合并2m次，共6m个校准观测；I结构固定0，无校准样本。独立合成标签仅用于此校准，不是评价任务隐藏类型。

策略记录字段：`policy`为可独立执行的树，`model_value/lower/upper`属于规划模型；`expected_utility/expected_tests/success_probability`属于真实评价模型；`evaluations`计后继更新，`planning_seconds/evaluation_seconds`分开计时；`status`保留正常停止状态。校准记录还保存`component`（both/history/future）、规划初始信念和规划表；真实条件模型由tasks和main的固定配置恢复。诊断切片中的单独误差不是相互可加的贡献。

阶段计时中`root_setup_seconds`包含根上界及动作结构，`myopic_seconds`为种子策略构造，`initialization_management_seconds`为其余初始化，`search_seconds`为后续计算；`total_seconds`逐条等于四者之和，`planning_wall`还包含调用包装和返回开销。`evaluations`为初始化更新，`search_evaluations`与其相加为`total_evaluations`。中位数分别计算，不能相加。

校准随机性单位是repeat；同一估计表的36任务不独立。机制网格点是确定性覆盖，不是总体抽样。每个固定策略价值是精确期望而非执行蒙特卡洛。核验入口`python -m scripts.verify_research`，哈希索引在`results/research/input_sha256.json`。

`research/deep_timing.json`补齐同24深任务、五方法、每方法三次的360条单进程计时。文件在测量前固定重复数、顺序种子2026111703、任务ID、预算10000和ε0；方法顺序随机，输入预先构造，缓存每次独立，数值库单线程。每条记录的策略、模型价值、界和更新数已与原机制记录核对。保存规划和评价耗时，不把双进程批处理耗时混入此组。它不增加基础任务或校准重复数。
