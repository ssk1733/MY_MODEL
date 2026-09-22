# MooRetro 正式小规模对比实验记录

## 1. 实验背景与目标

本次实验的目标是在**不改变 MooRetro 原始方法框架**的前提下，将正式的 PaRoutes 数据集接入现有代码，完成一版基于正式 benchmark 数据的小规模对比实验，并输出可用于与论文 `Table 1` 做形式对照的统计结果。

需要特别说明的是：

- 本次实验**只运行 MooRetro 自身代码**；
- 不运行、也不复现论文中使用的 Retro*；
- 论文中的 `Table 1` 主要作为**数据组织形式、统计口径与展示方式**的参考；
- 若结果存在差异，以服务于原始 MooRetro 代码与实验链路打通为优先目标。

本次工作重点包括：

1. 阅读和梳理 MooRetro 与 PaRoutes 相关代码结构；
2. 将原先适合单目标运行的 MooRetro 扩展为支持正式数据集批量实验；
3. 设计单目标与数据集级别的输出结构；
4. 使用正式 `n1` / `n5` 数据完成小规模真实实验；
5. 记录实验过程、修改内容、结果文件位置以及 Table 1 风格对比表述。

---

## 2. 项目与数据位置说明

### 2.1 MooRetro 项目位置

- 主项目目录：`/root/autodl-tmp/MooRetro-master`
- 多步逆合成核心代码：`/root/autodl-tmp/MooRetro-master/src`
- 单步逆合成模型相关代码：`/root/autodl-tmp/MooRetro-master/enzretro`

### 2.2 PaRoutes 参考项目与正式数据位置

- PaRoutes 参考项目：`/root/PaRoutes`
- 正式数据目录：`/root/PaRoutes/data/resource`

本次实验实际使用的正式数据文件包括：

- `n1-targets.txt`
- `n1-stock.txt`
- `n1-routes.json`
- `n5-targets.txt`
- `n5-stock.txt`
- `n5-routes.json`

对应完整路径：

- `/root/PaRoutes/data/resource/n1-targets.txt`
- `/root/PaRoutes/data/resource/n1-stock.txt`
- `/root/PaRoutes/data/resource/n1-routes.json`
- `/root/PaRoutes/data/resource/n5-targets.txt`
- `/root/PaRoutes/data/resource/n5-stock.txt`
- `/root/PaRoutes/data/resource/n5-routes.json`

---

## 3. 本次代码修改内容

本次真正涉及的核心修改文件如下：

1. `/root/autodl-tmp/MooRetro-master/test.py`
2. `/root/autodl-tmp/MooRetro-master/config/config.py`
3. `/root/autodl-tmp/MooRetro-master/src/analyzer.py`
4. `/root/autodl-tmp/MooRetro-master/utils/single_retro_fn.py`
5. `/root/autodl-tmp/MooRetro-master/utils/chem.py`

下面分别记录修改内容与目的。

### 3.1 `test.py`：扩展为数据集批量实验入口

`test.py` 是本次改动最大的文件，也是整个正式实验能跑通的核心。

主要新增和调整内容如下：

#### （1）新增命令行参数入口
原本运行方式更偏向单目标测试，本次新增了 `dataset` 模式，可通过命令行传入：

- `--targets-file`
- `--stock-file`
- `--references-file`
- `--dataset-name`
- `--output-root`
- `--limit`
- `--offset`
- `--max-iterations`
- `--max-simulation-depth`
- `--disable-graphs`

从而支持直接对正式数据文件进行批量实验。

#### （2）新增数据集实验主流程
增加了 `run_dataset_experiment`，负责：

- 读取正式目标分子列表；
- 读取商业可得分子列表；
- 对每个目标自动构造 `MCTSConfig`；
- 调用 MooRetro 原有搜索流程；
- 保存单目标结果；
- 汇总生成数据集级结果文件。

#### （3）新增单目标汇总文件 `run_summary.json`
每个目标运行结束后，自动输出单目标摘要信息，包括：

- 目标分子；
- 商业分子数；
- 是否 solved；
- 完整路径数；
- 耗时；
- 树节点规模；
- Pareto front 大小；
- 最佳路径长度；
- 最优路径多目标指标；
- PaRoutes 兼容路径导出文件位置。

#### （4）新增数据集级结果文件
在每个数据集实验目录下自动生成：

- `dataset_summary.json`
- `dataset_summary.csv`
- `table1_like_summary.json`
- `run_metadata.json`

其中：

- `dataset_summary.json/csv` 保存逐目标明细；
- `table1_like_summary.json` 保存 Table 1 风格汇总统计；
- `run_metadata.json` 保存本次实验参数、数据来源和统计口径说明。

#### （5）新增 Table 1 风格摘要逻辑
增加了 `build_table1_like_summary`，用于从所有目标运行结果中统计：

- 总目标数；
- solved 目标数；
- resolution rate；
- 平均耗时；
- 中位完整路径数；
- solved 样本平均最佳路径长度。

这样可以使 MooRetro 的结果在展示形式上与论文 `Table 1` 尽可能对齐。

#### （6）新增 PaRoutes 兼容路线导出
增加了路径转换层，将 MooRetro 输出的完整路径转换为递归 route tree，并导出为 `paroutes_routes.json` / `paroutes_routes.json.gz`。

这里特别修正了路径构造方向：

- 将 `molecule_nodes` 与 `reaction_nodes` 倒序；
- 使导出的 route tree 从**目标分子向商业可得分子**展开；
- 从而与 PaRoutes 兼容分析结构一致。

#### （7）改进异常处理与鲁棒性
在批量实验中，如果某个目标运行失败，不会中断整个数据集实验，而是：

- 记录失败目标；
- 输出该目标的 `run_summary.json`；
- 将错误信息写入结果中；
- 继续运行后续目标。

#### （8）使用 lazy import 避免启动期依赖报错
将 `single_fn` 的导入移动到函数内部，避免因为 `SmilesPE` 等依赖问题导致程序刚启动就失败。

---

### 3.2 `config/config.py`：使配置适配批量实验

对 `MCTSConfig` 进行了适配性改造。

主要调整如下：

#### （1）`commercial_molecules` 使用 `default_factory`
避免可变默认参数在批量实验中带来的潜在风险。

#### （2）`result_path` 改为可选
允许：

- 单目标运行时自动生成结果目录；
- 数据集运行时为每个目标显式指定独立输出目录。

#### （3）增加 `__post_init__`
在初始化时自动：

- 过滤空的商业分子字符串；
- 若未提供 `result_path`，则按时间戳自动生成默认输出目录。

这样可以保证批量运行时配置更稳定。

---

### 3.3 `src/analyzer.py`：将 `graphviz` 改为可选依赖

为避免服务器环境或批量实验环境中因为缺少 `graphviz` 而直接报错，本次将其改成了可选导入：

- 若已安装，则正常使用；
- 若未安装，则不影响主实验流程。

这样做的作用是：

- 批量实验可以不依赖图形绘制；
- 提升实验环境的兼容性与稳定性。

---

### 3.4 `utils/single_retro_fn.py`：为单步模型增加缓存

在正式数据实验中，单步逆合成模型如果每次都重新初始化，会导致整体运行极慢。

因此本次新增 `_MODEL_CACHE`，将不同配置下的单步模型缓存起来。

作用：

- 同一种 `single_fn_type + ec_level + device` 组合只初始化一次；
- 后续目标直接复用模型实例；
- 显著降低批量实验中的重复加载开销。

这项优化是正式实验能顺利跑完的重要前提之一。

---

### 3.5 `utils/chem.py`：为商业分子指纹构建增加缓存

正式 `n1` / `n5` 数据中的 stock 分子数量都在一万三千以上，如果每次都重新计算 RDKit 指纹，会带来很大的时间开销。

因此本次新增：

- `_FP_CACHE`
- `_build_fingerprints`

在 `similarity_calculator` 中复用已缓存的分子指纹。

作用：

- 对同一批商业可得分子，指纹只构建一次；
- 后续相似度比较直接复用；
- 显著提升批量运行性能。

这也是正式实验能完成的关键优化之一。

---

## 4. 依赖问题与处理记录

在实验过程中，还处理了若干运行环境问题：

1. `graphviz` 缺失：通过将其改为可选依赖解决；
2. `pymoo` 缺失：已安装；
3. `pandas` 缺失：已安装；
4. `rdkit` 缺失：已安装；
5. `SmilesPE` 导致启动报错：通过 lazy import 避免在程序启动期触发导入错误；
6. 数据汇总中 `None` 参与数值计算导致异常：在 `build_table1_like_summary` 中加入空值过滤逻辑。

---

## 5. 实验统计口径说明

### 5.1 solved 定义
本次实验中，`solved` 的定义为：

> 至少找到一条完整闭合的逆合成路径。

该统计口径已写入各数据集目录下的 `run_metadata.json`。

### 5.2 Table 1 风格统计项
本次用于与论文 `Table 1` 做形式对照的统计项包括：

- `total_targets`：总目标数
- `solved_targets`：求解成功目标数
- `resolution_rate`：求解率
- `mean_elapsed_time_sec`：平均运行时间
- `median_complete_paths`：完整路径数中位数
- `mean_best_path_length`：成功目标的平均最佳路径长度

需要注意的是：

- 这些统计项是为了与论文表格形式对齐；
- 但 MooRetro 的搜索机制与 Retro* 不同，因此这里只是**形式对齐**，不是方法完全一致下的严格数值复现。

---

## 6. 正式小规模实验设置

本次完成的是**正式数据上的小规模对比实验**，而非全量 benchmark。

### 6.1 数据来源

- `n1` 正式数据
- `n5` 正式数据

### 6.2 实验规模

- `n1`：前 3 个正式目标
- `n5`：前 3 个正式目标

### 6.3 运行参数

- `single_fn_type = biocyc`
- `max_iterations = 1`
- `max_simulation_depth = 10`
- `disable_graphs = true`

之所以采用这样的设置，是因为正式数据单目标运行耗时较长，因此先以小规模、低迭代的方式完成正式实验链路验证，并生成第一版真实结果。

---

## 7. 实验输出目录与文件说明

### 7.1 总输出目录

- `/root/autodl-tmp/MooRetro-master/results/paroutes_runs/`

### 7.2 数据集级目录

#### `n1_smallscale`
- `/root/autodl-tmp/MooRetro-master/results/paroutes_runs/n1_smallscale/`

#### `n5_smallscale`
- `/root/autodl-tmp/MooRetro-master/results/paroutes_runs/n5_smallscale/`

### 7.3 每个数据集目录下的核心文件

#### 数据集级文件

1. `table1_like_summary.json`
   - Table 1 风格汇总摘要
2. `dataset_summary.json`
   - 所有目标的逐条明细（JSON）
3. `dataset_summary.csv`
   - 所有目标的逐条明细（CSV）
4. `run_metadata.json`
   - 本次实验配置、统计口径和说明
5. `paroutes_routes.json.gz`
   - 所有目标导出的 PaRoutes 兼容路线集合

#### 单目标目录中的常见文件

每个目标目录下可能包含：

- `run_summary.json`
- `paroutes_routes.json`
- `synthesis_tree_all_data.json`
- `pruned/synthesis_tree_pruned_data.json`
- `pruned/pathwaySearch.json`
- `pruned/pathwayEvaluation_front0_0.json`
- `andor_tree_*.log`

说明：

- `run_summary.json`：单目标最重要的摘要文件；
- `paroutes_routes.json`：单目标的 PaRoutes 兼容路线导出；
- `synthesis_tree_all_data.json`：完整搜索树数据；
- `pruned` 目录：剪枝后的路径搜索与评价结果；
- `andor_tree_*.log`：运行日志。

---

## 8. 实验结果记录

### 8.1 `n1_smallscale` 结果

#### 数据集级结果

- 数据集：`n1_smallscale`
- 总目标数：3
- solved 目标数：2
- resolution rate：66.67%
- 平均耗时：86.247 秒
- median complete paths：1
- 成功样本平均最佳路径长度：2.0
- 参考路线文件：`/root/PaRoutes/data/resource/n1-routes.json`

#### 逐目标结果

1. 目标 1
   - solved：否
   - 完整路径数：0
   - 耗时：147.43 秒
   - 树节点数：35

2. 目标 2
   - solved：是
   - 完整路径数：1
   - 最佳路径长度：3
   - 耗时：73.597 秒

3. 目标 3
   - solved：是
   - 完整路径数：1
   - 最佳路径长度：1
   - 耗时：37.715 秒

### 8.2 `n5_smallscale` 结果

#### 数据集级结果

- 数据集：`n5_smallscale`
- 总目标数：3
- solved 目标数：1
- resolution rate：33.33%
- 平均耗时：135.613 秒
- median complete paths：0
- 成功样本平均最佳路径长度：2.0
- 参考路线文件：`/root/PaRoutes/data/resource/n5-routes.json`

#### 逐目标结果

1. 目标 1
   - solved：是
   - 完整路径数：1
   - 最佳路径长度：2
   - 耗时：87.934 秒

2. 目标 2
   - solved：否
   - 完整路径数：0
   - 耗时：109.87 秒

3. 目标 3
   - solved：否
   - 完整路径数：0
   - 耗时：209.036 秒

---

## 9. 对本次结果的理解

本次实验说明了以下几点：

1. MooRetro 已经能够直接读取并运行正式 PaRoutes 数据；
2. 在不改变原始 Pareto + MCTS 搜索框架的情况下，已经能够在正式 benchmark 数据上找到完整逆合成路径；
3. 结果表明 MooRetro 在正式数据上的确具有可运行性和一定的求解能力；
4. 当前的主要问题不是“不能跑”，而是“正式数据计算开销较大”，尤其是困难目标在低迭代下也会消耗较长时间；
5. 因此这次实验更适合定位为：
   - 正式数据链路打通；
   - 方法可运行性验证；
   - Table 1 风格初步对照；
   - 后续更大规模实验的基础版本。

---

## 10. 与论文 Table 1 的对比表述建议

下面给出适合写入实验记录、汇报或论文草稿中的正式表述。

### 10.1 实验定位表述

为了将本文提出的 MooRetro 多目标生物逆合成方法与论文中 `Table 1` 所采用的 PaRoutes benchmark 做形式上的对照，本研究在保留 MooRetro 原始搜索框架不变的前提下，将正式 PaRoutes 数据接入当前代码，并完成了一版小规模正式实验。需要说明的是，论文 `Table 1` 中使用的基线方法为 Retro*，而本次实验仅运行 MooRetro 自身，不复现 Retro*，因此本实验的主要目的在于验证 MooRetro 在正式 benchmark 数据上的可运行性、路线生成能力以及初步 solved 表现。

### 10.2 为什么称为 Table 1 风格对照

由于 MooRetro 采用的是基于 Pareto 选择的多目标 MCTS，并使用 And-Or Tree 表示分子与反应节点，因此其内部搜索机制与 Retro* 并不完全一致。为保证方法本体不被破坏，本次工作没有强行将 MooRetro 改造成 Retro* 风格实现，而是仅在输入数据组织、结果汇总口径与路线导出格式上与 PaRoutes / `Table 1` 做对齐。因此，本次结果应理解为“基于正式 PaRoutes 数据的 Table 1 风格对照结果”，而非对论文中 Retro* 数值的严格一比一复现实验。

### 10.3 本次小规模正式实验结论表述

在正式 PaRoutes 数据上进行的小规模实验结果表明，MooRetro 已能够直接处理 benchmark 中的目标分子与商业可得分子集合，并成功输出完整逆合成路线。在 `n1` 小规模实验中，MooRetro 对 3 个正式目标中的 2 个成功找到完整闭合路径，solved rate 为 `66.67%`；在 `n5` 小规模实验中，对 3 个正式目标中的 1 个找到完整闭合路径，solved rate 为 `33.33%`。这说明本文方法在正式 benchmark 条件下具备可运行性和一定的求解能力，但由于当前仅采用 1 次 MCTS 迭代且样本规模较小，因此这些结果主要用于验证实验链路与方法有效性，而不作为全量 benchmark 的最终结论。

### 10.4 与论文 Table 1 的关系表述

论文 `Table 1` 更适合作为 benchmark 统计口径与展示方式的参考。本次实验参考了其“数据集级 solved 表现”的组织方式，构建了 `table1_like_summary.json`，统计了总目标数、solved 目标数、resolution rate、平均耗时和平均最佳路径长度等指标。由于方法本体并非 Retro*，且本次实验仅为正式数据上的小规模运行，因此这些结果应视为面向论文 `Table 1` 的初步对照，而不是最终的同规模定量比较。

---

## 11. 本次工作总结

本次工作在不改变 MooRetro 原始多目标 MCTS + Pareto 选择框架的前提下，将正式 PaRoutes 数据集接入到现有代码中，完成了从单目标人工测试到正式数据小规模批量实验的扩展。为此，主要修改了 `test.py`、`config/config.py`、`src/analyzer.py`、`utils/single_retro_fn.py` 和 `utils/chem.py` 五个文件。其中，`test.py` 被扩展为支持数据集模式运行，能够批量读取 target/stock/reference 数据，逐目标调用 MooRetro 搜索，并自动生成单目标摘要、数据集级汇总以及 Table 1 风格统计文件；`config/config.py` 被调整为更适合批量实验的配置结构；`src/analyzer.py` 被改为支持在无 `graphviz` 环境下运行；`utils/single_retro_fn.py` 与 `utils/chem.py` 分别增加了模型缓存和分子指纹缓存，用于降低正式数据实验中的重复计算开销。

在此基础上，使用 `/root/PaRoutes/data/resource/` 中的正式数据完成了 `n1` 和 `n5` 的小规模实验，各取前 3 个目标分子进行测试。结果表明，MooRetro 已能够在正式 benchmark 数据上直接运行，并输出完整逆合成路径及其汇总统计。其中，`n1_smallscale` 的 solved rate 为 `66.67%`，`n5_smallscale` 的 solved rate 为 `33.33%`。这些结果已经保存在 `results/paroutes_runs/` 目录下，包含单目标结果、数据集汇总、运行元数据和 PaRoutes 兼容路线文件。需要说明的是，本次实验参考了论文 `Table 1` 的展示形式与统计口径，但并未运行 Retro*，因此应将本次结果视为“基于正式 PaRoutes 数据的 Table 1 风格对照结果”，而非对论文数值的严格复现。
