# HybridEnzymeRetroPlanner 真实运行问题诊断与优化任务书
## 供 Codex 直接用于修改本地 ChemCrow 项目

---

## 0. 文档目的

本文件用于指导 Codex 对当前 `chemcrow-public` 项目中的多步酶促逆合成规划器继续进行修改和优化。

当前项目已经具备并通过单元测试的核心模块包括：

- `EnzymeRetroPathFinder`
- `HybridEnzymeRetroPlanner`
- AND 语义
- Reaction direction 过滤
- Cycle 过滤
- Forward anchor
- Top-K 基础逻辑
- Database + Model candidate fusion
- BOTH provenance
- Consensus bonus
- Masked scoring
- Model cache
- Depth / expansion termination
- Legacy LangChain 兼容
- Agent tool registration

并且真实运行：

```python
from chemcrow.tools import HybridEnzymeRetroPlanner

tool = HybridEnzymeRetroPlanner()

result = tool.run(
    "target=CCO;"
    "building_blocks=OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O;"
    "top_k=10;"
    "max_depth=10"
)

print(result)
```

已经可以返回：

```text
ok = true
```

并找到一条完整的 glucose -> ethanol 路线。

因此当前阶段的问题已经不是“Planner 未实现”，而是：

> **Planner 已经能够工作，但真实运行时部分 Hybrid 搜索逻辑没有被真正触发；Top-K、多模型融合、路线级评价、碳流合理性以及部分评分指标仍需继续优化。**

---

# 1. 当前真实运行结果

输入：

```text
Target:
CCO
= Ethanol

Building block:
OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O
= D-Glucose
```

真实输出：

```text
route_count = 1
step_count = 6
complete = true
all_noncurrency_leaves_solved = true
```

找到的路线为：

```text
D-glucopyranose
→ D-Glucono-1,5-lactone
→ D-gluconate
→ 2-Dehydro-3-deoxy-D-gluconate
→ Pyruvate
→ Acetaldehyde
→ Ethanol
```

这条路线整体具有生化意义，可视为 glucose oxidation / Entner-Doudoroff-like route。

主要反应包括：

```text
Glucose
→ Gluconolactone

Gluconolactone
→ Gluconate

Gluconate
→ 2-Dehydro-3-deoxy-D-gluconate

2-Dehydro-3-deoxy-D-gluconate
→ Pyruvate + D-Glyceraldehyde

Pyruvate
→ Acetaldehyde + CO2

Acetaldehyde
→ Ethanol
```

因此：

> **真实数据库上的多步 forward route search 已经跑通。**

---

# 2. 当前最关键问题：Hybrid Planner 实际退化成纯数据库 Forward Search

真实统计中出现：

```json
{
  "states_expanded": 0,
  "states_generated": 0,
  "database_candidates": 0,
  "model_candidates": 0,
  "model_calls": 0,
  "model_failures": 0,
  "backward_queries": 0,
  "forward_queries": 233,
  "forward_anchor_count": 350,
  "complete_routes_found": 1,
  "termination_reason": "search_exhausted_with_solutions"
}
```

这说明本次真实运行中：

```text
Hybrid backward search 没有执行
MooRetro / EnzRetro 没有调用
DB backward query 没有执行
AND-OR RouteState 没有扩展
Beam / Best-first search 没有真正展开
Candidate Fusion 没有进入真实运行
```

实际流程基本是：

```text
D-Glucose
   |
   v
Database Forward Anchor Search
   |
   v
... 6 steps ...
   |
   v
Ethanol
   |
   v
直接生成 solution
```

而不是目标架构：

```text
Building block                        Target
     |                                  |
     v                                  v
DB Forward Search                DB + Model Backward Search
     |                                  |
     +------------- meet ---------------+
                     |
                     v
                 Top-K routes
```

---

# 3. 需要排查：是否存在 Forward Target Early Return

当前：

```text
forward_anchor_depth = 6
```

真实找到的路线：

```text
step_count = 6
```

因此 forward search 很可能在第 6 层直接到达 Target。

需要重点定位是否存在类似：

```python
forward_map = forward_anchor_search(...)

if target.key in forward_map:
    solution = build_forward_solution(...)
    return result
```

如果有，这就是当前 Hybrid Search 未触发的首要原因。

这种逻辑会导致：

```text
只要 Forward Search 能到 target
→ backward search 永远不运行
→ model_calls 永远为 0
→ candidate fusion 永远不运行
→ Top-K 只得到 forward route
```

---

# 4. 必须修改：Forward route 只能加入 solution pool，不能直接 return

建议改为：

```python
solutions = []

forward_map = forward_anchor_search(...)

if target.key in forward_map:
    forward_solutions = collect_forward_solutions(...)
    solutions.extend(forward_solutions)

# 不能 return

hybrid_solutions = run_backward_hybrid_search(...)
solutions.extend(hybrid_solutions)

solutions = deduplicate_routes(solutions)
solutions = score_routes(solutions)
solutions = diversity_rerank(solutions)

return solutions[:top_k]
```

正确流程应是：

```text
Forward Search
     |
     v
找到 known route
     |
     v
加入 solution pool
     |
     v
继续 Hybrid backward search
     |
     v
DB candidates + Model candidates
     |
     v
Candidate Fusion
     |
     v
AND-OR Graph
     |
     v
Beam / Best-first
     |
     v
更多完整路线
     |
     v
Top-K rerank
```

---

# 5. 当前 Top-K 没有满足

用户输入：

```text
top_k = 10
```

但真实输出：

```text
route_count = 1
```

需要明确：

```text
top_k = 最大返回路线数
```

不能被实现成：

```text
找到第一条完整路径就停止
```

当前 config 中已有：

```text
solution_pool_multiplier = 3
```

应真正用于 solution pool。

例如：

```python
target_solution_pool_size = top_k * solution_pool_multiplier
```

对于：

```text
top_k = 10
```

可以先收集约：

```text
30 条完整候选路线
```

然后：

```text
去重
→ Route scoring
→ Diversity reranking
→ 返回 Top10
```

如果真实搜索空间最终只有 3 条，则允许返回 3 条，但必须明确：

```text
requested_top_k = 10
returned_route_count = 3
termination_reason = SEARCH_EXHAUSTED_WITH_SOLUTIONS
```

---

# 6. Forward Anchor 不应只保存一条路线

需要检查当前 `ForwardAnchorMap` 是否类似：

```python
forward_map[molecule_key] = one_best_route
```

如果是，则会损失路线多样性。

例如数据库可能存在：

```text
Glucose → A → Pyruvate
Glucose → B → C → Pyruvate
Glucose → D → Pyruvate
```

但如果只保存：

```text
forward_map[Pyruvate] = Route A
```

则 B/C 路线永久丢失。

建议：

```python
forward_map[molecule_key].routes
```

保存有限数量的低代价路径。

例如：

```python
max_forward_routes_per_node = 3
```

或：

```python
max_forward_routes_per_node = 5
```

同时按以下条件局部剪枝：

```text
route_cost
route_length
carbon_efficiency
reaction provenance
```

---

# 7. Forward Anchor 的职责需要明确

Forward Anchor 应主要承担：

```text
1. 从 building block 构建已知可达网络
2. 提供 heuristic distance
3. 提供 meet-in-the-middle anchor
4. 提供 database-only baseline solution
```

它不应完全替代 Hybrid backward search。

应保证：

```text
Forward route found
≠
Planner finished
```

而应：

```text
Forward route found
→ 加入高可信 solution
→ 继续 backward hybrid search
```

---

# 8. 必须增加真实 Hybrid Search Integration Test

虽然现有 unit test 已经验证：

```text
candidate fusion
model cache
BOTH provenance
```

但真实 glucose -> ethanol 运行中：

```text
model_calls = 0
model_candidates = 0
backward_queries = 0
states_expanded = 0
```

因此还没有验证：

```text
真实 SQLite
+
真实 MooRetro
+
真实 Candidate Fusion
```

建议增加一个 integration test，强制让 backward search 执行。

例如测试配置：

```text
forward_anchor_depth = 2 或 3
```

使 forward search 无法直接到达 Ethanol。

测试至少要求：

```python
assert stats["states_expanded"] > 0
assert stats["states_generated"] > 0
assert stats["backward_queries"] > 0
assert stats["model_calls"] > 0
assert stats["model_candidates"] > 0
```

如果存在数据库与模型匹配步骤，还应：

```python
assert any(
    step["provenance"] == "BOTH"
    for route in routes
    for step in route["steps"]
)
```

---

# 9. Top-K Diversity 需要真实验证

当前：

```text
route_count = 1
```

所以：

```text
diversity_penalty
diversity_adjusted_score
```

没有真正发挥作用。

需要真实生成多条路线后验证：

```text
Route 1
Route 2
...
Route N
```

不是同一条路线的轻微重复。

推荐路线相似度：

```text
Reaction-ID Jaccard
+
Intermediate Jaccard
+
EC-sequence similarity
```

最终：

```python
diversity_adjusted_score = (
    base_route_score
    - diversity_penalty
      * max_similarity_to_selected_routes
)
```

---

# 10. 当前路线暴露出的重要科学问题：缺少 Carbon Efficiency

真实路线：

```text
Glucose C6
→ ...
→ 2-Dehydro-3-deoxy-D-gluconate C6
→ Pyruvate C3 + D-Glyceraldehyde C3
→ Acetaldehyde C2 + CO2 C1
→ Ethanol C2
```

主目标碳流：

```text
C6 Glucose
→ C2 Ethanol
+ C3 D-Glyceraldehyde
+ C1 CO2
```

如果只看：

```text
能否生成 Ethanol
```

那么路线成立。

但如果目标是：

```text
高效从 glucose 生产 ethanol
```

则这条路线只有约：

```text
2 / 6
```

的起始碳最终进入目标 ethanol。

而经典发酵：

```text
Glucose
→ 2 Ethanol + 2 CO2
```

进入 ethanol 的碳为：

```text
4 / 6
```

因此路线评价必须加入：

```text
target carbon efficiency
target carbon yield
major organic byproduct penalty
```

---

# 11. 必须新增 Carbon Efficiency 指标

建议定义：

```python
target_carbon_efficiency =
    carbon_atoms_reaching_target
    / carbon_atoms_from_starting_substrate
```

如能恢复完整 stoichiometry，则进一步计算：

```python
target_carbon_yield = (
    target_stoich_coefficient * target_carbon_count
    /
    substrate_stoich_coefficient * substrate_carbon_count
)
```

如果当前第一版无法完整追踪全路径 stoichiometry，至少实现：

```text
main-carbon-retention score
```

并记录：

```text
carbon retained in target branch
carbon lost to organic coproducts
carbon lost to CO2 / small molecules
```

---

# 12. 必须新增 Major Organic Byproduct Penalty

真实 Step 4：

```text
2-Dehydro-3-deoxy-D-gluconate
→ Pyruvate + D-Glyceraldehyde
```

Planner 继续沿：

```text
Pyruvate
```

通向目标。

但：

```text
D-Glyceraldehyde
```

携带 C3，是重要有机 coproduct。

当前虽然保存：

```json
"other_products": [...]
```

但 route scoring 中没有明显惩罚这种大量碳流流失。

建议：

```python
major_byproduct_penalty = sum(
    byproduct_carbon_atoms
    / main_precursor_carbon_atoms
    for each_noncurrency_organic_byproduct
)
```

对以下弱惩罚或不惩罚：

```text
CO2
H2O
H+
Pi
PPi
ATP / ADP
NAD(P)+ / NAD(P)H
常见 cofactors
```

对以下强惩罚：

```text
C2+
C3+
明显有机副产物
大比例携带 substrate carbon skeleton 的 coproduct
```

---

# 13. 当前 `atom_conservation` 数值需要重点检查

真实结果中出现：

```text
Glucono-1,5-lactone
→ Gluconate

atom_conservation = 0.153846
```

这非常可疑，因为 C6 主骨架几乎完整保留。

另外：

```text
Acetaldehyde
→ Ethanol

atom_conservation = 0.666667
```

从重原子骨架看：

```text
C-C-O
→ C-C-O
```

也应该高度保留。

因此必须检查：

```python
atom_conservation(...)
```

当前实际算的是什么。

可能实际上是：

```text
fingerprint overlap
MCS ratio
structure overlap
edit fraction
```

却被命名为：

```text
atom_conservation
```

如果是这样，必须改名。

---

# 14. 区分真实 atom mapping 和近似 structural overlap

建议：

```python
if reaction_atom_mapping_available:
    atom_conservation = calculate_from_atom_mapping(...)
    atom_mapping_source = "REACTION_MAPPING"
else:
    atom_conservation = None
    approx_structural_overlap = calculate_structure_overlap(...)
```

评分中可以：

```text
真实 atom_conservation 权重更高
approx_structural_overlap 仅作 fallback
```

字段不要混淆。

---

# 15. 建议接入已有 Reaction Atom Mapping 数据

项目历史数据中已有 reaction atom mapping 数据。

如果 SQLite 当前未保存 mapping，建议新增：

```text
reaction_atom_mapping
```

表，或 sidecar mapping index。

建议字段：

```text
reaction_id
mapped_reaction_smiles
atom_mapping_json
mapping_source
```

用于：

```text
main precursor identification
carbon-flow tracking
target carbon efficiency
atom conservation
false shortcut rejection
```

---

# 16. 必须检查 `building_block_progress`

当前真实路线中出现类似：

```text
Step 1 building_block_progress = 1.0
Step 2 = 0.39
Step 3 = 0.09
...
Step 6 = 0.0
```

需要明确它到底表示：

```text
越接近 building block 越高
```

还是：

```text
越接近 target 越高
```

由于搜索是 backward，输出又是 forward，因此很容易出现方向定义混乱。

建议改名为：

```text
proximity_to_building_block
```

或：

```text
building_block_similarity
```

并写清楚：

```text
1.0 = very close to building block
0.0 = far from building block
```

确保在 heuristic 和 step scoring 中方向一致。

---

# 17. 当前 Toxicity / Yield 指标必须排查

此前相同 MooRetro 输入多次运行时：

```text
yield
toxicity
```

出现明显变化。

同时真实数据库输出中：

```text
D-Glucose toxicity ≈ 0.97
Pyruvate toxicity ≈ 0.95
```

如果该值被解释为：

```text
真实毒性概率
```

则明显不合理。

必须排查：

```text
1. toxicity 是否随机生成
2. yield 是否随机生成
3. 是否存在 random.random()
4. 是否存在 np.random.rand()
5. 是否来自 placeholder/mock
6. 是否每次调用重新采样
7. 是否真正调用独立预测模型
8. 分数语义是什么
9. 越高是越毒还是越安全
```

---

# 18. 在确认 Yield / Toxicity 之前建议关闭其评分权重

当前配置：

```json
"yield_score": 0.1,
"inverse_toxicity": 0.1
```

建议临时改为：

```python
yield_score_weight = 0.0
inverse_toxicity_weight = 0.0
```

直到：

```text
来源明确
语义明确
重复运行稳定
模型验证完成
```

否则相同 route 可能多次运行排序不同。

---

# 19. 增加 Metric Determinism Test

新增：

```text
tests/test_metric_determinism.py
```

对于同一 target 连续调用：

```python
r1 = model.predict(target)
r2 = model.predict(target)
r3 = model.predict(target)
```

检查：

```python
assert r1["score"] == r2["score"]
assert r1["feasibility"] == r2["feasibility"]
```

如果 yield / toxicity 应是确定值：

```python
assert r1["yield"] == r2["yield"]
assert r1["toxicity"] == r2["toxicity"]
```

如果模型本身为 stochastic：

```text
必须允许固定 random seed
```

用于可重复 route planning。

---

# 20. Building Block Alias 输出需要优化

用户输入：

```text
C00031 / D-Glucose
```

路线起点却显示：

```text
C30791 / D-glucopyranose
```

两者 canonical SMILES 相同，因此 graph node merge 是合理的。

问题只在展示层。

建议输出：

```json
{
  "input_building_block": {
    "id": "C00031",
    "name": "D-Glucose"
  },
  "matched_database_alias": {
    "id": "C30791",
    "name": "D-glucopyranose"
  }
}
```

最终 route 文本优先显示：

```text
D-Glucose (C00031)
[matched alias: D-glucopyranose / C30791]
```

不要取消 canonical SMILES node deduplication。

---

# 21. Direction 字段建议重新命名

真实 candidate_id 中出现：

```text
db:R36636:right>left:...
```

但 metadata 又包含：

```text
precursor_side = left
product_side = right
```

虽然可能表示：

```text
retrosynthesis direction = right -> left
```

但字符串容易误解。

建议显式输出：

```json
{
  "stored_forward_direction": "left_to_right",
  "retrosynthesis_direction": "right_to_left",
  "precursor_side": "left",
  "target_side": "right"
}
```

不要只依赖：

```text
right>left
```

字符串编码。

---

# 22. 当前 termination_reason 语义不准确

真实结果：

```text
states_expanded = 0
```

但：

```text
termination_reason = search_exhausted_with_solutions
```

实际上 backward search 根本没有展开。

如果 forward target 直接命中，应使用：

```text
FORWARD_TARGET_FOUND
```

如果修改为继续 hybrid search，建议统一 termination reason：

```text
TOP_K_FOUND
SEARCH_EXHAUSTED_WITH_SOLUTIONS
SEARCH_EXHAUSTED_NO_SOLUTION
MAX_EXPANSIONS_REACHED
MAX_DEPTH_REACHED
FORWARD_TARGET_FOUND
INVALID_INPUT
MODEL_FAILURE
```

统计字段必须反映真实执行过程。

---

# 23. 性能问题需要优化

真实运行：

```text
elapsed_seconds ≈ 45.9
forward_queries = 233
forward_anchor_count = 350
model_calls = 0
```

约 46 秒几乎全花在数据库 forward traversal。

对于约：

```text
29159 reactions
42891 metabolites
```

的 SQLite 数据库，233 次 query 偏慢。

---

# 24. 建议建立内存 Forward / Backward Adjacency Index

Planner 初始化时一次加载：

```python
forward_index = {
    metabolite_key: [ReactionRecord, ...]
}

backward_index = {
    metabolite_key: [ReactionRecord, ...]
}
```

或者：

```text
metabolite -> reaction ids
reaction id -> participants
```

搜索流程从：

```text
每扩展一个节点
→ SQL query
```

改为：

```text
startup/load
→ in-memory graph traversal
```

以减少大量 SQLite 调用。

---

# 25. 检查 SQLite Index

如果仍然使用 SQL 动态查询，至少确认以下字段有索引：

```text
reaction_participants.metabolite_id
reaction_participants.reaction_id
reactions.id
metabolites.id
canonical_smiles
```

使用：

```sql
EXPLAIN QUERY PLAN
```

确保没有频繁 full table scan。

---

# 26. 增加 Forward Anchor Persistent Cache

如果用户经常从：

```text
D-Glucose
```

开始规划，可以缓存：

```python
forward_anchor_cache[
    (
        building_block_key,
        forward_anchor_depth,
        currency_config_hash,
        reaction_direction_config
    )
]
```

避免每次重新运行数百次 forward query。

---

# 27. 需要增加真实数据库 R04410 False Shortcut Regression Test

数据库存在：

```text
R04410

C04164 + Ethanol
<=>
C06359 + D-Glucose
```

一个 naive planner 很容易因为同一 reaction 中出现：

```text
Ethanol
Glucose
```

错误认为：

```text
Glucose → Ethanol
```

是一条一步路线。

必须新增：

```python
def test_r04410_is_not_direct_glucose_to_ethanol_route():
    ...
```

要求：

```text
除非所有 required noncurrency precursors 均被满足，
并且 main-carbon / atom-flow 判定成立，
否则不能将 R04410 作为 glucose -> ethanol 一步完整路线。
```

---

# 28. Main Precursor 判定优先级

推荐：

```text
1. Reaction atom mapping
2. Target carbon atom tracing
3. Heavy atom conservation
4. MCS
5. Structural similarity fallback
```

不能仅根据：

```text
SMILES similarity
```

判断 main precursor。

---

# 29. 多产物反应应显式记录 Carbon Branch

例如：

```text
KDG
→ Pyruvate + Glyceraldehyde
```

需要记录：

```json
{
  "main_target_product": "Pyruvate",
  "other_products": ["D-Glyceraldehyde"],
  "target_atom_fraction": ...,
  "carbon_retained_fraction": ...,
  "major_coproduct_penalty": ...
}
```

这样 Route Evaluator 才能衡量资源利用效率。

---

# 30. Route Scoring 建议分三个层次

不要把所有指标混成一个 step score。

## A. Reaction-level quality

```text
database support
model confidence
feasibility
EC consistency
direction confidence
atom conservation
```

## B. Route-level production quality

```text
route length
target carbon efficiency
major byproduct burden
cofactor burden
model-only step count
generic compound count
```

## C. Diversity / novelty

```text
reaction sequence diversity
intermediate diversity
novel model-only reaction count
```

最终：

```python
route_score = (
    reaction_quality_score
    + production_efficiency_score
    + route_prior_score
)
```

之后单独执行：

```python
diversity_adjusted_score = diversity_rerank(...)
```

---

# 31. Toxicity 不应直接作为无来源的 molecule scalar

如果继续保留 toxicity，必须明确：

```text
toxicity 模型是什么
endpoint 是什么
score 范围是什么
越高是否越毒
是否 deterministic
```

建议字段：

```json
{
  "toxicity_score": ...,
  "toxicity_model": ...,
  "toxicity_endpoint": ...,
  "toxicity_score_direction": "higher_is_more_toxic"
}
```

在未明确前，不应用于核心搜索排序。

---

# 32. Model-only Reaction 需要更强约束

当前 config 已有：

```text
max_model_only_steps = 5
model_only_penalty = 0.25
```

需要确认真实生效。

建议增加：

```python
min_model_score
min_feasibility
max_consecutive_model_only_steps
```

避免出现：

```text
连续多个低置信 model-only reaction
```

却仍被拼成完整路线。

---

# 33. Candidate Fusion 建议采用 Structure + EC 双重标准

真实 Hybrid Search 启动后，DB candidate 与 Model candidate 的匹配应综合：

```text
main precursor canonical identity
noncurrency precursor set overlap
EC prefix compatibility
reaction direction compatibility
target identity
```

建议：

```python
fusion_match_score = (
    w_structure * precursor_set_similarity
    + w_ec * ec_compatibility
    + w_direction * direction_compatibility
)
```

---

# 34. 防止错误 BOTH Provenance

不能因为：

```text
某一个 precursor 相同
```

就标为：

```text
BOTH
```

至少应同时满足：

```text
target 一致
main precursor 一致
reaction direction compatible
EC compatible
```

必要时再加：

```text
reaction center similarity
```

---

# 35. 增加真实 Hybrid Fusion Integration Test

选择一个实际目标，使：

```text
MooRetro predicted precursor
```

在真实数据库中存在对应反应。

运行：

```python
result = HybridEnzymeRetroPlanner(...)
```

要求：

```python
assert result["stats"]["model_calls"] > 0
assert result["stats"]["backward_queries"] > 0
assert any(
    step["provenance"] == "BOTH"
    for route in result["routes"]
    for step in route["steps"]
)
```

不要只依赖 mock fusion test。

---

# 36. Beam Search 需要真实验证

当前真实运行：

```text
states_expanded = 0
states_generated = 0
beam_states_pruned = 0
```

因此：

```text
candidate_beam_width = 20
global_beam_width = 50
```

还没有真实发挥作用。

必须通过强制 backward search 验证：

```text
states_generated > 0
beam_states_pruned >= 0
```

最好设计一个分支足够多的 integration case，使：

```text
states_generated > global_beam_width
beam_states_pruned > 0
```

---

# 37. Best-first Priority 需要真实验证

开启：

```text
include_search_trace = true
```

记录：

```json
{
  "state_id": ...,
  "g_cost": ...,
  "h_cost": ...,
  "f_cost": ...,
  "expanded_molecule": ...
}
```

确认：

```text
低 f_cost state 优先被展开
```

否则即使使用 `heapq`，也可能 priority key 实现错误。

---

# 38. Heuristic 需要做真实 Ablation

当前：

```text
heuristic_weight = 0.35
```

但真实 backward search 未运行。

后续至少比较：

```text
heuristic_weight = 0
heuristic_weight = 0.35
heuristic_weight = 1.0
```

统计：

```text
states_expanded
elapsed_time
complete_routes_found
best_route_cost
route diversity
```

用于后续实验分析。

---

# 39. 建议增加 Planner Debug Mode

支持：

```text
debug=true
```

或：

```text
include_search_trace=true
```

输出阶段性统计：

```text
PHASE A: FORWARD ANCHOR
PHASE B: BACKWARD HYBRID SEARCH
PHASE C: SOLUTION MERGE
PHASE D: ROUTE SCORING
PHASE E: DIVERSITY RERANK
```

例如：

```text
FORWARD anchors = 350
FORWARD direct solutions = 1

BACKWARD states expanded = ...
DB candidates = ...
Model candidates = ...
Fused candidates = ...

HYBRID complete routes = ...
```

方便后续调试。

---

# 40. 建议拆分 Solution Statistics

增加：

```json
{
  "forward_direct_solutions": ...,
  "forward_anchor_stitched_solutions": ...,
  "hybrid_backward_solutions": ...,
  "database_only_routes": ...,
  "model_assisted_routes": ...,
  "consensus_routes": ...
}
```

这样可以快速判断 Planner 到底用了什么机制。

---

# 41. 建议建立三层测试体系

## Unit Tests

当前已有，继续保留。

## Integration Tests

### Real SQLite + Mock Model

验证：

```text
database service
AND-OR search
forward/backward graph
```

### Real Model + Small Test DB

验证：

```text
model client
candidate normalization
candidate fusion
```

## End-to-End Tests

### Real SQLite + Real MooRetro

例如：

```text
Glucose -> Ethanol
```

这类测试建议标记：

```text
slow
integration
e2e
```

避免普通 unittest 每次都跑 40+ 秒。

---

# 42. Glucose -> Ethanol E2E 验收标准

不能只检查：

```text
ok == true
```

至少检查：

```text
1. 最终 target 是 Ethanol
2. 起始 molecule 与输入 D-Glucose canonical-equivalent
3. 每条 complete route 所有 noncurrency leaves 均 solved
4. 不允许错误的 R04410 一步 shortcut
5. 至少一条 route 包含 Pyruvate -> Acetaldehyde -> Ethanol
6. forward database route 被保留
7. backward hybrid search 实际触发
8. model_calls > 0
9. backward_queries > 0
10. states_expanded > 0
```

第一阶段目标：

```text
route_count >= 2
```

成熟阶段再要求：

```text
route_count >= 10
```

如果真实合理路线不足 10 条，则返回实际数量并说明原因。

---

# 43. 修改后的理想 Stats 示例

以下数值仅为示例，不要求完全一致：

```json
{
  "route_count": 10,
  "stats": {
    "forward_anchor_count": 350,
    "forward_direct_solutions": 1,

    "states_expanded": 320,
    "states_generated": 1400,

    "forward_queries": 233,
    "backward_queries": 140,

    "database_candidates": 1800,

    "model_calls": 75,
    "model_candidates": 640,
    "model_cache_hits": 28,

    "consensus_candidates": 110,

    "beam_states_pruned": 900,

    "complete_routes_found": 25,

    "termination_reason": "TOP_K_FOUND"
  }
}
```

关键不是具体数值，而是：

```text
Hybrid phases 必须都有非零统计
```

---

# 44. 修改后的 Route 输出建议

建议增加：

```json
{
  "rank": 1,
  "complete": true,

  "route_score": 0.88,
  "route_cost": 1.2,

  "target_carbon_efficiency": 0.67,
  "major_byproduct_penalty": 0.10,

  "provenance_counts": {
    "DATABASE": 7,
    "MODEL": 1,
    "BOTH": 2
  },

  "steps": [
    {
      "reaction_id": "...",
      "provenance": "BOTH",
      "ec": "...",
      "step_score": ...,

      "atom_conservation": ...,
      "approx_structural_overlap": ...,

      "major_byproducts": [...],

      "evidence": {
        "database": {...},
        "model": {...}
      }
    }
  ]
}
```

---

# 45. Agent 层暂时不要作为主要修改重点

目前优先保证：

```text
HybridEnzymeRetroPlanner direct run
```

本身正确。

只有 Planner 能真正做到：

```text
Forward DB
+
Backward DB
+
MooRetro
+
Candidate Fusion
+
AND-OR state expansion
+
Beam / Best-first
+
Top-K
+
Route-level evaluation
```

之后再回到 ChemCrow Agent。

Agent 仍然应该：

```text
任务理解
→ 一次调用 Hybrid Planner
→ 获取 Top-K
→ 比较路线
→ 调节参数
→ 必要时重新调用 Planner
```

不要让 Agent 重新手工遍历 reaction nodes。

---

# 46. 修改优先级

## P0：必须立即处理

### P0-1
排查并移除 forward-target early return。

### P0-2
Forward direct route 只能加入 solution pool。

### P0-3
保证 Hybrid backward search 可以继续执行。

### P0-4
真实运行必须出现：

```text
states_expanded > 0
backward_queries > 0
model_calls > 0
```

### P0-5
`top_k=10` 时不能找到一条 forward route 就停止。

### P0-6
增加 R04410 false shortcut regression test。

---

## P1：科学合理性优化

### P1-1
增加 target carbon efficiency。

### P1-2
增加 major organic byproduct penalty。

### P1-3
重新检查 `atom_conservation` 定义。

### P1-4
尽可能接入真实 reaction atom mapping。

### P1-5
修复 D-Glucose / D-glucopyranose alias display。

### P1-6
明确 forward / retro direction 字段。

---

## P2：评分可靠性

### P2-1
排查 yield / toxicity 来源。

### P2-2
验证前将 yield / toxicity 权重置 0。

### P2-3
增加 metric determinism test。

### P2-4
增加 score provenance。

---

## P3：性能

### P3-1
建立 in-memory forward/backward adjacency。

### P3-2
检查 SQLite indexes。

### P3-3
增加 persistent forward-anchor cache。

### P3-4
减少重复 normalization / canonicalization。

---

## P4：研究与实验能力

### P4-1
保留 Database-only / Model-only / Hybrid 模式。

### P4-2
输出 provenance route statistics。

### P4-3
支持 heuristic ablation。

### P4-4
支持 beam-width ablation。

### P4-5
支持 diversity reranking ablation。

### P4-6
支持 search trace。

---

# 47. 推荐 Codex 修改顺序

## Step 1

定位：

```text
forward_anchor_search
target in forward_map
direct solution return
```

相关逻辑。

确认是否存在 early return。

修为：

```text
forward direct solution
→ solution pool
→ continue backward hybrid search
```

---

## Step 2

增加一个强制 Hybrid Search 测试：

```text
forward_anchor_depth = 2 或 3
```

确认：

```text
states_expanded > 0
model_calls > 0
backward_queries > 0
```

---

## Step 3

修复 Top-K solution collection。

确保：

```text
找到第一条解 != 停止搜索
```

---

## Step 4

新增真实 R04410 regression test。

---

## Step 5

增加 carbon efficiency 和 major byproduct penalty。

---

## Step 6

检查 `atom_conservation`。

如果暂时没有真实 mapping：

```text
rename approximate metric
```

再逐步接入真实 mapping。

---

## Step 7

排查 toxicity / yield。

验证前关闭其权重。

---

## Step 8

优化数据库遍历性能。

---

# 48. Codex 修改后必须报告

完成后必须输出：

```text
1. 修改了哪些文件
2. 新增了哪些文件
3. Early return 原来位于哪里
4. 修改后的 solution pool 逻辑
5. Hybrid backward search 如何保证执行
6. Top-K 如何收集
7. Carbon efficiency 如何定义
8. Major byproduct penalty 如何定义
9. Atom conservation 当前实际来源
10. Yield / toxicity 当前实际来源
11. 是否存在随机值
12. 性能优化做了什么
13. 新增哪些测试
14. 所有测试结果
15. Glucose -> Ethanol 新的真实输出
16. 仍存在的限制
```

---

# 49. 最终验收命令

至少运行：

```bash
python - <<'PY'
from chemcrow.tools import HybridEnzymeRetroPlanner

tool = HybridEnzymeRetroPlanner()

result = tool.run(
    "target=CCO;"
    "building_blocks=OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O;"
    "top_k=10;"
    "max_depth=10"
)

print(result)
PY
```

---

# 50. 最终验收重点

不能只检查：

```text
ok = true
```

必须确认 Planner 实际执行了：

```text
Forward Database Search
+
Backward Database Search
+
MooRetro / EnzRetro Prediction
+
Candidate Fusion
+
AND-OR State Expansion
+
Beam / Best-first Search
+
Solution Pool
+
Route-level Scoring
+
Diversity Reranking
```

统计至少应看到：

```text
states_expanded > 0
states_generated > 0
backward_queries > 0
model_calls > 0
model_candidates > 0
```

如果真实存在 DB/Model 共识步骤，还应看到：

```text
provenance = BOTH
```

---

# 51. 当前项目状态总结

当前版本已经真实完成：

```text
数据库多步路径搜索
+
Forward anchor
+
完整路线生成
+
基础路线评分
+
Reaction provenance
```

但尚未真正完成真实运行下的：

```text
DB + Model Hybrid Search
+
Top10 route generation
+
真实 atom-flow evaluation
+
target carbon efficiency
+
major byproduct-aware ranking
+
可靠 yield/toxicity scoring
```

下一阶段的核心目标不是继续新增更多 Tool，而是：

> **让已经存在的 `HybridEnzymeRetroPlanner` 真正同时使用 Forward Database、Backward Database、MooRetro/EnzRetro、Candidate Fusion、AND-OR Search、Beam/Best-first Search 和 Route Evaluation，并稳定输出具有科学意义和多样性的 Top-K 完整路径。**
