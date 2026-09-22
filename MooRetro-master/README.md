# Synthesis Pathway Design based on AI

1. Pathway Representation: And-Or Tree
    - 'And': Reaction Node
    - 'Or' : Molecule Node
2. Expansion: 
    - Known Database: KEGG / Metanetx
    - Unknown Methods: Retrosynthesis Methods
3. Search Methods: 
    - MCTS
        - Selection
        - Expansion
        - Simulation
        - Background
4. Evaluation: Multi-objective & Pareto Front
    - Pathway Length
    - Gibbs
    - Yield
    - (Optional) Single-retrosynthesis score (多步要怎么计算呢？或者是只在搜索方法中进行启发式搜索，但是不进行最后的计算) 
    - (Optional) Similarity (同上，只在搜索方法中进行启发式搜索，不进行最后计算)
    - SCScore 合成可行性
5. Analysis Methods:
    - graphviz for search tree
    - graphviz for sucess tree
    - Pareto Front

## Expansion

### As known datasets：KEGG
    input: product_smiles
    output: candidate_reactants


## Evaluation

对于非确知反应，用以下三种方法进行评估：
1. path_length (minimize)

2. overall_retro_score (maximum)：将路径中每一步的retro score（逆合成得分）合并为一个整体得分。合并策略可以是：
    - ✅平均分：取所有步骤retro score的平均值。
    - 几何平均：考虑每一步的贡献，避免极端值的影响。
    - 最低分：路径中最弱的一步决定了整体可行性（木桶原理）

3. overall_feasibility_delta_sascore (maximum): 计算单步的 delta sascore ，合并策略为：
    1. ✅**最小值（min）**：整体可行性由最弱步骤决定，适合捕捉瓶颈：
    $$
    \text{overall feasibility} = \min(\text{feasibility}_i \text{ for step } i)
    $$
    2. **乘积（product）**：如果 feasibility 是概率-like 分数，乘积反映累积可行性：
    $$
    \text{overall feasibility} = \prod \text{feasibility}_i
    $$
    3. **平均值（mean）**：平滑评估，适合非概率分数：
    $$
    \text{overall feasibility} = \text{avg}(\text{feasibility}_i)
    $$


