### Delta SAscore 简介

**SAscore**（Synthetic Accessibility Score，合成可得性分数）是一种用于评估分子合成难度的指标，由 Ertl 和 Schuffenhauer 开发，集成在 RDKit 库中。它通过分析分子的结构特征（如环结构、杂原子、立体化学复杂性等）来估计合成难度，分数范围通常在 **1 到 10**，其中：

- **较低分数**（接近 1）：分子简单，易于合成（例如，简单的烷烃）。
- **较高分数**（接近 10）：分子复杂，合成难度大（例如，复杂的天然产物）。

在逆合成（retrosynthesis）路径评估中，**Delta SAscore** 是通过比较反应产物（product）和反应物（reactants）的 SAscore 来评估单步逆合成反应的可行性。它反映了从反应物到产物的合成复杂性变化，是评估逆合成步骤是否合理的一个关键指标。

---

### Delta SAscore 的定义

**Delta SAscore** 定义为：

$$
\Delta \text{SAscore} = \text{SAscore}_{\text{product}} - \text{avg}(\text{SAscore}_{\text{reactants}})
$$

- **$\text{SAscore}_{\text{product}}$**：产物分子的 SAscore。
- **$\text{avg}(\text{SAscore}_{\text{reactants}})$**：所有反应物分子 SAscore 的平均值（因为逆合成一步通常涉及多个反应物）。
- **意义**：
  - **正值**：产物比反应物更复杂，表明逆合成步骤从简单分子生成复杂分子，通常是可行的（逆合成的目标是分解复杂分子为更简单的起始物）。
  - **负值**：产物比反应物更简单，可能表明步骤不合理（逆合成不应增加复杂性）。
  - **接近零**：反应物和产物复杂性相近，可能需要进一步评估。

---

### 在逆合成中的应用

在逆合成路径设计中，Delta SAscore 用于评估单步逆合成的可行性，理由如下：

1. **复杂性降低**：逆合成的核心目标是将复杂的目标分子分解为更简单的、可商购的起始分子。Delta SAscore > 0 表明产物分子比反应物更复杂，符合逆合成逻辑（即反应物更容易合成）。
2. **路径整体评估**：对于多步路径，单步的 Delta SAscore 可以聚合（如取最小值、平均值或乘积）来评估整个路径的可行性，捕捉瓶颈步骤（最难合成的步骤）。
3. **与多目标结合**：Delta SAscore 可以作为多目标优化中的一个指标，与路径长度、retro score（逆合成模型置信度）等结合，纳入 Pareto 前沿分析。

---

### 计算 Delta SAscore

以下是计算 Delta SAscore 的典型流程（参考提供的 `calculate_step_feasibility_score` 函数）：

1. **计算产物 SAscore**：
   - 使用 RDKit 的 `sascorer.calculateScore` 函数计算产物分子的 SAscore。
2. **计算反应物 SAscore**：
   - 对每个反应物分子计算 SAscore，然后取平均值（因为一步反应可能涉及多个反应物）。
3. **计算 Delta SAscore**：
   - 差值：$\Delta \text{SAscore} = \text{SAscore}_{\text{product}} - \text{avg}(\text{SAscore}_{\text{reactants}})$。
4. **归一化**（可选）：
   - 为了便于与其他指标比较（如概率-like 分数 [0,1]），通常将 Delta SAscore 归一化。假设 Delta SAscore 的典型范围在 [-5, 5]（基于 SAscore 的分布），可以用线性或 sigmoid 映射到 [0,1]：
    $$
    \text{feasibility} = \max(0, \min(1, \frac{\Delta \text{SAscore}}{\text{max-delta}}))
    $$
     例如，`max_delta = 5` 是一个合理的经验值。

---

### 示例代码（基于提供函数）

```python
from rdkit import Chem
from rdkit.Chem import sascorer
import numpy as np

def calculate_step_feasibility_score(product_smiles: str, reactant_smiles_list: list[str]) -> float:
    """
    Calculate feasibility score for a single retrosynthesis step using Delta SAscore.
    
    Args:
        product_smiles: SMILES string of the product molecule.
        reactant_smiles_list: List of SMILES strings for reactant molecules.
    
    Returns:
        Feasibility score in [0,1]: higher means more feasible (complexity reduction).
    """
    # 解析分子
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        raise ValueError("Invalid product SMILES")
    
    reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list]
    if any(mol is None for mol in reactant_mols) or not reactant_mols:
        raise ValueError("Invalid reactant SMILES")
    
    # 计算 SAscore
    product_sa = sascorer.calculateScore(product_mol)
    reactant_sas = [sascorer.calculateScore(mol) for mol in reactant_mols]
    avg_reactant_sa = np.mean(reactant_sas)
    
    # 计算 Delta SAscore
    delta_sa = product_sa - avg_reactant_sa
    
    # 归一化到 [0,1]
    max_delta = 5.0  # 假设最大有意义的 Delta SAscore
    feasibility = max(0.0, min(1.0, delta_sa / max_delta))
    
    return feasibility
```

---

### 在路径整体评估中的使用

在多步逆合成路径中，单步 Delta SAscore（或归一化的 feasibility score）需要聚合以评估整体路径可行性。常见策略（参考前文）：

1. **最小值（min）**：整体可行性由最弱步骤决定，适合捕捉瓶颈：
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

**推荐**：在逆合成中，**最小值**常用于 SAscore 相关的可行性评估，因为它强调最难的步骤（瓶颈），符合化学合成的实际需求（参考文献如 SCScore 的应用 )。

在 `evaluator.py` 中集成：

```python
def evaluate_path(self, path: SynthesisPath) -> np.ndarray:
    """
    Evaluate a synthesis path with three objectives: length, retro score, feasibility.
    
    Args:
        path: SynthesisPath object containing molecules and reactions.
    
    Returns:
        np.ndarray: [-path_length, overall_retro_score, overall_feasibility]
    """
    path_length = len(path.reactions)
    retro_scores = [r.retro_score for r in path.reactions]  # 假设retro_score已存储
    feasibility_scores = []
    
    # 计算每步的 feasibility score
    for i, reaction in enumerate(path.reactions):
        product = path.molecules[i]  # 反应产物
        reactants = reaction.reactants  # 反应物列表
        feasibility_scores.append(calculate_step_feasibility_score(product, reactants))
    
    # 聚合分数
    overall_retro = np.prod(retro_scores) if retro_scores else 1.0
    overall_feasibility = min(feasibility_scores) if feasibility_scores else 1.0
    
    return np.array([-path_length, overall_retro, overall_feasibility])
```

---

### Delta SAscore 的优缺点

**优点**：
- **化学意义明确**：基于分子结构的复杂性，反映逆合成分解为更简单分子的逻辑。
- **计算高效**：RDKit 的 SAscore 计算快速，适合大规模路径评估。
- **通用性强**：可与其他指标（如 retro score、路径长度）结合用于多目标优化。

**缺点**：
- **局限性**：SAscore 基于启发式规则，可能不完全反映实际合成难度（如实验条件、催化剂可用性）。
- **归一化依赖**：Delta SAscore 的范围因分子而异，需经验性归一化（如 max_delta=5），可能需调参。
- **不考虑反应条件**：仅基于分子结构，未考虑反应可行性（如溶剂、温度）。

---

### 总结

Delta SAscore 是通过比较产物和反应物的 SAscore 来评估逆合成步骤可行性的有效指标，正值表明复杂度降低，符合逆合成目标。在路径评估中，单步 feasibility score 可通过最小值聚合到整体路径，确保捕捉瓶颈步骤。结合路径长度和 retro score，Delta SAscore 可作为多目标优化的一部分，通过 Pareto 前沿分析展示权衡（如短路径 vs 高可行性）。上述代码简洁、模块化，易于集成到现有 `Evaluator` 和 `Analyzer` 类中。


在多目标优化中，**Delta SAscore 应该最大化**。以下是详细解释：

### 为什么最大化 Delta SAscore？

**Delta SAscore** 定义为：

$$
\Delta \text{SAscore} = \text{SAscore}_{\text{product}} - \text{avg}(\text{SAscore}_{\text{reactants}})
$$

- **$\text{SAscore}_{\text{product}}$**：产物分子的合成可得性分数（1到10，值越大表示合成越复杂）。
- **$\text{avg}(\text{SAscore}_{\text{reactants}})$**：反应物分子 SAscore 的平均值。
- **含义**：
  - **正值**：产物比反应物更复杂（SAscore 更高），表明逆合成步骤从较简单的反应物生成较复杂的产物，符合逆合成的目标（分解复杂分子为更简单的起始物）。
  - **负值**：产物比反应物更简单，表明逆合成步骤增加了复杂性，通常不理想。
  - **零或接近零**：反应物和产物复杂性相似，可能需要进一步评估。

在逆合成路径设计中，目标是找到从目标分子到简单（通常是可商购）分子的路径。因此，**更高的 Delta SAscore** 表示单步逆合成显著降低了分子复杂性，这是理想的。最大化 Delta SAscore（或其归一化形式，如 feasibility score）确保优先选择那些将复杂分子分解为更简单分子的步骤。

### 在多目标优化中的处理

在多目标优化（如 Pareto 前沿分析）中，Delta SAscore 通常与其他指标（如路径长度、retro score）一起考虑。所有目标需统一优化方向（通常最大化）。因此：

- **Delta SAscore（或 feasibility score）**：直接最大化，因为正值表示更可行的逆合成步骤。
- **路径长度**：通常最小化，但为了统一方向，可取负值（-length）进行最大化。
- **Retro score**：假设为置信度（[0,1]，越高越好），直接最大化。

示例（在 `evaluator.py` 的 `evaluate_path` 中）：

```python
import numpy as np

def evaluate_path(self, path: SynthesisPath) -> np.ndarray:
    """
    Evaluate a synthesis path with three objectives: length, retro score, feasibility.
    
    Args:
        path: SynthesisPath object containing molecules and reactions.
    
    Returns:
        np.ndarray: [-path_length, overall_retro_score, overall_feasibility]
    """
    path_length = len(path.reactions)
    retro_scores = [r.retro_score for r in path.reactions]  # 假设retro_score在[0,1]
    feasibility_scores = []
    
    # 计算每步的 Delta SAscore（归一化到[0,1]）
    for i, reaction in enumerate(path.reactions):
        product = path.molecules[i]
        reactants = reaction.reactants
        feasibility_scores.append(calculate_step_feasibility_score(product, reactants))
    
    # 聚合分数
    overall_retro = np.prod(retro_scores) if retro_scores else 1.0  # 最大化
    overall_feasibility = min(feasibility_scores) if feasibility_scores else 1.0  # 最大化
    
    return np.array([-path_length, overall_retro, overall_feasibility])  # 最大化
```

其中，`calculate_step_feasibility_score`（参考前文）计算单步 Delta SAscore 并归一化到 [0,1]：

```python
from rdkit import Chem
from rdkit.Chem import sascorer

def calculate_step_feasibility_score(product_smiles: str, reactant_smiles_list: list[str]) -> float:
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        raise ValueError("Invalid product SMILES")
    
    reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list]
    if any(mol is None for mol in reactant_mols) or not reactant_mols:
        raise ValueError("Invalid reactant SMILES")
    
    product_sa = sascorer.calculateScore(product_mol)
    reactant_sas = [sascorer.calculateScore(mol) for mol in reactant_mols]
    avg_reactant_sa = np.mean(reactant_sas)
    
    delta_sa = product_sa - avg_reactant_sa
    feasibility = max(0.0, min(1.0, delta_sa / 5.0))  # 归一化，最大化
    return feasibility
```

### 为什么不是最小化？

最小化 Delta SAscore 会优先选择产物比反应物更简单的步骤（负值或小正值），这与逆合成的目标相悖。逆合成的核心是从复杂分子分解到简单分子，因此需要反应物比产物简单（即 Delta SAscore 越大越好）。

### 在 Pareto 前沿分析中的作用

在 Pareto 前沿分析中，最大化 Delta SAscore（或其归一化的 feasibility score）与其他目标（如最大化 retro score、最小化路径长度）一起，找出权衡解。例如：
- 一条路径可能有高 Delta SAscore（可行性高）但路径较长。
- 另一条路径可能较短但 Delta SAscore 较低（步骤可行性差）。
Pareto 前沿展示这些权衡，帮助选择最佳路径。

### 总结

**Delta SAscore 应该最大化**，因为它表示逆合成步骤从简单反应物生成复杂产物的能力，符合逆合成目标。在多目标优化中，归一化的 Delta SAscore（feasibility score）与 retro score 直接最大化，路径长度取负值最大化，以统一优化方向。上述代码确保模块化、简洁，易于集成到现有框架中。

