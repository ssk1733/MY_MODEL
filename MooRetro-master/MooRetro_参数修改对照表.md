# MooRetro 参数修改对照表

这份文档用于快速说明：

- 想改数据集时改哪里
- 想改实验规模时改哪里
- 想改搜索强度时改哪里
- 想改单目标输入时改哪里
- 想改输出目录和结果命名时改哪里

当前版本的 MooRetro 已经支持通过 `test.py` 的命令行参数进行正式数据集实验，因此：

> **批量实验优先改命令行参数，不再优先通过 `config/config.py` 手工改。**

---

## 1. 总体原则

### 1.1 什么时候改命令行参数

如果你做的是以下任意一种情况，优先改 `python test.py ...` 后面的命令行参数：

- 跑正式数据集实验
- 改 `n1` / `n5`
- 改前多少个目标
- 改从第几个目标开始
- 改迭代次数
- 改模拟深度
- 改输出目录
- 改实验名字

### 1.2 什么时候才改 `config/config.py`

`config/config.py` 现在主要用于：

- 作为默认参数模板
- 保存 MCTS 的基础默认设置
- 供 `test.py` 在批量模式中构造基础配置

因此：

- **数据集实验**：通常不需要直接改 `config`
- **单目标实验**：可以用命令行改，也可以继续手动改 `config`
- **想永久调整默认值**：才适合改 `config`

---

## 2. 想改“输入数据集”时改哪里

如果你想换数据集，主要改下面这三个参数：

- `--targets-file`
- `--stock-file`
- `--references-file`

### 作用说明

#### `--targets-file`
目标分子列表文件。

例如：
- `n1-targets.txt`
- `n5-targets.txt`

#### `--stock-file`
商业可得分子列表文件。

例如：
- `n1-stock.txt`
- `n5-stock.txt`

#### `--references-file`
参考路线文件，用于与 PaRoutes / Table 1 风格对照时记录参考来源。

例如：
- `n1-routes.json`
- `n5-routes.json`

### 常见示例

#### 使用 `n1`

```bash
--targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
--stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
--references-file "/root/PaRoutes/data/resource/n1-routes.json"
```

#### 使用 `n5`

```bash
--targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
--stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
--references-file "/root/PaRoutes/data/resource/n5-routes.json"
```

---

## 3. 想改“实验规模”时改哪里

如果你想控制跑多少个目标，或者从哪个目标开始跑，改下面两个参数：

- `--limit`
- `--offset`

### 作用说明

#### `--limit`
表示本次最多取多少个目标分子进行实验。

例如：
- `--limit 1`：只跑 1 个
- `--limit 3`：只跑前 3 个
- `--limit 10`：只跑前 10 个

#### `--offset`
表示跳过前多少个目标，从后面的位置开始跑。

例如：
- `--offset 10`：跳过前 10 个，从第 11 个开始

### 常见组合

#### 跑前 3 个

```bash
--limit 3
```

#### 跑前 10 个

```bash
--limit 10
```

#### 从第 11 个开始跑 10 个

```bash
--offset 10 \
--limit 10
```

---

## 4. 想改“搜索强度”时改哪里

如果你想让搜索更强或者更深，主要改：

- `--max-iterations`
- `--max-simulation-depth`

### 作用说明

#### `--max-iterations`
MCTS 最大迭代次数。

一般来说：
- 越大，搜索越充分；
- 但运行时间也会明显增加。

常见设置：
- `--max-iterations 1`：快速测试/小规模正式验证
- `--max-iterations 3`：稍强一点的实验
- `--max-iterations 5` 或 `10`：更充分但更慢

#### `--max-simulation-depth`
模拟最大深度。

一般来说：
- 越大，允许更深的路径搜索；
- 但也可能带来更高时间开销。

常见设置：
- `--max-simulation-depth 10`
- `--max-simulation-depth 15`

### 建议

- 如果只是先跑通：优先 `--max-iterations 1`
- 如果想提升 solved rate：先尝试增大 `--max-iterations`
- 如果怀疑路径太浅：再考虑增大 `--max-simulation-depth`

---

## 5. 想改“单步模型”时改哪里

改：

- `--single-fn-type`

### 作用说明

当前常用的是：

```bash
--single-fn-type biocyc
```

如果你后续还支持别的单步模型类型，也是在这里切换。

---

## 6. 想改“实验名字/结果目录名”时改哪里

改：

- `--dataset-name`

### 作用说明

这个参数会直接影响结果目录名。

例如：

```bash
--dataset-name n1_smallscale
```

对应输出目录就是：

```bash
/root/autodl-tmp/MooRetro-master/results/paroutes_runs/n1_smallscale/
```

### 建议

为了避免不同实验结果混在一起，建议每次实验都换一个新的名字，例如：

- `n1_smallscale`
- `n5_smallscale`
- `n1_top10`
- `n1_iter3`
- `n5_top10_iter5`

---

## 7. 想改“结果输出位置”时改哪里

改：

- `--output-root`

### 作用说明

这个参数决定所有实验结果放在哪个根目录下。

例如：

```bash
--output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs"
```

如果你改成：

```bash
--output-root "/root/my_results"
```

那么结果就会输出到类似：

```bash
/root/my_results/你的dataset-name/
```

---

## 8. 想关闭绘图、减少依赖和耗时时改哪里

改：

- `--disable-graphs`

### 作用说明

加上这个参数后：

- 不启用图形相关输出；
- 能减少环境依赖问题；
- 批量实验更稳；
- 正式数据集实验通常建议保留该参数。

推荐用法：

```bash
--disable-graphs
```

---

## 9. 想改单目标输入时改哪里

如果你不是跑数据集，而是跑单目标，则改：

- `--mode single`
- `--target`

### 示例

```bash
python test.py \
  --mode single \
  --target "目标分子SMILES" \
  --max-iterations 10 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc
```

### 说明

- `--target` 直接指定单个目标分子；
- 此时不需要 `--targets-file`。

---

## 10. 批量实验与单目标实验分别改什么

### 批量实验（推荐方式）

主要改命令行参数：

- 数据集：`--targets-file` / `--stock-file` / `--references-file`
- 规模：`--limit` / `--offset`
- 搜索强度：`--max-iterations` / `--max-simulation-depth`
- 命名：`--dataset-name`
- 输出：`--output-root`

### 单目标实验

可以两种方式：

#### 方式一：推荐，直接用命令行

- `--mode single`
- `--target`
- `--max-iterations`
- `--max-simulation-depth`

#### 方式二：手动改 `config/config.py`

这种方式现在不是主推荐方式，只适合：

- 想固定默认 target；
- 想长期改默认参数；
- 做代码层面的调试。

---

## 11. 现在最常见的参数修改需求对照表

| 需求 | 改哪个参数 | 备注 |
|---|---|---|
| 换成 `n1` 数据 | `--targets-file` `--stock-file` `--references-file` | 指向 `n1-*` 文件 |
| 换成 `n5` 数据 | `--targets-file` `--stock-file` `--references-file` | 指向 `n5-*` 文件 |
| 只跑前 3 个 | `--limit 3` | 小规模测试 |
| 只跑前 10 个 | `--limit 10` | 小批量实验 |
| 从第 11 个开始 | `--offset 10` | 常与 `--limit` 配合 |
| 增加搜索轮数 | `--max-iterations` | 例如 3、5、10 |
| 增加搜索深度 | `--max-simulation-depth` | 例如 15 |
| 改实验名 | `--dataset-name` | 影响输出目录名 |
| 改输出目录 | `--output-root` | 改结果根目录 |
| 关闭绘图 | `--disable-graphs` | 正式数据建议开启 |
| 跑单目标 | `--mode single` + `--target` | 不用 `targets-file` |
| 改单步模型 | `--single-fn-type` | 当前常用 `biocyc` |

---

## 12. 最简判断规则

如果你只想记住一句话，可以记成：

> **批量实验改命令行参数，`config` 主要保留给默认值和代码级调试。**

也就是说：

- 改数据集：改文件路径参数
- 改规模：改 `limit/offset`
- 改搜索强度：改 `max-iterations/max-simulation-depth`
- 改结果目录：改 `dataset-name/output-root`
- 跑单目标：改 `mode single + target`

---

## 13. 推荐使用顺序

### 场景 1：先验证能不能跑

优先改：

- `--limit 1`
- `--max-iterations 1`
- `--disable-graphs`

### 场景 2：做小规模正式实验

优先改：

- `--limit 3` 或 `10`
- `--max-iterations 1`
- `--dataset-name`

### 场景 3：做更强一点的实验

优先改：

- `--limit`
- `--max-iterations 3/5`
- `--max-simulation-depth 10/15`

### 场景 4：继续分批跑数据

优先改：

- `--offset`
- `--limit`
- `--dataset-name`

---

## 14. 补充说明

虽然现在大部分实验参数已经从 `config` 转移到了命令行，但底层仍然是通过 `MCTSConfig` 来承载配置。因此如果后续你希望：

- 永久修改默认迭代次数
- 永久修改默认搜索深度
- 永久修改默认单步模型类型
- 增加新的全局配置项

那么仍然可以回到：

- `/root/autodl-tmp/MooRetro-master/config/config.py`

进行修改。

但对于日常实验来说，优先用命令行参数即可。
