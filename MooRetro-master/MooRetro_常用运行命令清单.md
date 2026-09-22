# MooRetro 常用运行命令清单

项目目录默认是：

```bash
cd /root/autodl-tmp/MooRetro-master
---

## 一、单目标实验

### 1. 运行一个单目标分子

```bash
python test.py \
  --mode single \
  --target "目标分子SMILES" \
  --max-iterations 10 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc
```

#### 说明
- `--target`：目标分子 SMILES
- `--max-iterations`：MCTS 迭代次数
- `--max-simulation-depth`：模拟深度
- `--single-fn-type`：单步模型类型，当前一般用 `biocyc`

---


单个测试的命令
python test.py \
  --mode dataset \
  --dataset-name Ubiquinone_9\
  --targets-file "/root/autodl-tmp/MooRetro-master/results/Ubiquinone_9_targets.txt" \
  --stock-file "/root/autodl-tmp/MooRetro-master/results/Ubiquinone_9_stock.txt" \
  --references-file null \
  --output-root "/root/autodl-tmp/MooRetro-master/results/Ubiquinone_9" \
  --limit 1 \
  --max-iterations 50 \
  --max-simulation-depth 15 \
  --single-fn-type biocyc 



python test.py --mode dataset --dataset-name Ubiquinone_9 --targets-file "results/Ubiquinone_9_targets.txt" --stock-file "results/Ubiquinone_stock.txt" --references-file null --output-root "results" --limit 1 --max-iterations 50 --max-simulation-depth 15 --single-fn-type biocyc



python my_test.py --mode dataset --dataset-name test --targets-file "results/Ubiquinone_9_targets.txt" --stock-file "results/test.txt" --references-file null --output-root "results" --limit 1 --max-iterations 50 --max-simulation-depth 15 --single-fn-type model_xlsx --model-xlsx "model_output.xlsx" 




python my_test.py --mode dataset --dataset-name HT_test_xlsx --targets-file "Z_ht/target.txt" --stock-file "Z_ht/stock1.txt" --references-file null --output-root "results" --limit 1 --max-iterations 100 --max-simulation-depth 20 --single-fn-type model_xlsx --model-xlsx "Z_ht/kegg_metacyc_extended.xlsx" 

python test.py --mode dataset --dataset-name HT_test --targets-file "Z_ht/target.txt" --stock-file "Z_ht/stock1.txt" --references-file null --output-root "results" --limit 1 --max-iterations 50 --max-simulation-depth 15 --single-fn-type biocyc


python my_test.py \
  --mode dataset \
  --dataset-name HT_test_kegg \
  --targets-file "Z_ht/target.txt" \
  --stock-file "Z_ht/stock1.txt" \
  --output-root "results" \
  --limit 1 \
  --max-iterations 100 \
  --max-simulation-depth 20 \
  --single-fn-type model_xlsx \
  --model-xlsx "Z_ht/kegg_model_output.xlsx"





多个测试命令
python test.py \
  --mode dataset \
  --dataset-name n5_smallscale-no\
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 2 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc 














### 2. 单目标快速测试版

适合先看能不能跑通：

```bash
python test.py \
  --mode single \
  --target "目标分子SMILES" \
  --max-iterations 1 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

## 二、正式数据集批量实验

### 3. 跑 `n1` 前 3 个目标

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_smallscale\
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 2 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

### 4. 跑 `n5` 前 3 个目标

```bash
python test.py \
  --mode dataset \
  --dataset-name n5_smallscale \
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 1 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

### 5. 跑 `n1` 前 10 个目标

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_top100-deep \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 100 \
  --max-iterations 5 \
  --max-simulation-depth 5 \
  --single-fn-type biocyc \
  --disable-graphs
```
```bash



---
---

### 6. 跑 `n5` 前 10 个目标

```bash
python test.py \
  --mode dataset \
  --dataset-name n5_top100\
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 100 \
  --max-iterations 2 \
  --max-simulation-depth 5 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

### 7. 从某个位置开始跑，例如 `n1` 从第 11 个开始跑 40 个

```bash
python test.py \
  --mode dataset \
  --dataset-name n5_500 \
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --offset 0 \
  --limit 100 \
  --max-iterations 10 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

#### 说明
- `--offset 10` 表示跳过前 10 个，从第 11 个开始
- `--limit 10` 表示再取 10 个

---

## 三、提高搜索强度的实验

### 8. 跑 `n1` 前 3 个目标，但把迭代次数提高到 3

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_iter3 \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 3 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

### 9. 跑 `n5` 前 3 个目标，迭代次数提高到 5

```bash
python test.py \
  --mode dataset \
  --dataset-name n5_iter5 \
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 5 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

### 10. 提高模拟深度

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_depth15 \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 3 \
  --max-simulation-depth 15 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

## 四、结果目录控制

### 11. 改输出目录

如果你不想都放在 `results/paroutes_runs` 下面，可以改：

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_custom_output \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/my_experiment_results" \
  --limit 3 \
  --max-iterations 1 \
  --disable-graphs
```

这样结果会放在：

```bash
/root/my_experiment_results/n1_custom_output/
```

---

## 五、最常用参数改哪里

你以后最常改的通常就是下面这些：

### 改数据集
- `--targets-file`
- `--stock-file`
- `--references-file`

### 改实验名字
- `--dataset-name`

### 改规模
- `--limit`
- `--offset`

### 改搜索强度
- `--max-iterations`
- `--max-simulation-depth`

### 改输出位置
- `--output-root`

### 改单步模型
- `--single-fn-type`

### 关闭图生成
- `--disable-graphs`

---

## 六、推荐的常用组合

### 1. 快速检查正式数据能不能跑通

```bash
python test.py \
  --mode dataset \
  --dataset-name quick_check_n1 \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 1 \
  --max-iterations 1 \
  --disable-graphs
```

### 2. 做一版正式小规模实验

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_smallscaletopktest \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 3 \
  --max-iterations 10 \
  --disable-graphs
```

### 3. 做稍微正式一点的小批量实验

```bash
python test.py \
  --mode dataset \
  --dataset-name n1_top10_iter3 \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 10 \
  --max-iterations 3 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

## 七、最简记忆版模板

你只要记住这个模板就行：

```bash
python test.py \
  --mode dataset \
  --dataset-name 实验名 \
  --targets-file "targets文件" \
  --stock-file "stock文件" \
  --references-file "routes文件" \
  --output-root "输出根目录" \
  --limit 取多少个 \
  --offset 从第几个开始 \
  --max-iterations 迭代次数 \
  --max-simulation-depth 模拟深度 \
  --single-fn-type biocyc \
  --disable-graphs
```

---

## 八、使用建议

- 如果你的目标是先验证能不能跑通，优先用：`--limit 1` 或 `--limit 3`
- 如果你的目标是先拿一版正式结果，优先用：`--limit 3` 或 `--limit 10`
- 如果你的目标是提升 solved rate，可以尝试增加：`--max-iterations`
- 如果你担心运行太慢，建议保留：`--disable-graphs`
- 如果你要继续做正式记录，建议每次都改一个新的 `--dataset-name`，避免结果目录被混淆
python test.py \
  --mode dataset \
  --dataset-name n5_top10topk \
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --limit 10 \
  --max-iterations 5 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs



  ```bash
python test.py \
  --mode dataset \
  --dataset-name n5_500 \
  --targets-file "/root/PaRoutes/data/resource/n5-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n5-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n5-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --offset 0 \
  --limit 500 \
  --max-iterations 10 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs


  ```bash
python test.py \
  --mode dataset \
  --dataset-name n1_500 \
  --targets-file "/root/PaRoutes/data/resource/n1-targets.txt" \
  --stock-file "/root/PaRoutes/data/resource/n1-stock.txt" \
  --references-file "/root/PaRoutes/data/resource/n1-routes.json" \
  --output-root "/root/autodl-tmp/MooRetro-master/results/paroutes_runs" \
  --offset 0 \
  --limit 500 \
  --max-iterations 10 \
  --max-simulation-depth 10 \
  --single-fn-type biocyc \
  --disable-graphs