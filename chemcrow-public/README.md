[![tests](https://github.com/ur-whitelab/chemcrow-public/actions/workflows/tests.yml/badge.svg)](https://github.com/ur-whitelab/chemcrow-public)
[![PyPI](https://img.shields.io/pypi/v/chemcrow)](https://img.shields.io/pypi/v/chemcrow)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chemcrow)](https://img.shields.io/pypi/pyversions/chemcrow)
[![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)
[![DOI](https://zenodo.org/badge/649361700.svg)](https://zenodo.org/doi/10.5281/zenodo.10884638)




<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chemcrow_dark_bold.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="assets/chemcrow_light_bold.png" width='100%'>
  <img alt="ChemCrow logo" src="/assets/" width="100%">
</picture>


<br></br>


ChemCrow is an open source package for the accurate solution of reasoning-intensive chemical tasks.

Built with Langchain, it uses a collection of chemical tools including RDKit, paper-qa, as well as some relevant databases in chemistry, like Pubchem and chem-space.

## 🤗 Try it out in [HuggingFace](https://huggingface.co/spaces/doncamilom/ChemCrow)!

[![ChemCrow Demo](assets/hf-demo.png)](https://huggingface.co/spaces/doncamilom/ChemCrow)


## ⚠️ Note

This package does not contain all the tools described in the [ChemCrow paper](https://arxiv.org/abs/2304.05376) because
of API usage restrictions. This repo will not give the same results as that paper.

All the experiments have been released under [ChemCrow runs](https://github.com/ur-whitelab/chemcrow-runs).


## 👩‍💻 Installation

```
pip install chemcrow
```

## 🔥 Usage
First set up your API keys in your environment.
```
export OPENAI_API_KEY=your-openai-api-key
```

You can optionally use Serp API:

```
export SERP_API_KEY=your-serpapi-api-key
```

In a Python session:
```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, streaming=False)
chem_model.run("What is the molecular weight of tylenol?")
```


## 🛠️ Self-hosting of some tools.

By default, ChemCrow relies on the RXN4Chem API for retrosynthetic planning and reaction product prediction. This can however be slow and depends on you having an API key.

Optionally, you can also self host these tools by running some pre-made docker images.

Run 

```
docker run --gpus all -d -p 8051:5000 doncamilom/rxnpred:latest
docker run --gpus all -d -p 8052:5000 doncamilom/retrosynthesis:latest
```


Now ChemCrow can be used like this:

```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(model="gpt-4-0613", temp=0.1, streaming=False, local_rxn=True)
chem_model.run("What is the product of the reaction between styrene and dibromine?")
```


### Local MooRetro enzymatic single-step prediction

ChemCrow also registers `MooRetroSingleStepPredict` as an agent tool. By
default it loads the sibling project at `../MooRetro-master`; set
`MOO_RETRO_PATH` if the model is stored elsewhere. Model weights are loaded
only on the first tool call and then reused by MooRetro's model cache.

```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(
    model="gpt-4-0613",
    temp=0.1,
    streaming=False,
)
chem_model.run(
    "Use the local MooRetro model to predict one enzymatic retrosynthesis "
    "step for product [NH3+]CCCC[NH3+]."
)
```

The tool can also be tested without an LLM:

```python
from chemcrow.tools import MooRetroSingleStepPredict

tool = MooRetroSingleStepPredict(single_fn_type="biocyc", ec_level=3)
print(tool.run(
    '{"target_smiles":"[NH3+]CCCC[NH3+]",'
    '"building_blocks":["NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]"]}'
))
```

The output is JSON. Each prediction contains precursor SMILES, EC number,
model score, reaction template, similarity, and feasibility.


### AND-OR Monte Carlo tree search

`EnzymeRetrosynthesisSearch` is the agent-facing complete-route planner. It
represents molecules as OR nodes and reactions as AND nodes. Every non-currency
precursor of a chosen reaction must be solved. The default policy is
`and_or_mcts`, using UCB selection, reaction expansion, bounded
evaluator-guided rollout, scalar reward backpropagation, and progressive
widening. Database and MooRetro candidates are fused before expansion.

```python
import json
from chemcrow.tools import EnzymeRetrosynthesisSearch

tool = EnzymeRetrosynthesisSearch(config={
    "use_database": True,
    "use_model": True,
    "max_depth": 10,
    "max_expansions": 500,
    "mcts_exploration_constant": 1.4,
    "mcts_rollout_depth": 3,
    "mcts_random_seed": 100,
})

result = json.loads(tool.run(json.dumps({
    "target": "CCO",
    "building_blocks": ["OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"],
    "top_k": 5,
    "search_algorithm": "and_or_mcts",
})))
print(json.dumps(result, ensure_ascii=False, indent=2))
```

The previous policies remain available as `best_first_beam` and
`uniform_cost`. MCTS execution can be verified through
`stats.mcts_iterations`, `stats.mcts_selection_steps`,
`stats.mcts_rollouts`, `stats.mcts_backpropagations`, and
`stats.mcts_root_visits`. The exported `and_or_tree` contains the compact
Top-K complete solution subgraph. Every returned route also contains
`enzyme_annotated_retrosynthesis`, formatted as
`target ←[EC number; reaction; provenance] intermediate ... ←[enzyme] building block`.
For reactions with multiple required substrates, use the target-first
`retrosynthesis_reactions[].equation` records; they retain every AND branch.
Missing enzyme annotations are written as `EC unknown` rather than inferred.

The search tool also accepts `targets` for multiple products and a local
`building_block_library` in TXT/SMI, CSV/TSV, JSON/JSONL, or XLSX format.
Use `endpoint_mode=any_stock` for a stock library, or `exact_set` only when all
supplied building blocks must appear as final leaves. Multi-target output contains
per-target `target_results` and evaluated Top-K `route_bundles`. To run this flow
through ChemCrow Agent and save the complete structured result, use
`scripts/run_agent_retrosynthesis.py`.

### RetroBioCat 52-pathway accuracy benchmark

Build the canonical, agent-ready benchmark from the source workbook and run a
resumable Top-K evaluation:

```bash
python scripts/build_retrobio_cat52_benchmark.py
python scripts/evaluate_retrobio_cat52.py \
  --model deepseek-flash \
  --top-k 10 \
  --report-top-k 1,3,10 \
  --max-depth 10 \
  --agent-attempts 2 \
  --overwrite
```

The adapted dataset and its metric definitions are documented in
`../dataset/retrobiocat52/README.md`. The source provides targets, literature
starting materials, enzyme classes, and DOI, but not intermediate structures;
the benchmark therefore reports endpoint, step-count, enzyme-class, and Top-K
route recovery without claiming exact intermediate accuracy.

### Automatic run transcripts

ChemCrow saves every top-level `agent.run(...)` or
`agent.agent_executor(...)` execution to `result/YYYYMMDDHHMM.txt` by
default. Each transcript contains the user input, ReAct action log, tool input,
tool observation, final answer, and execution error if one occurred. Runs that
finish in the same minute are appended to the same file instead of overwriting
one another.

Use a different directory or disable this behavior when constructing the agent:

```python
agent = ChemCrow(
    # ...model and API configuration...
    result_dir="/path/to/results",
    save_transcripts=True,
)

print(agent.transcript_handler.last_path)
```

## ✅ Citation
Bran, Andres M., et al. "ChemCrow: Augmenting large-language models with chemistry tools." arXiv preprint arXiv:2304.05376 (2023).

```bibtex
@article{bran2023chemcrow,
      title={ChemCrow: Augmenting large-language models with chemistry tools},
      author={Andres M Bran and Sam Cox and Oliver Schilter and Carlo Baldassari and Andrew D White and Philippe Schwaller},
      year={2023},
      eprint={2304.05376},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      publisher={arXiv}
}
```
