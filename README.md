# R2WBS

**Randomized Two-Way Block Splitting for Hybrid Federated Convex Optimization**

This repository contains the code for reproducing the numerical experiments. R2WBS lets individual matrix-block clients update using cached row and column aggregates.

## Requirements

- Python 3.10 or later (the archived runs used Python 3.12.14)
- Dependencies listed in `requirements.txt`

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Run the complete experiment grid and generate its summaries and figures:

```bash
python experiments/run_experiments.py --output reproduced/results
python experiments/summarize_results.py --results reproduced/results --output reproduced/summary
python scripts/make_figures.py --results reproduced/results --output reproduced/figures
python scripts/make_algorithm_schematic.py --output reproduced/figures
```

The experiment runner generates the problem instances and writes run records, trajectories, and arrays under `reproduced/results/`. The plotting scripts read those outputs. The final timing fields measure single-threaded sequential Python execution; they are not federated wall-clock speedups. Communication is counted as real-valued unicast payload under the stated logical protocol, not measured network traffic.

## Repository contents

- `experiments/run_experiments.py`: problem generation, R2WBS updates, certificates, and experiment runner
- `experiments/summarize_results.py`: summaries and the generated results table
- `scripts/make_figures.py`: numerical figures from run records
- `scripts/make_algorithm_schematic.py`: algorithm schematic
- `scripts/verify_records.py`: checks the archived records and can compare a new run with them

This is a code-only repository: archived data, run records, and rendered figures are not included. `scripts/verify_records.py` therefore requires the separate experimental archive, including its `results/` directory. The diabetes example is loaded through scikit-learn.

## 中文说明

本仓库仅包含代码和依赖清单，不含归档数据、实验记录或生成的图件。运行 `run_experiments.py` 可重新生成实验数据与结果；`verify_records.py` 则需要另行提供完整实验归档中的 `results/` 目录。
