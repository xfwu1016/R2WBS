# R2WBS

Code for reproducing the numerical experiments accompanying **Randomized Two-Way Block Splitting for Hybrid Federated Convex Optimization**.

The repository contains source code and instructions only. It does not contain the manuscript, archived experiment data, run records, or generated figures.

## Requirements

- Python 3.12 recommended. The archived runs used Python 3.12.14.
- Install the packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the main experiment grid

Generate the 39 runs, problem arrays, random seeds, audit trajectories, and run records:

```bash
python experiments/run_experiments.py --output results
python experiments/summarize_results.py --results results --output reproduced/summary
python scripts/make_figures.py --results results --output reproduced/figures
python scripts/make_algorithm_schematic.py --output reproduced/figures
```

The runner creates `results/` locally. This generated directory is ignored by Git and is not part of the repository. The figures use these newly generated results.

## Reproduce the additional analyses in the revised study

Run the main experiment grid above first. The following scripts read its `results/` data and write derived files under `reproduced/`.

### Stopping rules and accuracy diagnostics

```bash
python experiments/audit_stopping_rules.py --results results --output reproduced/diagnostics
```

Recomputes the paired comparison at the joint original-gap and lifted-residual stopping rule, compares it with stopping on the lifted residual alone, and writes the accuracy and stopping tables plus per-run diagnostics.

### Randomized column-block baseline

```bash
python experiments/column_baseline.py --results results --output reproduced/column_baseline.csv
```

Runs a uniformly sampled column-block proximal-gradient method on the generated problem instances and compares its payload with quarter-participation R2WBS at a common original-problem duality-gap threshold.

### Audit interval, message deployment, and nonzero-graph initialization

```bash
python experiments/protocol_checks.py --output reproduced/protocol_checks
```

Replays audit intervals of 10 and 50 rounds using the same generated arrays and activation seeds, compares message counts for the documented deployments and merged audit messages, and checks the initialization-bias formula on a nonzero graph.

### Local contraction diagnostic

```bash
python experiments/local_rate_check.py --results results --output reproduced/local_rate_check.csv
```

Measures the distance-to-residual ratio and one-step contraction on the specified full-participation instance. These are finite-sample diagnostics, not a uniform error-bound estimate.

### Penalty sensitivity

```bash
python experiments/penalty_sensitivity.py --output reproduced/penalty_sensitivity.csv
```

Repeats one fixed problem and activation stream at three values of the Douglas–Rachford penalty.

## Reproduce archived-record checks

The optional record-verification script compares a newly generated run with an archived `results/` directory. It requires both datasets to be available locally. No archived records are included here.

## Interpretation

Communication counts are real-valued unicast payloads under the stated logical deployment, not measured network traffic. Timings measure single-threaded sequential Python execution and are not distributed wall-clock speedups. Graph-projection counts, total communication, and elapsed computation are separate measures.

## Code map

- `experiments/run_experiments.py` generates the instances and executes R2WBS.
- `experiments/summarize_results.py` aggregates run records.
- `experiments/audit_stopping_rules.py` compares stopping criteria.
- `experiments/column_baseline.py` runs the randomized column-block comparator.
- `experiments/protocol_checks.py` studies audit intervals, message counts, and initialization bias.
- `experiments/local_rate_check.py` computes the local contraction diagnostic.
- `experiments/penalty_sensitivity.py` varies the penalty parameter.
- `scripts/` contains plotting, schematic, and archived-record verification utilities.
