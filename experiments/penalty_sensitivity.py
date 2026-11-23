"""Vary the DR penalty on one archived correlated-Lasso instance and mask stream."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_experiments import Problem, run_method, write_csv


def solver_model(original, rho):
    # CacheState uses the threshold in model.tau. The proximal objective is
    # F/rho, whereas the primal/dual gap still concerns the unscaled F.
    model = copy.copy(original)
    model.rho = rho
    model.tau = original.tau / rho
    model.certificate = original.certificate

    def squared_loss_prox(q, i):
        return (rho * q + original.b[original.rows[i]]) / (1.0 + rho)

    model.prox_loss = squared_loss_prox
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reproduced" / "penalty_sensitivity.csv")
    args = parser.parse_args()
    results = args.results
    meta = next(item for item in json.loads((results / "problems.json").read_text())
                if item["case"] == "correlated_lasso" and item["repeat"] == 0)
    with (results / "runs.csv").open(newline="") as handle:
        archive = next(row for row in csv.DictReader(handle)
                       if row["case"] == meta["case"] and row["repeat"] == "0"
                       and float(row["participation"]) == 0.5 and row["mask_type"] == "bernoulli")

    rows = []
    with np.load(results / meta["data_file"]) as data, threadpool_limits(limits=1):
        problem = Problem(data["A"], data["b"], meta)
        for rho in (0.5, 1.0, 2.0):
            result, _, _ = run_method(solver_model(problem, rho), 0.5,
                                      int(archive["activation_seed"]), 40000, 25, 1e-6, 1e-5)
            if not result["converged"]:
                raise RuntimeError(f"Cap reached for rho={rho}")
            if rho == 1.0:
                for field in ("rounds", "total_graph_evaluations", "total_scalar_payload"):
                    assert result[field] == int(archive[field]), field
                for field in ("relative_gap", "normalized_residual", "primal"):
                    np.testing.assert_allclose(result[field], float(archive[field]), rtol=1e-6, atol=1e-12)
            rows.append(dict(case=meta["case"], repeat=0, participation=0.5,
                             activation_seed=int(archive["activation_seed"]), rho=rho, theta=1.0,
                             rounds=result["rounds"], total_graph_evaluations=result["total_graph_evaluations"],
                             total_scalar_payload=result["total_scalar_payload"], primal=result["primal"],
                             relative_gap=result["relative_gap"], normalized_residual=result["normalized_residual"],
                             lifted_certificate=np.sqrt(rho) * result["fixed_point_residual"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
