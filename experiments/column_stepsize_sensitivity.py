"""Compare two column step sizes under matched original-gap stopping and routing."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from threadpoolctl import threadpool_limits

from column_baseline import compare, gram_setup_cost, run_column_blocks
from run_experiments import Problem, write_csv


def step_comparison(results):
    manifest = json.loads((results / "manifest.json").read_text())
    if manifest["check_every"] != 25 or manifest["gap_tolerance"] != 1e-6:
        raise ValueError("Use the study grid with 25-round audits and gap tolerance 1e-6.")
    # compare() evaluates the norm-based column solver and the R2WBS gap-only
    # stopping time on the same primary arrays, partitions and audit records.
    prior = {(row["case"], row["repeat"]): row for row in compare(results)
             if float(row["p"]) == 0.25 and row["mask"] == "bernoulli"}
    records = []
    for meta in json.loads((results / "problems.json").read_text()):
        key = meta["case"], meta["repeat"]
        if key not in prior:
            raise ValueError(f"Missing quarter-participation Bernoulli run for {key}")
        row = prior[key]
        input_path = results / meta["data_file"]
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Data file disagrees with the recorded hash: {input_path}")
        with np.load(input_path) as arrays:
            p = Problem(arrays["A"], arrays["b"], meta)
        gap_cost = int(p.graph_payload.sum() + p.row_payload + 3*p.M)
        rounds, payload, gap = run_column_blocks(p, row["seed"], gap_cost, share_gram=True)
        rank_deficiencies = [min(block.shape) - np.linalg.matrix_rank(block)
                             for blocks in p.block for block in blocks]
        base = row["r2wbs_payload"]
        records.append(dict(case=key[0], repeat=key[1], activation_seed=row["seed"],
                            r2wbs_quarter_payload=base, published_rounds=row["column_rounds"],
                            gram_rounds=rounds, published_payload=row["column_payload"],
                            gram_payload=payload, gram_setup_scalars=gram_setup_cost(p),
                            published_ratio=row["column_payload"]/base, gram_ratio=payload/base,
                            published_relative_gap=row["column_gap"], gram_relative_gap=gap,
                            rank_deficient_local_blocks=sum(v > 0 for v in rank_deficiencies),
                            local_blocks=len(rank_deficiencies)))
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "reproduced/column_stepsize.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        records = step_comparison(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, records)
    for case in dict.fromkeys(r["case"] for r in records):
        rows = [r for r in records if r["case"] == case]
        print(case, "local-norm and Gram medians:",
              median(r["published_ratio"] for r in rows), median(r["gram_ratio"] for r in rows))
