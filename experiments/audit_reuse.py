"""Reprice the same R2WBS trajectories with reuse of frozen audit projections.

Supply the main results directory and the audit_spacing.csv produced by
protocol_checks.py for the same instances and activation streams.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def activity_counts(seed, probability, mask, rounds, rows, columns):
    if mask == "rectangle_2x2":
        # Which clients are selected is random; the graph count is always four.
        if (rows, columns, probability) != (4, 4, 0.25):
            raise ValueError("The rectangle schedule requires a 4-by-4 grid with marginal 1/4.")
        return np.full(rounds, 4, dtype=int)
    if mask != "bernoulli":
        raise ValueError(f"Unsupported mask {mask!r}")
    if probability == 1:
        return np.full(rounds, rows * columns, dtype=int)
    rng = np.random.default_rng(seed)
    return np.count_nonzero(rng.random((rounds, rows, columns)) < probability, axis=(1, 2))


def reprice(results, spacing_file):
    runs = read_rows(results / "runs.csv")
    seeds = {(r["case"], int(r["repeat"]), float(r["participation"]), r["mask_type"]):
             int(r["activation_seed"]) for r in runs}
    instances = {(m["case"], m["repeat"]): m for m in
                 json.loads((results / "problems.json").read_text(encoding="utf-8"))}
    rows = []
    for observation in read_rows(spacing_file):
        case, repeat = observation["case"], int(observation["repeat"])
        prob, mask = float(observation["probability"]), observation["mask"]
        interval, rounds = int(observation["spacing"]), int(observation["rounds"])
        meta = instances[case, repeat]
        M, N = len(meta["row_sizes"]), len(meta["col_sizes"])
        counts = activity_counts(seeds[case, repeat, prob, mask], prob, mask, rounds, M, N)
        n_audits = (rounds + interval - 1) // interval
        recorded = int(observation["graph_calls"])
        calculated = int(np.sum(counts)) + n_audits * M * N
        if calculated != recorded:
            raise ValueError(f"Original graph counts disagree: {case}, {repeat}, {interval}, {prob}, {mask}")
        # Array offset interval is round interval + 1, the first round
        # whose input state was frozen at a previous audit. A final audit
        # has no successor in the prefix.
        saved = int(np.sum(counts[interval::interval]))
        m, n = sum(meta["row_sizes"]), sum(meta["col_sizes"])
        # Remote graph pairs and row roles return one squared norm per
        # role. Column regularizers are colocated and carry zero network cost.
        saved_scalars = n_audits * (N*m + M*n + m - (M*N + M))
        payload = int(observation["payload"])
        rows.append(dict(case=case, repeat=repeat, probability=prob, mask=mask,
                         interval=interval, rounds=rounds, graph_recorded=recorded,
                         graph_saved=saved, graph_reusing=recorded-saved,
                         payload_recorded=payload, payload_saved_norm=saved_scalars,
                         payload_with_norm_replies=payload-saved_scalars))
    full = {(r["case"], r["repeat"], r["interval"]): r for r in rows if r["probability"] == 1}
    pairs = []
    for r in rows:
        if r["probability"] < 1:
            base = full[r["case"], r["repeat"], r["interval"]]
            pair = {k: r[k] for k in ("case", "repeat", "probability", "mask", "interval")}
            for metric in ("graph_recorded", "graph_reusing", "payload_recorded", "payload_with_norm_replies"):
                pair[metric + "_ratio"] = r[metric] / base[metric]
            pairs.append(pair)
    return rows, pairs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--spacing", type=Path,
                        default=root / "reproducibility/protocol_checks/audit_spacing.csv")
    parser.add_argument("--output", type=Path, default=root / "reproduced/audit_reuse")
    args = parser.parse_args()
    observations, pairs = reprice(args.results, args.spacing)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "audit_reuse_runs.csv", observations)
    write_rows(args.output / "audit_reuse_pairs.csv", pairs)
    for interval in sorted({r["interval"] for r in pairs}):
        subset = [r for r in pairs if r["interval"] == interval]
        count = sum(r["graph_reusing_ratio"] < 1 for r in subset)
        print(f"L={interval}, fewer graph calls in {count}/{len(subset)} pairs after reuse")
