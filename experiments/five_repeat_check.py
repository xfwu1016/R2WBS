"""Archive two additional repetitions and pool five matched comparisons per setting.

First run run_experiments.py with --repeats 5. The optional --primary selects a
separate three-repeat grid for repeats 0--2, as used with the journal archive.
"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median

from audit_reuse import activity_counts
from protocol_checks import deployments
from run_experiments import write_csv


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_key(row):
    return row["case"], row["repeat"], row["participation"], row["mask_type"]


def make_summary(grid, primary, output):
    rows = read_json(grid / "runs.json")
    primary_rows = [row for row in read_json(primary / "runs.json") if row["repeat"] < 3]
    extended_rows = [row for row in rows if row["repeat"] in (3, 4)]
    meta = {(p["case"], p["repeat"]): p for p in read_json(grid / "problems.json")}
    original_meta = {(p["case"], p["repeat"]): p for p in read_json(primary / "problems.json")}
    plan = read_json(grid / "manifest.json")
    primary_plan = read_json(primary / "manifest.json")
    for field in ("cases", "probabilities", "rho", "theta",
                  "initial_state", "max_rounds", "check_every", "gap_tolerance",
                  "normalized_residual_tolerance"):
        if plan[field] != primary_plan[field]:
            raise ValueError(f"The primary and extended grids disagree on {field}")
    if (plan["repeats"], plan["check_every"], plan["gap_tolerance"],
            plan["normalized_residual_tolerance"]) != (5, 25, 1e-6, 1e-5):
        raise ValueError("Use the study's five-repeat grid and joint stopping rule.")
    if (len(rows), len(primary_rows), len(extended_rows), len(meta)) != (65, 39, 26, 20):
        raise ValueError("Expected four classes, 20 instances/partitions and 65 runs.")
    replay = {run_key(row): row for row in rows}
    if len(replay) != 65:
        raise ValueError("Repeated configuration identifier in the five-repeat grid.")
    for row in primary_rows:
        other = replay[run_key(row)]
        if any(row[field] != other[field] for field in ("data_seed", "activation_seed")):
            raise ValueError(f"Seed mismatch for {run_key(row)}")

    pooled = primary_rows + extended_rows
    totals = {}
    for row in pooled:
        case, repeat, prob, mask = run_key(row)
        source = original_meta if repeat < 3 else meta
        problem = source[case, repeat]
        rounds, interval = row["rounds"], plan["check_every"]
        counts = activity_counts(row["activation_seed"], prob, mask, rounds,
                                 len(problem["row_sizes"]), len(problem["col_sizes"]))
        if not row["converged"] or counts.sum() != row["update_graph_evaluations"]:
            raise ValueError(f"Incomplete trajectory or inconsistent graph counts for {run_key(row)}")
        graph = row["total_graph_evaluations"]
        totals[run_key(row)] = dict(
            graph=graph,
            graph_reuse=graph - int(counts[interval::interval].sum()),
            payload_remote=row["total_scalar_payload"],
            payload_colocated=deployments(row, problem)["separate_colocated"])

    pairs = []
    groups = defaultdict(list)
    for (case, repeat, prob, mask), costs in sorted(totals.items()):
        if prob == 1:
            continue
        base = totals[case, repeat, 1, "bernoulli"]
        pair = dict(case=case, repeat=repeat, participation=prob, mask_type=mask)
        pair.update({metric + "_ratio": value / base[metric] for metric, value in costs.items()})
        pairs.append(pair)
        groups[case, prob, mask].append(pair)
    if len(groups) != 9 or any(len(members) != 5 for members in groups.values()):
        raise ValueError("Expected five paired runs in each of nine partial-participation settings.")
    ranges = []
    for (case, prob, mask), members in groups.items():
        for metric in ("graph_ratio", "graph_reuse_ratio", "payload_remote_ratio", "payload_colocated_ratio"):
            values = [pair[metric] for pair in members]
            ranges.append(dict(case=case, participation=prob, mask_type=mask,
                               quantity=metric, pairs=len(values), minimum=min(values),
                               median=median(values), maximum=max(values), below_one=sum(v < 1 for v in values)))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {output}")
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    additional_problems = [p for p in meta.values() if p["repeat"] in (3, 4)]
    for p in additional_problems:
        path = grid / p["data_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != p["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path}")
        shutil.copyfile(path, output / p["data_file"])
    for row in extended_rows:
        trajectory = Path(row["trajectory_file"])
        counts = trajectory.with_name(trajectory.stem + "_activation_counts.json")
        for path in (trajectory, counts):
            shutil.copyfile(grid / path, output / path)
    write_json(output / "problems.json", additional_problems)
    environment = read_json(grid / "environment.json")
    write_json(output / "environment.json", {k: environment[k] for k in (
        "python", "numpy", "scipy", "sklearn", "platform", "blas_threads")})
    write_csv(output / "added_runs.csv", extended_rows)
    write_csv(output / "five_repeat_pairs.csv", pairs)
    write_csv(output / "five_repeat_ranges.csv", ranges)
    print(f"{len(extended_rows)} additional runs; {len(pairs)} paired comparisons; {len(groups)} settings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="Output of run_experiments.py --repeats 5.")
    parser.add_argument("--primary", type=Path, help="Primary results for repetitions 0--2 (defaults to --grid).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reproduced" / "five_repeats")
    args = parser.parse_args()
    make_summary(args.grid, args.primary or args.grid, args.output)
