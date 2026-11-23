#!/usr/bin/env python3
"""Verify the complete archived study without rerunning an optimizer.

Checks hashes, every planned record, arrays, seeds, activation streams,
accounting, trajectories, summary/table values, and paired figure inputs.
An optional --compare directory checks a new subset against the archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_experiments import make_case  # Only problem generation; no optimizer.

CASES = ["correlated_lasso", "rank_deficient_lasso", "uneven_huber", "diabetes_quadratic"]
LABELS = dict(zip(CASES, ["Correlated Lasso", "Rank-deficient Lasso", "Uneven Huber", "Diabetes quadratic"]))
SIZES = {
    "correlated_lasso": ([60] * 4, [30] * 4),
    "rank_deficient_lasso": ([45] * 4, [60] * 4),
    "uneven_huber": ([30, 60, 90, 180], [15, 30, 45, 90]),
    "diabetes_quadratic": ([110, 110, 111, 111], [16, 16, 16, 17]),
}
TIMING = {"sequential_seconds", "setup_seconds", "sequential_plus_setup_seconds"}
CHECKS = 0


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row["case"], int(row["repeat"]), float(row["participation"]), row["mask_type"]


def same(actual, expected, label, rtol=1e-12, atol=1e-13):
    if isinstance(expected, bool):
        require(actual == expected or actual == str(expected), label)
    elif isinstance(expected, int):
        require(float(actual) == expected, label)
    elif isinstance(expected, float):
        require(math.isclose(float(actual), expected, rel_tol=rtol, abs_tol=atol), label)
    else:
        require(actual == expected, label)


def verify_csv_json(csv_rows, json_rows, label):
    require(len(csv_rows) == len(json_rows), label + ": row count")
    for index, (actual, expected) in enumerate(zip(csv_rows, json_rows)):
        require(set(actual) == set(expected), f"{label}[{index}]: columns")
        for name, value in expected.items():
            same(actual[name], value, f"{label}[{index}].{name}")


def strategy(row):
    if row["mask_type"] == "rectangle_2x2":
        return "rectangle"
    return {1.0: "full", 0.5: "half", 0.25: "quarter"}[float(row["participation"])]


def replay_masks(row, rows, cols):
    rounds = row["rounds"]
    rng = np.random.default_rng(row["activation_seed"])
    shape = (rounds, len(rows), len(cols))
    if row["mask_type"] == "rectangle_2x2":
        masks = np.zeros(shape, dtype=bool)
        for k in range(rounds):
            masks[k][np.ix_(rng.choice(4, 2, replace=False), rng.choice(4, 2, replace=False))] = True
    elif row["participation"] == 1:
        masks = np.ones(shape, dtype=bool)
    else:
        masks = rng.random(shape) < row["participation"]
    graph_payload = 2 * (np.asarray(rows)[:, None] + np.asarray(cols)[None, :])
    cumulative_graph = np.cumsum(masks.sum(axis=(1, 2)))
    cumulative_payload = np.cumsum((masks * graph_payload).sum(axis=(1, 2)) + 2 * sum(rows))
    return cumulative_graph, cumulative_payload, masks.sum(axis=0)


def verify_archive():
    folder = ROOT / "results"
    figures = ROOT / "reference_figures"
    manifest = read_json(folder / "manifest.json")
    hashes = read_json(ROOT / "reproducibility" / "reference_sha256.json")
    result_paths = {p.relative_to(ROOT).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(result_paths == {p for p in hashes if p.startswith("results/")}, "Reference hash coverage")
    for name, expected in hashes.items():
        require(digest(ROOT / name) == expected, f"Reference hash changed: {name}")

    for name, expected in dict(cases=CASES, repeats=3, probabilities=[1.0, 0.5, 0.25],
            rho=1.0, theta=1.0, initial_state="zero", max_rounds=40000, check_every=25,
            gap_tolerance=1e-6, normalized_residual_tolerance=1e-5, planned_run_count=39).items():
        same(manifest[name], expected, f"Protocol {name}")
    runs = read_json(folder / "runs.json")
    verify_csv_json(read_csv(folder / "runs.csv"), runs, "runs.csv / runs.json")
    expected_grid = {(case, repeat, p, "bernoulli") for case in CASES for repeat in range(3) for p in [1.0, 0.5, 0.25]}
    expected_grid |= {("correlated_lasso", repeat, 0.25, "rectangle_2x2") for repeat in range(3)}
    require(len(runs) == 39 and {key(r) for r in runs} == expected_grid, "Complete, unique 39-run grid")
    seed_columns = ["case", "repeat", "participation", "mask_type", "data_seed", "activation_seed"]
    verify_csv_json(read_csv(ROOT / "reproducibility" / "seed_schedule.csv"),
                    [{name: r[name] for name in seed_columns} for r in runs], "Exact seed schedule")

    problems = read_json(folder / "problems.json")
    require(len(problems) == 12, "Twelve problem metadata records")
    problem_map = {(p["case"], p["repeat"]): p for p in problems}
    require(set(problem_map) == {(case, repeat) for case in CASES for repeat in range(3)}, "Complete problem grid")
    require(len(list((folder / "data").glob("*.npz"))) == 12, "Twelve problem arrays")
    for meta in problems:
        case, repeat = meta["case"], meta["repeat"]
        tag = f"{case}/r{repeat}"
        require(meta["data_seed"] == 20261122 + 100 * repeat, tag + ": data seed")
        rows, cols = SIZES[case]
        require(meta["row_sizes"] == rows and meta["col_sizes"] == cols, tag + ": partition sizes")
        require(meta["rho"] == meta["theta"] == 1.0, tag + ": parameters")
        require(digest(folder / meta["data_file"]) == meta["sha256"], tag + ": data checksum")
        with np.load(folder / meta["data_file"], allow_pickle=False) as array:
            require(set(array.files) == {"A", "b", "reference_x", "row_sizes", "col_sizes"}, tag + ": NPZ fields")
            A, b, x = array["A"], array["b"], array["reference_x"]
            require(A.shape == (sum(rows), sum(cols)) and b.shape == (sum(rows),), tag + ": dimensions")
            require(x.shape == (sum(cols),), tag + ": reference dimensions")
            require(all(np.isfinite(array[n]).all() for n in array.files), tag + ": finite arrays")
            require(np.array_equal(array["row_sizes"], rows) and np.array_equal(array["col_sizes"], cols), tag + ": stored partition")
            require(np.allclose(A.mean(axis=0), 0, atol=1e-13) and np.allclose(np.linalg.norm(A, axis=0), 1, atol=1e-13), tag + ": normalization")
            require(np.linalg.matrix_rank(A) == meta["numerical_rank"], tag + ": recorded numerical rank")
            generated_A, generated_b, generated_meta = make_case(case, repeat)
            require(np.allclose(A, generated_A, rtol=1e-12, atol=1e-13), tag + ": regenerated design")
            require(np.allclose(b, generated_b, rtol=1e-12, atol=1e-13), tag + ": regenerated response")
            for name in ["m", "n", "loss", "delta", "construction_rank"]:
                same(meta[name], generated_meta[name], tag + ": generated " + name)
            grad0 = -b if meta["loss"] == "squared" else np.clip(-b, -meta["delta"], meta["delta"])
            same(meta["tau"], 0.08 * float(np.max(np.abs(A.T @ grad0))), tag + ": tau")
            residual = A @ x - b
            if meta["loss"] == "squared":
                gradient = residual
                loss = float(0.5 * residual @ residual)
            else:
                delta = meta["delta"]
                gradient = np.clip(residual, -delta, delta)
                loss = float(np.where(np.abs(residual) <= delta, 0.5 * residual ** 2, delta * np.abs(residual) - 0.5 * delta ** 2).sum())
            max_at = float(np.max(np.abs(A.T @ gradient)))
            alpha = min(1.0, meta["tau"] / max(max_at, 1e-300)) * (1 - 16 * np.finfo(float).eps)
            q = alpha * gradient
            primal = loss + meta["tau"] * float(np.abs(x).sum())
            dual = float(-0.5 * q @ q - b @ q)
            same(primal, meta["reference"]["primal"], tag + ": reference primal")
            same(dual, meta["reference"]["dual"], tag + ": reference dual")
            require((primal - dual) / max(1.0, abs(primal)) <= 1.01e-10, tag + ": stored reference accuracy")

    total_audits = 0
    for r in runs:
        tag = str(key(r))
        meta = problem_map[r["case"], r["repeat"]]
        rows, cols = meta["row_sizes"], meta["col_sizes"]
        M, N, m, n = len(rows), len(cols), sum(rows), sum(cols)
        require(r["activation_seed"] == 73000 + 1000 * CASES.index(r["case"]) + r["repeat"], tag + ": activation seed")
        require(r["data_seed"] == meta["data_seed"], tag + ": run data seed")
        same(r["tau"], meta["tau"], tag + ": tau")
        trace = read_csv(folder / r["trajectory_file"])
        rounds = list(range(25, r["rounds"] + 1, 25))
        if r["rounds"] % 25:
            rounds.append(r["rounds"])
        require([int(t["round"]) for t in trace] == rounds, tag + ": audit schedule")
        require(len(trace) == r["audits"], tag + ": audit count")
        require(r["rounds"] <= 40000, tag + ": iteration cap")
        cumulative_graph, cumulative_payload, counts = replay_masks(r, rows, cols)
        stored_counts = read_json(folder / Path(r["trajectory_file"]).with_name(Path(r["trajectory_file"]).stem + "_activation_counts.json"))
        require(np.array_equal(counts, stored_counts), tag + ": replayed client activations")
        initialization = (N + 1) * m + M * n + M
        audit_payload = 4 * (N * m + M * n) + 4 * m + 3 * M
        previous_seconds = -1.0
        for number, t in enumerate(trace, 1):
            k = int(t["round"])
            expected = dict(update_graph_evaluations=int(cumulative_graph[k - 1]), audit_graph_evaluations=number * M * N,
                update_scalar_payload=int(cumulative_payload[k - 1]), audit_scalar_payload=number * audit_payload,
                initialization_scalar_payload=initialization,
                total_scalar_payload=int(cumulative_payload[k - 1]) + number * audit_payload + initialization)
            for name, value in expected.items():
                same(t[name], value, f"{tag}: audit {number} {name}")
            require(all(math.isfinite(float(value)) for value in t.values()), tag + ": finite trace")
            require(float(t["gap"]) >= -1e-9 and float(t["relative_gap"]) > 0, tag + ": gap domain")
            same(t["gap"], float(t["primal"]) - float(t["dual"]), tag + ": gap identity")
            same(t["relative_gap"], float(t["gap"]) / max(1.0, abs(float(t["primal"]))), tag + ": relative gap")
            require(float(t["dual_constraint_ratio"]) <= 1 + 1e-12 and float(t["dual_box_ratio"]) <= 1 + 1e-12, tag + ": dual feasibility")
            require(float(t["sequential_seconds"]) >= previous_seconds, tag + ": increasing measured time")
            previous_seconds = float(t["sequential_seconds"])
            if number < len(trace):
                require(not (float(t["relative_gap"]) <= 1e-6 and float(t["normalized_residual"]) <= 1e-5), tag + ": earliest passing audit")
        for name in trace[-1]:
            if name not in {"round", "sequential_seconds"}:
                same(r[name], float(trace[-1][name]), tag + ": final " + name)
        require(r["sequential_seconds"] >= float(trace[-1]["sequential_seconds"]), tag + ": final time ordering")
        same(r["total_graph_evaluations"], r["update_graph_evaluations"] + r["audit_graph_evaluations"], tag + ": total graph calls")
        same(r["loss_prox_evaluations"], (r["rounds"] + r["audits"]) * M, tag + ": loss calls")
        same(r["regularizer_prox_evaluations"], (r["rounds"] + r["audits"]) * N, tag + ": regularizer calls")
        same(r["objective_minus_reference"], r["primal"] - meta["reference"]["primal"], tag + ": reference objective difference")
        require(r["converged"] == (r["relative_gap"] <= 1e-6 and r["normalized_residual"] <= 1e-5), tag + ": convergence flag")
        total_audits += len(trace)

    log = [json.loads(line) for line in (folder / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    log_runs = [entry for entry in log if "case" in entry]
    require(len(log_runs) == 39 and {key(r) for r in log_runs} == expected_grid, "Execution log covers the complete grid")
    run_lookup = {key(r): r for r in runs}
    for entry in log_runs:
        for name, value in entry.items():
            same(value, run_lookup[key(entry)][name], "Execution log " + name)
    cache = read_json(folder / "cache_validation.json")
    require(len(cache) == 12 and {(c["case"], c["repeat"]) for c in cache} == set(problem_map), "Cache validation covers every instance")
    require(all(c["passed"] and c["random_state_steps"] == 12 and c["max_absolute_cache_or_direct_update_error"] < 1e-10 for c in cache), "Archived cache checks passed")
    bias = read_json(folder / "bias_validation.json")
    require(bias["passed"] and bias["iterations"] == 200 and len(bias["records"]) == 5, "Archived deterministic bias/repair check")
    require(all(max(b["bias_error"], b["repair_error"]) < 1e-12 for b in bias["records"]), "Archived bias/repair errors")
    correction = read_json(folder / "accounting_correction.json")
    require(len(correction["records"]) == 39, "Initialization correction covers all runs")
    for entry in correction["records"]:
        require(entry["corrected_initialization_payload"] == run_lookup[key(entry)]["initialization_scalar_payload"]
                and entry["measured_runtime_unchanged"], "Initialization correction matches archived counts")

    full = {(r["case"], r["repeat"]): r for r in runs if r["participation"] == 1}
    groups = {}
    for r in runs:
        enriched = dict(r)
        base = full[r["case"], r["repeat"]]
        enriched["paired_payload_ratio"] = r["total_scalar_payload"] / base["total_scalar_payload"]
        enriched["paired_graph_ratio"] = r["total_graph_evaluations"] / base["total_graph_evaluations"]
        groups.setdefault((r["case"], r["participation"], r["mask_type"]), []).append(enriched)
    summary = read_json(folder / "summary.json")
    require(len(summary) == len(groups) == 13, "Thirteen summary groups")
    verify_csv_json(read_csv(folder / "summary.csv"), summary, "summary.csv / summary.json")
    metrics = ["rounds", "total_graph_evaluations", "total_scalar_payload", "sequential_seconds", "paired_payload_ratio", "paired_graph_ratio", "relative_gap", "normalized_residual"]
    table = (folder / "results_table.tex").read_text(encoding="utf-8")
    for s in summary:
        group = groups[s["case"], s["participation"], s["mask_type"]]
        require(s["replicates"] == len(group) == 3, "Three repetitions per summary group")
        require(s["converged"] == sum(r["converged"] for r in group), "Summary convergence count")
        for metric in metrics:
            for name, function in [("median", np.median), ("min", np.min), ("max", np.max)]:
                same(s[f"{metric}_{name}"], float(function([r[metric] for r in group])), "Summary " + metric + " " + name)
        sampling = f"$p={s['participation']:g}$" if s["mask_type"] == "bernoulli" else r"$2\times2$ rectangle"
        line = f"{LABELS[s['case']]} & {sampling} & {s['rounds_median']:,.0f} & {s['total_graph_evaluations_median']:,.0f} & {s['total_scalar_payload_median']/1e6:.3f} & {s['paired_payload_ratio_median']:.3f}" + r"\\"
        require(table.count(line) == 1, "Table row: " + s["case"] + "/" + sampling)

    points = read_csv(figures / "paired_workload_ratios.csv")
    require(len(points) == 39, "Thirty-nine plotted paired observations")
    for point, r in zip(points, runs):
        base = full[r["case"], r["repeat"]]
        expected = dict(case=r["case"], repeat=r["repeat"], strategy=strategy(r),
            graph_ratio=r["total_graph_evaluations"] / base["total_graph_evaluations"],
            payload_ratio=r["total_scalar_payload"] / base["total_scalar_payload"],
            graph_count=r["total_graph_evaluations"], baseline_graph_count=base["total_graph_evaluations"],
            payload_count=r["total_scalar_payload"], baseline_payload_count=base["total_scalar_payload"])
        for name, value in expected.items():
            same(point[name], value, "Figure paired input " + name)
    provenance = read_json(figures / "figure_provenance.json")
    require(provenance["total_runs"] == provenance["convergence_trajectories"] == provenance["participation_cost_source_points"] == 39, "All runs appear in both figures")
    require(provenance["excluded_runs"] == [], "No excluded plot records")
    require(provenance["manuscript_figure_mapping"] == {"convergence": 2, "participation_costs": 3}, "Manuscript figure mapping")
    require(set(provenance["sources"]) == {"runs.csv", "manifest.json"} | {r["trajectory_file"] for r in runs}, "Figure source coverage")
    for name, sha in provenance["sources"].items():
        require(digest(folder / name) == sha, "Figure source hash: " + name)
    require(len(provenance["centers"]) == 13, "Thirteen figure medians")
    for center in provenance["centers"]:
        group = [p for p in points if p["case"] == center["case"] and p["strategy"] == center["strategy"]]
        require(len(group) == center["n"] == 3, "Figure group size")
        same(center["graph_median"], float(np.median([float(p["graph_ratio"]) for p in group])), "Figure graph median")
        same(center["payload_median"], float(np.median([float(p["payload_ratio"]) for p in group])), "Figure payload median")
    return runs, dict(archived_runs=len(runs), archived_problem_arrays=len(problems), audit_rows=total_audits,
        replayed_activation_rounds=sum(r["rounds"] for r in runs), summary_and_table_rows=len(summary),
        paired_figure_points=len(points), hashed_reference_files=len(hashes), regenerated_problem_arrays=12,
        optimizers_executed=0, all_archived_runs_converged=all(r["converged"] for r in runs))


def compare_new(folder, reference):
    new = read_json(folder / "runs.json")
    require(bool(new), "Comparison directory contains run records")
    lookup = {key(r): r for r in reference}
    require(len({key(r) for r in new}) == len(new), "Comparison has unique run keys")
    for r in new:
        require(key(r) in lookup, "Comparison run belongs to the archived grid")
        base = lookup[key(r)]
        require(set(r) == set(base), "Comparison run fields match")
        for name, expected in base.items():
            if name not in TIMING:
                same(r[name], expected, "Reproduced " + str(key(r)) + ": " + name, rtol=1e-8, atol=1e-10)
        with np.load(folder / "data" / f"{r['case']}_r{r['repeat']}.npz", allow_pickle=False) as actual, \
             np.load(ROOT / "results" / "data" / f"{r['case']}_r{r['repeat']}.npz", allow_pickle=False) as archived:
            require(set(actual.files) == set(archived.files), "Comparison array fields")
            for name in actual.files:
                require(np.allclose(actual[name], archived[name], rtol=1e-8, atol=1e-10), "Reproduced array " + name)
    return dict(compared_runs=len(new), exact_integer_counts=True, numerical_relative_tolerance=1e-8,
        numerical_absolute_tolerance=1e-10, excluded_fields=sorted(TIMING),
        note="Times are measured anew and are not required to match. Other environments can change floating-point termination counts; a mismatch is reported, never silently accepted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="Compare an independently reproduced subset against the archive.")
    parser.add_argument("--report", type=Path, default=ROOT / "reproduced" / "verification.json")
    args = parser.parse_args()
    report_path = args.report.expanduser().resolve()
    for protected in [ROOT / "results", ROOT / "reference_figures"]:
        if report_path == protected or protected in report_path.parents:
            parser.error("Verification reports must be written outside the reference archives.")
    report = dict(status="failed", python=platform.python_version(), numpy=np.__version__,
                  scope="All archived metadata and data; no optimizer rerun by this command.")
    try:
        with threadpool_limits(limits=1):
            runs, counts = verify_archive()
            report.update(counts)
            if args.compare:
                report["comparison"] = compare_new(args.compare.expanduser().resolve(), runs)
        report["status"] = "passed"
    except (AssertionError, ValueError, KeyError, OSError) as error:
        report["error"] = str(error)
    report["checks"] = CHECKS
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
