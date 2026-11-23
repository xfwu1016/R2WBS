"""Rebuild stopping-rule comparisons from the archived experiment records.

Usage: python experiments/audit_stopping_rules.py [--results results]
The original inputs are read only. Derived CSV, JSON and TeX files go to
results/diagnostics/. No numerical packages are required.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


LABELS = {"correlated_lasso": "Correlated", "rank_deficient_lasso": "Rank-def.",
          "uneven_huber": "Huber", "diabetes_quadratic": "Diabetes"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_work(counts, audits, row_sizes, col_sizes):
    # Leading flops for the actual cached-factor graph() branches.
    # Two dense matrix-vector products on the column branch, three on the row
    # branch, plus two triangular solves on the smaller dimension.
    total = 0
    for i, m in enumerate(row_sizes):
        for j, n in enumerate(col_sizes):
            products = 2 if n <= m else 3
            weight = 2 * products * m * n + 2 * min(m, n) ** 2
            total += (counts[i][j] + audits) * weight
    return total


def table(caption, label, columns, header, rows):
    lines = [r"\begin{table}[tbp]", r"\centering\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + columns + "}", r"\toprule", header + r"\\", r"\midrule"]
    lines.extend(" & ".join(row) + r"\\" for row in rows)
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main():
    default_results = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument("--output", type=Path,
                        default=default_results.parent / "reproduced" / "diagnostics")
    args = parser.parse_args()
    results, out = args.results, args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((results / "manifest.json").read_text())
    problems = {(r["case"], r["repeat"]): r for r in json.loads((results / "problems.json").read_text())}
    records, audit_ratios = [], defaultdict(list)
    normalized_ratios = defaultdict(list)
    for run in read_csv(results / "runs.csv"):
        case, repeat = run["case"], int(run["repeat"])
        problem = problems[case, repeat]
        trajectory_path = results / run["trajectory_file"]
        trajectory = read_csv(trajectory_path)
        for audit in trajectory:
            cert = math.sqrt(problem["rho"]) * float(audit["fixed_point_residual"])
            if cert > 0:
                audit_ratios[case].append(float(audit["relative_gap"]) / cert)
                normalized_ratios[case].append(float(audit["relative_gap"]) / float(audit["normalized_residual"]))
        first = next(r for r in trajectory if float(r["normalized_residual"]) <= manifest["normalized_residual_tolerance"])
        joint = next(r for r in trajectory if float(r["normalized_residual"]) <= manifest["normalized_residual_tolerance"]
                     and float(r["relative_gap"]) <= manifest["gap_tolerance"])
        assert int(joint["round"]) == int(run["rounds"])
        row_sizes, col_sizes = problem["row_sizes"], problem["col_sizes"]
        full_graphs = len(row_sizes) * len(col_sizes)
        audit_cost = 2 * (len(col_sizes) * sum(row_sizes) + len(row_sizes) * sum(col_sizes) + sum(row_sizes))
        counts = json.loads(trajectory_path.with_name(trajectory_path.stem + "_activation_counts.json").read_text())
        assert sum(map(sum, counts)) == int(run["update_graph_evaluations"])
        for rule, row in (("joint", joint), ("lifted", first)):
            audits = int(row["audit_graph_evaluations"]) // full_graphs
            entry = dict(case=case, repeat=repeat, p=float(run["participation"]), mask=run["mask_type"], rule=rule,
                         rounds=int(row["round"]), graph=int(row["audit_graph_evaluations"])+int(row["update_graph_evaluations"]),
                         payload=int(row["total_scalar_payload"]), relative_gap=float(row["relative_gap"]),
                         normalized_residual=float(row["normalized_residual"]))
            entry["lifted_audit_payload_only"] = (int(row["initialization_scalar_payload"]) + int(row["update_scalar_payload"])
                                                  + audits * audit_cost)
            entry["serial_seconds"] = float(run["sequential_seconds"]) if rule == "joint" else ""
            entry["weighted_graph"] = graph_work(counts, audits, row_sizes, col_sizes) if rule == "joint" else ""
            records.append(entry)
    bases = {(r["case"], r["repeat"], r["rule"]): r for r in records if r["p"] == 1}
    metrics = ("rounds", "graph", "payload", "lifted_audit_payload_only", "serial_seconds", "weighted_graph")
    groups = defaultdict(list)
    for row in records:
        base = bases[row["case"], row["repeat"], row["rule"]]
        for metric in metrics:
            row[metric + "_ratio"] = row[metric] / base[metric] if row[metric] != "" else ""
        groups[row["case"], row["p"], row["mask"], row["rule"]].append(row)
    write_csv(out / "paired_stops.csv", records)
    summaries = []
    for key, repetitions in groups.items():
        summary = dict(zip(("case", "p", "mask", "rule"), key))
        for metric in ("relative_gap", "normalized_residual", "rounds", "graph", "payload") + tuple(m+"_ratio" for m in metrics):
            values = [r[metric] for r in repetitions if r[metric] != ""]
            for name, fn in (("min", min), ("median", median), ("max", max)):
                summary[metric+"_"+name] = fn(values) if values else ""
        summaries.append(summary)
    write_csv(out / "paired_ranges.csv", summaries)
    audit_stats = {case: dict(n=len(v), minimum=min(v), median=median(v), maximum=max(v))
                   for case, v in audit_ratios.items()}
    for case, values in normalized_ratios.items():
        audit_stats[case]["using_normalized_residual"] = dict(minimum=min(values), median=median(values), maximum=max(values))
    (out / "gap_certificate_ratio.json").write_text(json.dumps(audit_stats, indent=2)+"\n")
    indexed = {(r["case"],r["p"],r["mask"],r["rule"]):r for r in summaries}
    tab1, tab2 = [], []
    for (case,p,mask,rule), row in indexed.items():
        if rule != "joint":
            continue
        first = indexed[case,p,mask,"lifted"]
        label = r"$2{\times}2$" if mask.startswith("rectangle") else {1:"1", 0.5:"1/2", 0.25:"1/4"}[p]
        tab1.append([LABELS[case], label, f"{row['rounds_median']:,.0f}",f"{row['graph_median']:,.0f}",
                     f"{row['payload_median']/1e6:.3f}",f"{row['relative_gap_max']*1e6:.3f}",
                     f"{row['normalized_residual_max']*1e6:.3f}"])
        if p<1:
            tab2.append([LABELS[case],label,f"{row['graph_ratio_median']:.3f}",f"{row['payload_ratio_median']:.3f}",
                         f"{row['serial_seconds_ratio_median']:.3f}",f"{first['graph_ratio_median']:.3f}",
                         f"{first['payload_ratio_median']:.3f}",f"{first['relative_gap_median']*1e5:.2f}"])
    (out/"accuracy_table.tex").write_text(table(
        r"Joint stopping rule. Rounds, graph calls and payload (millions of scalars) are medians over three runs. The last two columns give the largest relative gap and normalized residual, both in units of $10^{-6}$, over the same runs. Sampling labels give the Bernoulli probability or the rectangle size.",
        "tab:numerical_results","llrrrrr",r"Problem & Mask & Rounds & Graph calls & Payload & Gap & Residual",tab1))
    (out/"stop_table.tex").write_text(table(
        r"Paired medians relative to full participation at the same stopping rule. $G$ counts graph projections, $B$ counts transmitted scalars, and $T$ is recorded serial solve time. Both payload columns include the diagnostic gap pass. The last column is the median relative gap at the first lifted stop in units of $10^{-5}$.",
        "tab:stop_rules","llrrrrrr",r" & & \multicolumn{3}{c}{Joint stop} & \multicolumn{3}{c}{Lifted stop}\\ \cmidrule(lr){3-5}\cmidrule(lr){6-8} Problem & Mask & $G$ & $B$ & $T$ & $G$ & $B$ & Gap",tab2))
    for case in LABELS:
        full = indexed[case,1,"bernoulli","joint"]["rounds_median"]
        quarter = indexed[case,0.25,"bernoulli","joint"]["rounds_median"]
        print(case, "round endpoint exponent", math.log(quarter/full)/math.log(4))
    print(json.dumps(audit_stats, indent=2))


if __name__ == "__main__":
    main()
