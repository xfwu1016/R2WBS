"""Post hoc protocol checks on the archived data and activation streams.

Run from any directory. Output defaults to reproduced/protocol_checks.
Archived L=25 observations are inputs; L=10 and L=50 are independent replays.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from run_experiments import Problem, run_method, write_csv


def check_tilt():
    # Order: row loss y, regularizer x, graph y, graph x. rho = 1.
    a, b = 1.3, 0.7

    def iterate(initial_dual):
        eta = np.zeros(4)
        dual = np.array(initial_dual, dtype=float)
        for _ in range(200):
            y, x, yg, xg = eta - dual
            graph_x = (xg + a * yg) / (1 + a*a)
            xi = np.array([(y + b)/2, x/2, a * graph_x, graph_x])
            eta = (xi + xi[[2, 3, 0, 1]]) / 2
            dual += xi - eta
        return eta[1]

    records = []
    for c in [-2., -1., 0.5, 1.25, 2.]:
        tilted = iterate([0, c, 0, 0])
        repaired = iterate([0, c/2, 0, -c/2])
        pred = (a*b-c)/(1+a*a)
        orig = a*b/(1+a*a)
        records.append(dict(c=c, tilted=tilted, predicted_tilted=pred,
                            repaired=repaired, original=orig,
                            tilt_error=abs(tilted-pred), repair_error=abs(repaired-orig)))
    assert max(max(r['tilt_error'], r['repair_error']) for r in records) < 1e-12
    return records


def deployments(row, problem):
    rounds, audits = int(row['rounds']), int(row['audits'])
    original = int(row['total_scalar_payload'])
    m, n = problem['m'], problem['n']
    groups = len(problem['row_sizes'])
    # a_j is transmitted once per (i,j), saving M*n scalars per joint audit.
    fused = original - audits*groups*n
    # Moving row losses and responses to the hub removes row--hub messages.
    colocated = original - (m+groups) - 2*m*rounds - (4*m+3*groups)*audits
    return dict(separate_remote=original, fused_remote=fused,
                separate_colocated=colocated, fused_colocated=colocated-audits*groups*n)


def paired_summary(rows, measures, extra_key=None):
    def group_key(row):
        return (row['case'], row['repeat'], row[extra_key] if extra_key else None)

    baseline = {group_key(r): r for r in rows if r['probability'] == 1}
    grouped = {}
    for row in rows:
        if row['probability'] == 1:
            continue
        reference = baseline[group_key(row)]
        for metric in measures:
            key = (row['case'], row['probability'], row['mask'],
                   row[extra_key] if extra_key else None, metric)
            grouped.setdefault(key, []).append(row[metric]/reference[metric])
    return [dict(case=c, probability=p, mask=m, spacing=l or 25, metric=metric,
                 pairs=len(v), median=float(np.median(v)), minimum=min(v), maximum=max(v),
                 below_one=sum(r < 1 for r in v)) for (c,p,m,l,metric),v in grouped.items()]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root/'results')
    parser.add_argument('--output', type=Path, default=root/'reproduced'/'protocol_checks')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.results
    with (data/'runs.csv').open(newline='', encoding='utf-8') as stream:
        runs = list(csv.DictReader(stream))
    problems = {(r['case'], r['repeat']): r for r in json.loads((data/'problems.json').read_text())}
    write_csv(args.output/'nonzero_graph_tilt.csv', check_tilt())
    charges, spacings = [], []
    with threadpool_limits(limits=1):
        for row in runs:
            case, repeat = row['case'], int(row['repeat'])
            cfg = dict(case=case, repeat=repeat, probability=float(row['participation']), mask=row['mask_type'])
            meta = problems[case, repeat]
            charges.append(dict(**cfg, **deployments(row, meta)))
            with np.load(data/meta['data_file']) as archive:
                problem = Problem(archive['A'], archive['b'], meta)
            for spacing in (10, 25, 50):
                if spacing == 25:
                    rec = row
                else:
                    rec, _, _ = run_method(problem, cfg['probability'], int(row['activation_seed']),
                                          40000, spacing, 1e-6, 1e-5, cfg['mask'])
                assert str(rec['converged']).lower() == 'true'
                spacings.append(dict(**cfg, spacing=spacing, rounds=int(rec['rounds']),
                                     graph_calls=int(rec['total_graph_evaluations']),
                                     payload=int(rec['total_scalar_payload']),
                                     relative_gap=float(rec['relative_gap']),
                                     normalized_residual=float(rec['normalized_residual'])))
            print(case, repeat, cfg['probability'], cfg['mask'], flush=True)
    write_csv(args.output/'audit_spacing.csv', spacings)
    write_csv(args.output/'deployment_charges.csv', charges)
    summary = paired_summary(charges, ('separate_remote','fused_remote','separate_colocated','fused_colocated'))
    summary += paired_summary(spacings, ('rounds','graph_calls','payload'), extra_key='spacing')
    write_csv(args.output/'paired_summary.csv', summary)


if __name__ == '__main__':
    main()
