"""Column-block proximal gradient and a matched original-gap R2WBS comparison."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_experiments import Problem, shrink, write_csv


def gram_setup_cost(p):
    return p.M * sum(int(n) * (int(n) + 1) // 2 for n in p.nj)


def run_column_blocks(problem, seed, audit_cost, *, share_gram=False):
    p = problem
    generator = np.random.default_rng(seed)
    x = np.zeros(p.A.shape[1])
    residuals = [-p.b[row].copy() for row in p.rows]
    if share_gram:
        # Each owner sends its Gram upper triangle. Symmetry determines the rest.
        lipschitz = []
        for j in range(p.N):
            gram = sum(p.block[i][j].T @ p.block[i][j] for i in range(p.M))
            lipschitz.append(np.linalg.eigvalsh(gram)[-1])
        payload = p.initialization_payload + gram_setup_cost(p)
    else:
        lipschitz = [sum(np.linalg.norm(p.block[i][j], 2)**2 for i in range(p.M))
                     for j in range(p.N)]
        payload = p.initialization_payload + p.M * p.N
    for iteration in range(1, 40001):
        j = int(generator.integers(p.N))
        col = p.cols[j]
        grad = sum(p.block[i][j].T @ p.gradient(residuals[i]) for i in range(p.M))
        new = shrink(x[col] - grad / lipschitz[j], p.tau / lipschitz[j])
        delta = new - x[col]
        for i in range(p.M):
            residuals[i] += p.block[i][j] @ delta
        x[col] = new
        payload += 4 * len(p.b) + 2 * p.M * p.nj[j]
        if iteration % 25 == 0:
            payload += audit_cost
            certificate = p.certificate(x, distributed=True)
            if certificate['relative_gap'] <= 1e-6:
                np.testing.assert_allclose(np.concatenate(residuals), p.A @ x - p.b,
                                           atol=1e-10, rtol=1e-10)
                return iteration, int(payload), certificate['relative_gap']
    raise RuntimeError(f"Gap tolerance missed for {p.meta['case']}, seed {seed}")


def compare(results):
    with (results / 'runs.csv').open() as stream:
        archived = list(csv.DictReader(stream))
    records = []
    for meta in json.loads((results / 'problems.json').read_text()):
        case, repeat = meta['case'], meta['repeat']
        with np.load(results / 'data' / f'{case}_r{repeat}.npz') as data:
            p = Problem(data['A'], data['b'], meta)
        gap_cost = int(p.graph_payload.sum() + p.row_payload + 3*p.M)
        seed = 82000 + repeat
        rounds, payload, gap = run_column_blocks(p, seed, gap_cost)
        for run in archived:
            if (run['case'], int(run['repeat'])) != (case, repeat):
                continue
            with (results / run['trajectory_file']).open() as stream:
                row = next(r for r in csv.DictReader(stream) if float(r['relative_gap']) <= 1e-6)
            k = int(row['round'])
            charged = p.initialization_payload + int(row['update_scalar_payload']) + k//25*gap_cost
            records.append(dict(case=case, repeat=repeat, seed=seed, p=run['participation'],
                                mask=run['mask_type'], column_rounds=rounds, column_payload=payload,
                                column_gap=gap, r2wbs_rounds=k, r2wbs_payload=charged,
                                r2wbs_gap=row['relative_gap'], column_over_r2wbs=payload/charged))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('reproduced/column_baseline.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        write_csv(args.output, compare(args.results))
