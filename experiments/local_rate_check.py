"""Measure the deterministic distance/residual ratio on correlated Lasso, repeat 0."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_experiments import CacheState, Problem, write_csv


def fixed_state(p, reference):
    # Recover the active solution, then verify primal uniqueness and all KKT conditions.
    support = np.abs(reference) > 1e-8
    active = p.A[:, support]
    signs = np.sign(reference[support])
    x = np.zeros_like(reference)
    x[support] = np.linalg.solve(active.T @ active, active.T @ p.b - p.tau * signs)
    if np.linalg.matrix_rank(p.A) != p.A.shape[1]:
        raise ValueError("This diagnostic requires a full-column-rank design")
    np.testing.assert_array_equal(np.sign(x[support]), signs)
    prediction = p.A @ x
    g = prediction - p.b
    Atg = p.A.T @ g
    np.testing.assert_allclose(Atg[support], -p.tau * signs, atol=1e-12, rtol=0)
    if np.max(np.abs(Atg[~support])) > p.tau + 1e-12:
        raise ValueError("Inactive KKT conditions failed")
    # The unique primal lift plus the unique scaled dual (rho=1).
    z = CacheState(p)
    for i, row in enumerate(p.rows):
        z.y[i] = prediction[row] - g[row]
    for j, col in enumerate(p.cols):
        z.x[j] = x[col] + Atg[col]
        for i, row in enumerate(p.rows):
            a = p.block[i][j]
            z.yg[i][j] = a @ x[col] + g[row]
            z.xg[i][j] = x[col] - a.T @ g[row]
    z.C = [z.x[j] + sum(z.xg[i][j] for i in range(p.M)) for j in range(p.N)]
    z.E = [z.y[i] - sum(z.yg[i]) for i in range(p.M)]
    if z.audit()['fixed_point_residual'] > 1e-11:
        raise ValueError("Computed KKT lift is not a numerical fixed point")
    return z.vector()


def measure(results):
    case = 'correlated_lasso'
    meta = next(m for m in json.loads((results/'problems.json').read_text())
                if (m['case'], m['repeat']) == (case, 0))
    with np.load(results/'data'/f'{case}_r0.npz') as data:
        problem = Problem(data['A'], data['b'], meta)
        target = fixed_state(problem, data['reference_x'])
    with (results/'trajectories'/f'{case}_r0_p1_bernoulli.csv').open() as stream:
        audits = {int(r['round']): float(r['fixed_point_residual']) for r in csv.DictReader(stream)}
    state = CacheState(problem)
    full_mask = np.ones((problem.M, problem.N), dtype=bool)
    records = []
    for k in range(max(audits) + 1):
        distance = np.linalg.norm(state.vector() - target)
        if k in audits:
            residual = state.audit()['fixed_point_residual']
            np.testing.assert_allclose(residual, audits[k], rtol=1e-7, atol=1e-13)
        state.step(full_mask)
        if k in audits:
            ratio = distance / residual
            contraction = (np.linalg.norm(state.vector() - target) / distance)**2
            upper = 1 - 1 / ratio**2
            if contraction > upper + 1e-8:
                raise ValueError("Firm nonexpansiveness check failed")
            records.append(dict(round=k, distance=distance, residual=residual,
                                distance_over_residual=ratio, next_squared_ratio=contraction,
                                pointwise_upper=upper))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('reproduced/local_rate_check.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        write_csv(args.output, measure(args.results))
