#!/usr/bin/env python3
"""Aggregate all runs without selecting seeds or discarding unfavorable results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced'/'summary',
                    help='Write derived summaries here; input records are never modified.')
    args=ap.parse_args(); folder=args.results.expanduser().resolve(); output=args.output.expanduser().resolve()
    root=Path(__file__).resolve().parents[1]
    for protected in [folder,root/'results',root/'reference_figures']:
        if output == protected or protected in output.parents:
            ap.error('Summary output must be separate from input records and reference archives.')
    output.mkdir(parents=True,exist_ok=True)
    runs=json.loads((folder/'runs.json').read_text(encoding='utf-8'))
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not runs:
        ap.error('Input contains no completed run records.')
    groups={}
    full={(r['case'],r['repeat']):r for r in runs if r['participation']==1. and r['mask_type']=='bernoulli'}
    for r in runs:
        if (r['case'],r['repeat']) not in full:
            ap.error('Each case and repeat needs its full-participation run for paired ratios.')
        base=full[(r['case'],r['repeat'])]
        r['paired_payload_ratio']=r['total_scalar_payload']/base['total_scalar_payload']
        r['paired_graph_ratio']=r['total_graph_evaluations']/base['total_graph_evaluations']
        groups.setdefault((r['case'],r['participation'],r['mask_type']),[]).append(r)
    rows=[]
    metrics=['rounds','total_graph_evaluations','total_scalar_payload','sequential_seconds',
        'paired_payload_ratio','paired_graph_ratio','relative_gap','normalized_residual']
    for key,rs in groups.items():
        entry=dict(case=key[0],participation=key[1],mask_type=key[2],replicates=len(rs),converged=sum(r['converged'] for r in rs))
        for metric in metrics:
            values=[r[metric] for r in rs]
            for name,func in [('median',np.median),('min',np.min),('max',np.max)]:
                entry[f'{metric}_{name}']=float(func(values))
        rows.append(entry)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    labels={'correlated_lasso':'Correlated Lasso','rank_deficient_lasso':'Rank-deficient Lasso',
        'uneven_huber':'Uneven Huber','diabetes_quadratic':'Diabetes quadratic'}
    lines=[r'\begin{table}[t]',r'\caption{Common-accuracy results. Entries are medians of three repetitions; the last column is the median paired payload ratio relative to full participation. Graph counts and payloads include every audit. The raw records include the full ranges.}',
        r'\label{tab:archived_run_summary}',r'\centering',r'\small',r'\setlength{\tabcolsep}{3.3pt}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Problem & Graph sampling & Rounds & Graph prox. & Payload ($10^6$) & Ratio\\',r'\midrule']
    for row in rows:
        sample=f"$p={row['participation']:g}$" if row['mask_type']=='bernoulli' else r'$2\times2$ rectangle'
        lines.append(f"{labels[row['case']]} & {sample} & {row['rounds_median']:,.0f} & {row['total_graph_evaluations_median']:,.0f} & {row['total_scalar_payload_median']/1e6:.3f} & {row['paired_payload_ratio_median']:.3f}"+r'\\')
    if any(row['replicates'] != 3 for row in rows):
        lines[1]=lines[1].replace('three repetitions','the available repetitions')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (output/'results_table.tex').write_text('\n'.join(lines)+'\n')
    planned={(case,repeat,probability,'bernoulli') for case in manifest['cases']
             for repeat in range(manifest['repeats']) for probability in manifest['probabilities']}
    include_rectangle=manifest.get('include_rectangle',manifest['planned_run_count']>len(planned))
    if include_rectangle and 'correlated_lasso' in manifest['cases']:
        planned.update(('correlated_lasso',repeat,0.25,'rectangle_2x2') for repeat in range(manifest['repeats']))
    observed={(r['case'],r['repeat'],r['participation'],r['mask_type']) for r in runs}
    audit=dict(run_count=len(runs),all_converged=all(r['converged'] for r in runs),
        maximum_relative_gap=max(r['relative_gap'] for r in runs),
        maximum_normalized_residual=max(r['normalized_residual'] for r in runs),
        maximum_final_cache_error=max(r['cache_max_absolute_error'] for r in runs),
        minimum_objective_minus_reference=min(r['objective_minus_reference'] for r in runs),
        maximum_objective_minus_reference=max(r['objective_minus_reference'] for r in runs),
        all_planned_runs_retained=(
            len(runs)==manifest['planned_run_count'] and
            len(observed)==len(runs) and observed==planned),
        statistics='Median and full min-max range over three independent synthetic instances, or three sample partitions of the fixed real-data objective. Paired ratios use the same repeat full-participation run. No hypothesis test or confidence interval is used.')
    if any(row['replicates'] != 3 for row in rows):
        audit['statistics']='Median and full min-max range over all available repetitions; group counts are recorded in summary.json. Paired ratios use the same-repeat full-participation run. No hypothesis test or confidence interval is used.'
    (output/'integrity_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
