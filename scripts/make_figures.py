# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-d) Convergence traces -> LineTrend/plot_sweep.py -> param inherit
# (a-d) Paired workload tradeoffs -> cross-type inherit (LineTrend marker/line system) -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.
# The source asset's common-x, linear-scale sweep does not support ragged
# logarithmic optimization traces. All plotted values here come from recorded
# experiment outputs; no synthetic display data or seed selection is used.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, MaxNLocator

CASES = [
    ("correlated_lasso", "Correlated Lasso"),
    ("rank_deficient_lasso", "Rank-deficient Lasso"),
    ("uneven_huber", "Uneven Huber"),
    ("diabetes_quadratic", "Diabetes, quadratic features"),
]
# Color + line + marker redundantly encode the same strategy in both figures.
STYLES = {
    "full": dict(label="Full participation", color=CATEGORICAL[0], linestyle="-", marker="o"),
    "half": dict(label="Bernoulli, p = 1/2", color=CATEGORICAL[2], linestyle="--", marker="^"),
    "quarter": dict(label="Bernoulli, p = 1/4", color=CATEGORICAL[4], linestyle="-.", marker="s"),
    "rectangle": dict(label="Rectangle, p = 1/4", color=CATEGORICAL[1], linestyle=(0, (1, 1)), marker="D"),
}


def strategy(row):
    if row["mask_type"] == "rectangle_2x2":
        return "rectangle"
    return {1.0: "full", 0.5: "half", 0.25: "quarter"}[float(row["participation"])]


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def setup_grid(height_mm):
    # JOTA's smallextended template has a 119 mm single text column.
    fig, axes = plt.subplots(2, 2, figsize=(119/25.4, height_mm/25.4))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.12, top=0.84,
                        wspace=0.37, hspace=0.50)
    for k, (ax, (_, title)) in enumerate(zip(axes.flat, CASES)):
        ax.text(-0.20, 1.075, chr(97+k), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", color=BLACK)
        ax.text(0.01, 1.075, title, transform=ax.transAxes,
                fontsize=7.2, va="bottom", color=BLACK)
        ax.tick_params(length=2.6, pad=2)
        ax.grid(False)
    return fig, axes


def legend(fig, *, points=False):
    handles = []
    for st in STYLES.values():
        handles.append(Line2D([0], [0], color=st["color"],
            linestyle="none" if points else st["linestyle"],
            marker=st["marker"] if points else None, markersize=4,
            linewidth=1.0, label=st["label"]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.005),
               ncol=2, frameon=False, columnspacing=1.1, handlelength=2.2,
               handletextpad=0.5, borderaxespad=0, fontsize=7.2)


def make_convergence(rows, results, out, stem="convergence"):
    fig, axes = setup_grid(117)
    legend(fig)
    trace_rows=[]; included=[]
    min_gap=float("inf"); max_gap=0
    for row in rows:
        trace=load_csv(results/row["trajectory_file"])
        gaps=np.array([float(t["relative_gap"]) for t in trace])
        payload=np.array([int(t["total_scalar_payload"]) for t in trace])
        assert np.isfinite(gaps).all() and (gaps>0).all()
        assert np.all(np.diff(payload)>0)
        assert payload[-1]==int(row["total_scalar_payload"])
        assert np.isclose(gaps[-1],float(row["relative_gap"]),rtol=1e-13,atol=0)
        min_gap=min(min_gap,float(gaps.min()));max_gap=max(max_gap,float(gaps.max()))
        trace_rows.append((row,trace,payload/1e6,gaps))
    for ax,(case,_) in zip(axes.flat,CASES):
        case_rows=[t for t in trace_rows if t[0]["case"]==case]
        max_x=0
        for row,trace,x,y in case_rows:
            st=STYLES[strategy(row)]
            ax.plot(x,y,color=st["color"],linestyle=st["linestyle"],
                    linewidth=0.85,alpha=0.76,zorder=3)
            ax.plot(x[-1],y[-1],linestyle="none",marker=st["marker"],
                    color=st["color"],markersize=2.6,markeredgewidth=0.4,zorder=4)
            max_x=max(max_x,float(x[-1]))
            included.append(row["trajectory_file"])
        ax.set_yscale("log")
        ax.set_ylim(min(3e-8,min_gap*0.7),max(1.,max_gap*1.1))
        ax.set_xlim(0,max_x*1.045)
        ax.yaxis.set_major_locator(FixedLocator([1.,1e-2,1e-4,1e-6]))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.minorticks_off()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.axhline(1e-6,color=GREY,linewidth=0.6,linestyle=(0,(2,2)),zorder=1)
    for ax in axes[:,0]:ax.set_ylabel("Relative gap (log scale)",labelpad=2.5,fontsize=7.5)
    for ax in axes[1,:]:ax.set_xlabel("Payload (million real scalars)",labelpad=3,fontsize=7.2)
    assert len(included)==len(rows)==39 and len(set(included))==39
    save_cns_figure(fig,out/stem)
    plt.close(fig)
    return included


def make_tradeoffs(rows,out,stem="participation_costs"):
    fig,axes=setup_grid(108)
    legend(fig,points=True)
    base={(r["case"],r["repeat"]):r for r in rows if strategy(r)=="full"}
    points=[]; centers=[]
    for r in rows:
        b=base[r["case"],r["repeat"]]
        points.append(dict(case=r["case"],repeat=int(r["repeat"]),
            strategy=strategy(r),graph_ratio=int(r["total_graph_evaluations"])/int(b["total_graph_evaluations"]),
            payload_ratio=int(r["total_scalar_payload"])/int(b["total_scalar_payload"]),
            graph_count=int(r["total_graph_evaluations"]),baseline_graph_count=int(b["total_graph_evaluations"]),
            payload_count=int(r["total_scalar_payload"]),baseline_payload_count=int(b["total_scalar_payload"])))
    for ax,(case,_) in zip(axes.flat,CASES):
        for key,st in STYLES.items():
            p=[v for v in points if v["case"]==case and v["strategy"]==key]
            if not p:continue
            assert len(p)==3
            xy=np.array([[v["graph_ratio"],v["payload_ratio"]] for v in p])
            ax.plot(xy[:,0],xy[:,1],linestyle="none",marker=st["marker"],
                    markerfacecolor=st["color"],markeredgecolor=st["color"],
                    markersize=3.4,markeredgewidth=0.5,alpha=0.8,zorder=3)
            center=np.median(xy,axis=0)
            # A larger unfilled symbol surrounds the center without hiding an
            # individual point when the center coincides with an observation.
            ax.plot(center[0],center[1],linestyle="none",marker=st["marker"],
                    markerfacecolor="none",markeredgecolor=st["color"],
                    markersize=7.1,markeredgewidth=1.0,zorder=4)
            centers.append(dict(case=case,strategy=key,graph_median=float(center[0]),payload_median=float(center[1]),n=3))
        ax.axvline(1.,color=GREY,linewidth=0.6,linestyle=(0,(3,2)),zorder=1)
        ax.axhline(1.,color=GREY,linewidth=0.6,linestyle=(0,(3,2)),zorder=1)
        ax.set_xlim(0.69,1.035)
        ax.set_ylim(0.84,1.49)
        ax.set_xticks([0.7,0.8,0.9,1.0])
        ax.set_yticks([0.9,1.1,1.3,1.5])
        ax.text(0.03,0.96,"3 runs per strategy",transform=ax.transAxes,
                fontsize=7,ha="left",va="top",color=CATEGORICAL[5])
    for ax in axes[:,0]:ax.set_ylabel("Total payload ratio",labelpad=3,fontsize=7.5)
    for ax in axes[1,:]:ax.set_xlabel("Graph projection ratio",labelpad=3,fontsize=7.5)
    save_cns_figure(fig,out/stem)
    plt.close(fig)
    with (out/"paired_workload_ratios.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    return points,centers


def main():
    parser=argparse.ArgumentParser(description="Regenerate JOTA figures from all recorded numerical runs.")
    parser.add_argument("--results",type=Path,default=Path(__file__).resolve().parents[1]/"results")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"reproduced"/"figures")
    parser.add_argument("--figure-start",type=int,default=None,
                        help="Use manuscript stems FigN and Fig(N+1), e.g. --figure-start 2.")
    args=parser.parse_args()
    if args.figure_start is not None and args.figure_start < 1:
        parser.error("--figure-start must be a positive integer.")
    args.results=args.results.expanduser().resolve();args.output=args.output.expanduser().resolve()
    root=Path(__file__).resolve().parents[1]
    for protected in [args.results,root/"results",root/"reference_figures"]:
        if args.output == protected or protected in args.output.parents:
            parser.error("Figure output must be separate from input records and reference archives.")
    args.output.mkdir(parents=True,exist_ok=True)
    stems=("convergence","participation_costs") if args.figure_start is None else (
        f"Fig{args.figure_start}",f"Fig{args.figure_start+1}")
    rows=load_csv(args.results/"runs.csv")
    assert len(rows)==39
    assert all(r["converged"]=="True" for r in rows)
    traces=make_convergence(rows,args.results,args.output,stems[0])
    points,centers=make_tradeoffs(rows,args.output,stems[1])
    sources=[args.results/"runs.csv",args.results/"manifest.json"]+[args.results/t for t in traces]
    report=dict(total_runs=39,convergence_trajectories=39,participation_cost_source_points=39,
        output_stems=dict(convergence=stems[0],participation_costs=stems[1]),
        manuscript_figure_mapping=dict(convergence=2 if args.figure_start is None else args.figure_start,
            participation_costs=3 if args.figure_start is None else args.figure_start+1),
        excluded_runs=[],resampling="none",smoothing="none",interpolation="none",
        participation_cost_center="coordinatewise median of three paired ratios; hollow larger symbol",
        participation_cost_spread="all individual paired observations shown; no error bars or confidence interval",
        convergence_center="none; all original run traces plotted",
        baseline="full participation of same case and repeat; three baseline points coincide at (1,1) in each panel",
        replicates="Synthetic cases: independently seeded problem instances and activation schedules. Diabetes: same objective with different seeded row assignments and activation schedules.",
        statistics="No statistical significance tests or multiple-comparison correction; optimization workload comparison at a common certificate tolerance.",
        prediction="No train/test split or predictive accuracy claim; all observations define the displayed optimization problems.",
        payload="Charged real-valued unicast payload, including initialization and scheduled full audits; not measured network traffic.",
        graph_counts="Update and audit graph-projection calls included. Local factorization setup is separate and identical across strategies for one instance.",
        nominal_figure_width_mm=119,
        font_floor_pt=7,
        centers=centers,
        sources={str(p.relative_to(args.results)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/"figure_provenance.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["total_runs","convergence_trajectories","participation_cost_source_points","excluded_runs"]}))

if __name__=="__main__":main()
