# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Single two-way architecture schematic -> cross-type inherit -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.
# No algorithm schematic asset exists. SankeyDiagram/plot_SankeyDiagram.py and
# its companion PNG were inspected: its numerical flow data are incompatible.
# Inherited: Arial-compatible sans typography, directly labelled flows,
# and no axes/grid. Exact role/arrow geometry is
# original and follows sections/theory.tex; no synthetic numerical data are used.

"""Render a compact two-way role architecture as PDF/SVG/PNG.

Standalone usage (no manuscript or LaTeX installation required)::

    python scripts/make_algorithm_schematic.py
    python scripts/make_algorithm_schematic.py --output path/to/figures

Optional manuscript validation and QA report::

    python scripts/make_algorithm_schematic.py \
        --manuscript-dir path/to/manuscript --output path/to/figures \
        --qa path/to/algorithm_schematic_qa.txt

The default output directory is <repository root>/reproduced/figures.
--manuscript-dir is optional; only an explicitly supplied path triggers reads
of sections/theory.tex, main.tex, and svglov3.clo. --qa writes an optional text
report and a companion JSON report. Python, Matplotlib, and its dependencies
(including fontTools) suffice. Each export also has an identical Fig1 alias.

Scientific provenance: "Randomized Two-Way Block Splitting for Hybrid Federated
Convex Optimization", Algorithm "Randomized Two-Way Block Splitting" and source
labels alg:randomized, eq:cache, eq:projection-cache, eq:row-update,
eq:column-update, eq:client-update, and eq:cache-commit in sections/theory.tex.
The final dimensions are 119 x 70 mm: the 119 mm width follows the manuscript's
svjour3 smallextended option. Fourteen English words label the coordinator,
two shared message directions, role types, partition axes, and activation key.
The generic 3-by-3 display represents row indices 1,i,M and column indices
1,j,N; it is an illustration, not a fixed deployment or a numerical dataset.
The caption and Algorithm 1 supply exact message symbols and audit semantics.
There are no numerical observations, random samples, or statistics.
"""

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


from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from fontTools import subset
from fontTools.ttLib import TTFont


# Scientific contract:
# The coordinator exchanges current aggregates and increments with active
# local roles. Each graph client retains its matrix A_ij; row loss roles and
# column regularizer roles align with the sample-by-feature partition.
# Active roles have a filled corner dot and blue edge; inactive roles remain
# visible in grey. The illustrative selected rows/columns are not a required
# sampling scheme. The two arrows terminate at a logical-role grouping, not
# at data blocks, and do not require inactive roles to communicate.
# Message symbols, snapshot/commit semantics, and full audits are in the
# manuscript caption, equations, and Algorithm 1 rather than inside the image.
# Archetype: schematic-led, single two-way architecture diagram.
# Scope: logical roles can share a physical node. Matrices stay at graph clients.
# Neither privacy guarantees nor a required physical placement are asserted.
# Statistics: not applicable; all content is an exact protocol schematic.

ROOT = Path(__file__).resolve().parents[1]
WIDTH_MM = 119.0  # main.tex: smallextended; svglov3.clo: textwidth=11.9 cm.
HEIGHT_MM = 70.0
RASTER_DPI = 1200
BLUE = CATEGORICAL[0]
TEXT = BLACK
LINE = "#666666"
BLUE_FILL = "#F4F8FC"
ACTIVE_FILL = "#E6EFF7"
INACTIVE_FILL = "#F6F7F8"
INACTIVE_EDGE = "#A8B0B7"
INACTIVE_TEXT = "#66717A"
GROUP_EDGE = "#CDD3D8"
WHITE = "#FFFFFF"

mpl.rcParams.update({
    "mathtext.fontset": "dejavusans",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
})
# Resolve one installed family so SVG embedding follows the actual PDF font
# on Linux, Windows, or macOS. DejaVu Sans is bundled with Matplotlib.
_body_font_path = font_manager.findfont(font_manager.FontProperties(
    family=["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]))
BODY_FONT = font_manager.FontProperties(fname=_body_font_path).get_name()
mpl.rcParams["font.sans-serif"] = [BODY_FONT]
mpl.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": BODY_FONT,
    "mathtext.it": f"{BODY_FONT}:italic",
    "mathtext.bf": f"{BODY_FONT}:bold",
    "mathtext.cal": BODY_FONT,
    "mathtext.sf": BODY_FONT,
    "mathtext.tt": BODY_FONT,
    "mathtext.fallback": None,
})


def compose_figure(archetype="schematic-led"):
    if archetype != "schematic-led":
        raise ValueError("This exact protocol figure uses a schematic-led layout.")
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH_MM), ylim=(0, HEIGHT_MM))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, size=10.5, weight="normal", color=TEXT,
          ha="center", va="center", **kwargs):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, clip_on=False, **kwargs)


def box(ax, x, y, width, height, edge=LINE, fill=WHITE, dashed=False, lw=0.7,
        radius=1.1, zorder=1, gid=None):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=zorder,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=LINE, dashed=False, lw=0.8, head=7):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head,
        shrinkA=0, shrinkB=0, linewidth=lw, color=color,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=2,
    )
    ax.add_patch(patch)
    return patch


def draw_roles(ax):
    """Align local roles on the genuine sample-by-feature partition.

    These nine displayed graph blocks stand for generic indices 1,i,M and
    1,j,N. The first two displayed rows/columns are selected for this
    illustrative round, a valid instance of the paper's rectangular sampler.
    This is a structural illustration and has no numerical data entry point.
    """
    # One logical-role group receives the shared arrow pair. There are no
    # direct matrix arrows and no communication links incident on grey nodes.
    box(ax, 48.5, 4.0, 68.0, 62.0, edge=GROUP_EDGE, lw=0.6,
        radius=1.5, zorder=0, gid="local-role-group")

    box(ax, 2.0, 31.0, 22.0, 16.0, edge=BLUE, fill=BLUE_FILL,
        lw=0.85, radius=1.2, gid="coordinator")
    label(ax, 13.0, 39.0, "Coordinator", size=10.0, weight="bold", color=BLUE)
    arrow(ax, (24.2, 43.0), (48.3, 43.0), color=BLUE, lw=0.9, head=7.5)
    arrow(ax, (48.3, 35.0), (24.2, 35.0), color=TEXT,
          dashed=True, lw=0.85, head=7.5)
    label(ax, 36.2, 49.0, "Current\naggregates", size=9.5,
          color=BLUE, linespacing=1.15)
    label(ax, 36.2, 30.8, "Increments", size=9.5)

    # The headings and aligned nodes carry the row/column association.
    label(ax, 94.0, 63.4, "Feature groups", size=9.5)
    ax.plot([72.5, 72.5, 115.5, 115.5], [58.8, 60.0, 60.0, 58.8],
            color=GROUP_EDGE, lw=0.65, solid_capstyle="butt", zorder=1)
    label(ax, 51.9, 34.0, "Sample groups", size=9.5, rotation=90)
    label(ax, 61.5, 55.6, "Loss", size=9.5, weight="bold")
    label(ax, 94.0, 55.6, "Graph clients", size=9.5, weight="bold")
    label(ax, 60.8, 10.0, "Regularizers", size=9.0, weight="bold")

    row_ids, col_ids = ["1", "i", "M"], ["1", "j", "N"]
    rows, columns = [47.0, 34.0, 21.0], [78.5, 94.0, 109.5]
    for row, (row_id, y) in enumerate(zip(row_ids, rows)):
        local_role(ax, 61.5, y, 9.5, 9.5, f"$l_{{{row_id}}}$",
                   active=row < 2, gid=f"loss-{row_id}")
        for col, (col_id, x) in enumerate(zip(col_ids, columns)):
            local_role(ax, x, y, 12.0, 9.5,
                       f"$A_{{{row_id}{col_id}}}$",
                       active=row < 2 and col < 2,
                       gid=f"graph-{row_id}-{col_id}")
    for col, (col_id, x) in enumerate(zip(col_ids, columns)):
        local_role(ax, x, 10.0, 12.0, 6.5, f"$r_{{{col_id}}}$",
                   active=col < 2, gid=f"regularizer-{col_id}")

    # Filled dots encode activation independently of blue/grey color.
    for y, title, active in [(17.0, "Active", True), (10.0, "Inactive", False)]:
        local_role(ax, 7.0, y, 4.3, 4.1, None, active=active,
                   gid=f"key-{title.lower()}", marker_radius=0.43)
        label(ax, 11.2, y, title, size=9.5, ha="left")


def local_role(ax, x, y, width, height, text, active, gid, marker_radius=0.5):
    """One local role; filled corner dots redundantly denote activation."""
    edge = BLUE if active else INACTIVE_EDGE
    fill = ACTIVE_FILL if active else INACTIVE_FILL
    box(ax, x - width / 2, y - height / 2, width, height, edge=edge,
        fill=fill, lw=0.9 if active else 0.65, radius=0.65, gid=gid)
    if text is not None:
        # Mathtext renders subscripts at 70% of 13 pt = 9.1 pt.
        label(ax, x, y - 0.25, text, size=13.0,
              color=BLUE if active else INACTIVE_TEXT)
    if active:
        dot = Circle((x + width / 2 - 1.15, y + height / 2 - 1.15),
                     marker_radius, facecolor=BLUE, edgecolor="none", zorder=3)
        ax.add_patch(dot)


def embed_svg_fonts(path):
    """Keep SVG text editable while embedding the actual subset font files.

    Matplotlib's svg.fonttype='none' preserves text but otherwise depends on
    fonts installed on the reader's computer. Embedded WOFF subsets make the
    exported SVG self-contained without turning editable text into paths.
    """
    svg = path.read_text(encoding="utf-8")
    characters = "".join(ET.fromstring(svg).itertext())
    declarations = []
    embedded = []
    for family, style, weight in [
        (BODY_FONT, "normal", "normal"),
        (BODY_FONT, "normal", "bold"),
        (BODY_FONT, "italic", "normal"),
    ]:
        properties = font_manager.FontProperties(family=family, style=style, weight=weight)
        font_path = font_manager.findfont(properties, fallback_to_default=False)
        font = TTFont(font_path)
        options = subset.Options()
        options.flavor = "woff"
        options.drop_tables.append("FFTM")
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        font.flavor = "woff"
        buffer = io.BytesIO()
        font.save(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        css_weight = "700" if weight == "bold" else "400"
        declarations.append(
            f"@font-face {{font-family:'{family}';font-style:{style};"
            f"font-weight:{css_weight};src:url(data:font/woff;base64,{encoded}) "
            "format('woff');}"
        )
        embedded.append({"family": family, "style": style, "weight": css_weight})
    font_style = '<style type="text/css"><![CDATA[\n' + "\n".join(declarations) + "\n]]></style>"
    if "<defs>" not in svg:
        raise ValueError("Matplotlib SVG has no definitions section for embedded fonts.")
    svg = svg.replace("<defs>", "<defs>\n" + font_style, 1)
    path.write_text(svg, encoding="utf-8")
    return embedded


def check_layout(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [(item, item.get_window_extent(renderer)) for item in ax.texts]
    outside = [item.get_text() for item, extent in extents
               if not (extent.x0 >= 0 and extent.y0 >= 0 and
                       extent.x1 <= fig.bbox.width and extent.y1 <= fig.bbox.height)]
    overlaps = []
    for index, (first, first_extent) in enumerate(extents):
        for second, second_extent in extents[index + 1:]:
            if first_extent.overlaps(second_extent):
                overlaps.append([first.get_text(), second.get_text()])
    if outside or overlaps:
        raise ValueError(f"Text layout failed: outside={outside}; overlaps={overlaps}")
    labels = [item.get_text() for item in ax.texts]
    prose = " ".join(re.sub(r"\$.*?\$", "", text) for text in labels)
    word_count = len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", prose))
    expected = {"Coordinator", "Loss", "Graph clients", "Regularizers",
                "Current\naggregates", "Increments", "Feature groups",
                "Sample groups", "Active", "Inactive"}
    text_labels = {text for text in labels if not text.startswith("$")}
    graph_boxes = sum((item.get_gid() or "").startswith("graph-")
                      for item in ax.patches)
    loss_boxes = sum((item.get_gid() or "").startswith("loss-")
                     for item in ax.patches)
    regularizer_boxes = sum((item.get_gid() or "").startswith("regularizer-")
                            for item in ax.patches)
    message_arrows = sum(isinstance(item, FancyArrowPatch) for item in ax.patches)
    minimum_font = min(item.get_fontsize() for item in ax.texts)
    if (len(labels) != 25 or text_labels != expected or word_count != 14 or
            graph_boxes != 9 or loss_boxes != 3 or regularizer_boxes != 3 or
            message_arrows != 2 or minimum_font < 9):
        raise ValueError("The two-way architecture figure contract failed.")
    return {"text_outside_page": outside, "text_overlaps": overlaps,
            "labels": labels, "local_roles": {
                "graph": graph_boxes, "loss": loss_boxes,
                "regularizer": regularizer_boxes},
            "activation_pattern": {
                "active_graph_indices": ["11", "1j", "i1", "ij"],
                "active_loss_indices": ["1", "i"],
                "active_regularizer_indices": ["1", "j"],
                "scope": "Illustrative rectangular selection, not a required sampler"},
            "message_arrows": message_arrows,
            "label_count": len(labels), "english_words_outside_math": word_count,
            "font_size_pt": {"minimum": minimum_font,
                             "maximum": max(item.get_fontsize() for item in ax.texts),
                             "math_subscript": 13.0 * 0.7}}


def validate_sources(manuscript_dir):
    """Validate a supplied manuscript tree; never require it for rendering."""
    if manuscript_dir is None:
        return {"validation": "not requested", "source": None, "sha256": None}
    source_path = manuscript_dir / "sections" / "theory.tex"
    source = source_path.read_text(encoding="utf-8")
    class_text = (manuscript_dir / "svglov3.clo").read_text(encoding="utf-8")
    main_text = (manuscript_dir / "main.tex").read_text(encoding="utf-8")
    expected_labels = ["eq:cache", "eq:projection-cache", "eq:row-update",
                       "eq:column-update", "eq:client-update", "eq:cache-commit",
                       "alg:randomized"]
    for expected in expected_labels:
        if expected not in source:
            raise ValueError(f"Missing manuscript source label: {expected}")
    if r"\setlength{\textwidth}{11.9cm}" not in class_text:
        raise ValueError("Recheck the native manuscript text width.")
    if "smallextended" not in main_text:
        raise ValueError("The 119 mm schematic assumes svjour3 smallextended.")
    return {"validation": "passed", "source": str(source_path),
            "sha256": hashlib.sha256(source.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced" / "figures",
                        help="Output directory (default: repository_root/reproduced/figures).")
    parser.add_argument("--manuscript-dir", type=Path,
                        help="Optional manuscript tree to validate before rendering.")
    parser.add_argument("--qa", type=Path,
                        help="Optional text QA/provenance report path (also writes a JSON companion).")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.manuscript_dir is not None:
        args.manuscript_dir = args.manuscript_dir.expanduser().resolve()
    qa_json = None
    if args.qa is not None:
        args.qa = args.qa.expanduser().resolve()
        qa_json = args.qa.with_suffix(".json")
        if qa_json == args.qa:
            qa_json = args.qa.with_name(args.qa.stem + "_data.json")
    protected_roots = {ROOT}
    if args.manuscript_dir is not None:
        protected_roots.add(args.manuscript_dir)
    destinations = [args.output] + ([] if args.qa is None else [args.qa, qa_json])
    for destination in destinations:
        for protected in [base / name for base in protected_roots
                          for name in ("results", "reference_figures")]:
            protected = protected.resolve()
            if destination == protected or protected in destination.parents:
                parser.error("Reference archives are read-only inputs; choose a different output path.")
    provenance = validate_sources(args.manuscript_dir)
    out = args.output / "algorithm_schematic"
    fig, ax = compose_figure()
    draw_roles(ax)
    layout = check_layout(fig, ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Native 119 mm width is required by the actual manuscript class. Disable
    # automatic tight cropping locally to preserve physical size and font sizes.
    # The baseline above remains verbatim for traceability.
    with mpl.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=RASTER_DPI,
                        bbox_inches=None, pad_inches=0, facecolor=WHITE)
    plt.close(fig)
    embedded_fonts = embed_svg_fonts(out.with_suffix(".svg"))
    files = []
    for ext in ("pdf", "svg", "png"):
        source = out.with_suffix(f".{ext}")
        alias = out.parent / f"Fig1.{ext}"
        shutil.copyfile(source, alias)
        files.extend([str(source), str(alias)])
    caption = (
        "Two-way architecture and messages in an illustrative update round. "
        "Graph clients retain local matrices A_ij; loss roles align with sample "
        "groups and regularizer roles with feature groups. Generic displayed "
        "indices are 1,i,M and 1,j,N. Blue roles with filled dots are active; "
        "grey roles retain their states. Solid arrows summarize current "
        "aggregates sent to active roles; dashed arrows summarize their "
        "increments returned to the coordinator. Loss roles receive t_i^k, "
        "regularizer roles receive a_j^k, and graph clients receive both. "
        "The group boundary is logical and implies neither matrix transfer nor "
        "replies from inactive roles. Logical roles may share a physical node. "
        "Algorithm 1 specifies the common snapshot, commits, and full audits."
    )
    qa_data = {
        "figure": "Two-way architecture and update-round messages",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "previous_dimensions_mm": [119.0, 52.0],
        "raster_dpi": RASTER_DPI,
        "provenance": provenance,
        "layout": layout,
        "svg_embedded_fonts": embedded_fonts,
        "pdf_font_embedding": "TrueType via pdf.fonttype=42; verify with pdffonts",
        "message_mapping": {"solid_blue_right": "Current aggregates to active roles",
                            "dashed_black_left": "Increments from active roles"},
        "proposed_caption": caption,
        "visual_verification": "Pending native-size inspection of exported PDF/PNG",
        "statistics": "Not applicable: exact protocol schematic, no numerical observations",
        "files": files,
    }
    report = (
        "Two-way architecture schematic QA\n"
        "=================================\n"
        f"Manuscript validation: {provenance['validation']}\n"
        f"Source: {provenance['source'] or 'See scientific provenance in script docstring'}\n"
        f"Source SHA-256 at rendering: {provenance['sha256'] or 'Not requested'}\n"
        "Scientific contract: the coordinator exchanges current aggregates and "
        "increments with active loss, graph, and regularizer roles. Graph clients "
        "hold local A_ij blocks in the sample-by-feature partition.\n"
        "One architecture diagram: generic 3-by-3 graph grid; three row loss "
        "roles; three column regularizer roles; one coordinator; two shared "
        "message arrows. Ten prose labels contain fourteen English words.\n"
        "Partition semantics: rows 1,i,M correspond to sample groups and "
        "columns 1,j,N to feature groups. Losses l_i align by row and "
        "regularizers r_j align by column. The displayed nodes illustrate "
        "generic indices, not a fixed 3-by-3 deployment.\n"
        "Activation: the first two displayed rows and columns are selected; "
        "eight local roles are active and seven are inactive. Blue outlines "
        "and filled corner dots redundantly mark active roles; inactive roles "
        "remain in grey. This illustrates a valid rectangular activation, "
        "without making that sampler mandatory.\n"
        "Shared arrow mapping: solid blue rightward = current aggregates to "
        "active roles; dashed black leftward = increments from active roles. "
        "Both terminate at the logical-role group boundary. They are not "
        "matrix transfers or a requirement for inactive roles to reply.\n"
        "The image contains no equations, audit branch, narrative sentences, "
        "or duplicate per-role message labels. Exact messages, common snapshot, "
        "commits, and audits remain in the caption and Algorithm 1.\n"
        f"Native figure dimensions: {WIDTH_MM} x {HEIGHT_MM} mm.\n"
        "Target: JOTA svjour3 smallextended, textwidth 11.9 cm in svglov3.clo.\n"
        f"Typography: {BODY_FONT}; prose labels 9-10 pt; math labels 13 pt "
        "with 9.1 pt subscripts. All actual text glyphs meet the 9 pt floor.\n"
        "Exports: vector PDF with embedded TrueType fonts; editable-text SVG "
        "with embedded WOFF regular/bold/italic subsets; PNG at "
        f"{RASTER_DPI} dpi. Fig1 aliases are byte-identical.\n"
        "Predicted issues handled: row/column role alignment communicates "
        "partition structure; one pair of shared arrows avoids a crossing "
        "network; activation dots remain readable in grayscale; native sizes "
        "and 9.1 pt subscripts preserve print legibility.\n"
        "Asset scan: no exact algorithm architecture asset; Sankey asset and "
        "companion image inspected, but numerical flow ribbons are incompatible. "
        "Cross-type styling only; original vector geometry.\n"
        "Pass 0 AP-0..AP-7: verbatim style baselines; restrained custom palette; "
        "no axes/grid; direct labels; vector exports; explicit sans-serif font. "
        "Small-n quantitative checks are not applicable.\n"
        "Pass 1 CL-1..CL-7: drawn prose >=9 pt, subscripts 9.1 pt; native 119 mm "
        "manuscript width overrides generic CNS presets; 1200 dpi raster; "
        "fonttype 42 and embedded SVG fonts; vector/raster outputs complete.\n"
        "Pass 2 VI-1..VI-7: all local roles, row/column meaning, partial "
        "activation, and both message directions preserved; direction, line "
        "style, and dots are independent of color; no numerical axes or "
        "statistical claims. Revision case 10 is addressed by one visual "
        "language, font family, and vector backend.\n"
        "Pass 3 preliminary: every text bounding box lies inside the page; "
        "no text boxes overlap; label, word, role, and arrow counts validated. "
        "Visual verification is recorded after inspection.\n"
        "Scientific scope: spatial grouping shows local-role associations, "
        "not a physical deployment. No raw-data transfer, privacy guarantee, "
        "mandatory sampling pattern, or new update rule is claimed.\n"
        "Statistics and reproducibility: no numerical observations, replicates, "
        "summary statistics, uncertainty intervals, tests, or generated data. "
        "Roles and directions derive from the problem formulation, Algorithm 1, "
        "and cache/update equations in sections/theory.tex.\n"
        f"Caption guidance: {caption}\n"
    )
    if args.qa is not None:
        args.qa.parent.mkdir(parents=True, exist_ok=True)
        args.qa.write_text(report, encoding="utf-8")
        qa_json.write_text(json.dumps(qa_data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": files,
                      "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
                      "raster_dpi": RASTER_DPI,
                      "layout": layout,
                      "provenance": provenance,
                      "qa": str(args.qa) if args.qa is not None else None,
                      "qa_json": str(qa_json) if qa_json is not None else None}, indent=2))


if __name__ == "__main__":
    main()
