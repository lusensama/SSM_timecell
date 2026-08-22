"""Shared plotting style for the response-letter figures.

Every colour here is lifted from the original repo -- see the block comment
below for the file and line each one comes from.  Where the repo has no
convention (the LSTM arm, the lap-counting trained/untrained pair) the dataviz
categorical slots are used, chosen so the resulting set still clears the
colourblind gates beside the repo colours:

    tab:red + tab:blue                    all-pairs CVD dE 21.1, normal 31.7
    + aqua (LSTM H=50)                    all-pairs CVD dE 11.4, normal 21.2
    + yellow (LSTM H=32)                  all-pairs CVD dE  9.1, normal 21.2

Aqua and yellow sit below 3:1 on the light surface, so every panel that uses
them carries visible direct value labels (the relief rule).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODE_COLOR = {
    "hippo": "#d62728",
    "rand_complex": "#1f77b4",
}

CB = {
    "hippo": "#0072B2", "rand_complex": "#D55E00", "spectrum_matched": "#009E73",
    "freq_matched": "#CC79A7", "perturbed_hippo": "#56B4E9",
    "alt_basis": "#E69F00", "real_diagonal": "#666666",
    "s4d_lin": "#F0E442", "s4d_inv": "#000000",
}
CB_SHADES = ["#0072B2", "#D55E00", "#666666"]

SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_3 = "#1baf7a"

C = {
    "blue":    SERIES_1,
    "orange":  SERIES_2,
    "aqua":    SERIES_3,
    "yellow":  "#eda100",
    "magenta": "#e87ba4",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
}

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

HEATMAP_CMAP = "jet"
HEATMAP_INTERP = "none"

SERIES = {
    "hippo": MODE_COLOR["hippo"],
    "HiPPO-LegS": MODE_COLOR["hippo"],
    "SSM HiPPO": MODE_COLOR["hippo"],
    "rand_complex": MODE_COLOR["rand_complex"],
    "SSM random": MODE_COLOR["rand_complex"],
    "LSTM H=50": C["aqua"],
    "LSTM H=32": C["yellow"],
    "trained": SERIES_1,
    "untrained": SERIES_2,
    "zero-shot transfer": SERIES_2,
    "readout-only recovery": SERIES_1,
}

CLASS_COLOR = {
    "learns + cascade": SERIES_1,
    "learns, no cascade": SERIES_2,
    "cascade, cannot learn": SERIES_3,
}

ORDINAL_2 = ["#8f1f1e", "#e0605f"]

SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("respseq", SEQ_STEPS)

DIV = LinearSegmentedColormap.from_list(
    "respdiv", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                "#f3a3a2", "#e34948", "#8f1f1e"])

BARE = os.environ.get("RESP_BARE", "1") != "0"

NOTEXT = os.environ.get("RESP_NOTEXT", "1") != "0"
BARE = BARE or NOTEXT

DPI = int(os.environ.get("RESP_DPI", "600"))

def strip_text(fig, keep_in_axes=False):
    """Remove every Text artist from a finished figure.

    keep_in_axes=True keeps the text drawn INSIDE the data area -- ax.text() and
    annotate() -- and removes only the chrome around it: titles, axis labels, tick
    labels, legends, figure text and the colorbar's own labels.  That is the
    confusion matrix's case: its cell values are the figure, everything else
    around them is a label a downstream text layer can replace.

    Done at save time rather than at each call site: the eight plot scripts set
    text through a dozen different APIs (set_title/set_xlabel/set_xticklabels/
    text/annotate/legend/suptitle/fig.text/Colorbar.set_label), and a post-pass
    over the artist tree catches all of them, including the ones inside helpers
    like chance_line() and fan_labels() and the colorbar axes matplotlib creates
    for itself.

    Note that removing an annotate() removes its leader line with it, which is
    what we want: a leader pointing at nothing is worse than no leader.
    """
    for ax in list(fig.axes):
        for loc in ("left", "center", "right"):
            ax.set_title("", loc=loc)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
        for axis in (ax.xaxis, ax.yaxis):
            axis.offsetText.set_visible(False)
        if not keep_in_axes:
            for t in list(ax.texts):
                t.remove()
        for tbl in list(getattr(ax, "tables", [])):
            tbl.remove()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for leg in list(fig.legends):
        leg.remove()
    for t in list(fig.texts):
        t.remove()
    return fig

def note(ax, x, y, s, **kw):
    """Explanatory prose inside a panel.  Suppressed in BARE mode."""
    if BARE:
        return None
    return ax.text(x, y, s, **kw)

def callout(ax, *args, **kw):
    """An annotate() with a leader line, used for prose.  Suppressed in BARE."""
    if BARE:
        return None
    return ax.annotate(*args, **kw)

def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "text.color": INK,
        "figure.dpi": 110,
    })

def grid(ax, axis="y"):
    """Hairline recessive grid, solid, behind the marks."""
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)

def chance_line(ax, y, label="chance", color=INK_MUTED, x=0.995, ha="right"):
    """A solid hairline reference rule, labelled in ink rather than in colour."""
    ax.axhline(y, color=color, lw=0.9, zorder=1)
    ax.text(x, y, f" {label} ", transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", fontsize=7, color=INK_2)

def title(fig, main, sub=None):
    """Title block placed ABOVE the figure canvas.  Suppressed in BARE mode.

    Positions are computed in inches and converted to figure fractions, so the
    block never lands on a panel title however tall the figure is; bbox_inches
    ="tight" pulls it back into the saved image.
    """
    if BARE:
        return
    h = fig.get_figheight()
    nl = (sub.count("\n") + 1) if sub else 0
    sub_top = 1.0 + 0.14 / h
    main_y = sub_top + (nl * 0.155 + 0.04) / h if sub else 1.0 + 0.10 / h
    fig.suptitle(main, y=main_y, va="bottom", fontsize=11.5,
                 fontweight="bold", color=INK)
    if sub:
        fig.text(0.5, sub_top, sub, ha="center", va="bottom",
                 fontsize=8.5, color=INK_2, linespacing=1.5)

def stamp(fig, text, y=None):
    """Provenance line: which response section and which CSV a panel came from.

    Drawn below the canvas so it cannot collide with rotated tick labels; pass a
    more negative `y` when the x labels are long.  Suppressed in BARE mode.
    """
    if BARE:
        return
    if y is None:
        y = -0.16 / fig.get_figheight()
    fig.text(0.0, y, text, ha="left", va="top", fontsize=6.2, color=INK_MUTED)

def fan_labels(ax, points, x_anchor, side="right", min_gap_frac=0.052,
               fontsize=8, leader=True):
    """Direct-label crowded scatter points without overlaps.

    `points` is [(x, y, text, colour), ...].  `x_anchor` is an AXES FRACTION
    (0..1), so the helper works on log axes as well as linear.  Labels are
    stacked at that column in y order, pushed apart to at least `min_gap_frac` of
    the y-range, and joined to their marker by a hairline leader.  This keeps
    direct labelling -- so identity is never colour-alone -- where a naive offset
    would pile six labels on top of each other.

    Call it after the axis limits are final; the anchor is resolved against them.
    """
    if not points:
        return
    inv = ax.transData.inverted()
    x_anchor = float(inv.transform(ax.transAxes.transform((x_anchor, 0.0)))[0])
    pad = float(inv.transform(ax.transAxes.transform((0.012, 0.0)))[0]) - \
        float(inv.transform(ax.transAxes.transform((0.0, 0.0)))[0])
    y0, y1 = ax.get_ylim()
    gap = abs(y1 - y0) * min_gap_frac
    pts = sorted(points, key=lambda p: p[1])
    ys = [p[1] for p in pts]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    hi = max(y0, y1) - gap * 0.5
    shift = max(0.0, ys[-1] - hi)
    ys = [y - shift for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        if ys[i + 1] - ys[i] < gap:
            ys[i] = ys[i + 1] - gap
    ha = "left" if side == "right" else "right"
    stop = x_anchor - pad if side == "right" else x_anchor + pad
    for (x, y, text, col), ty in zip(pts, ys):
        if leader:
            ax.plot([x, stop], [y, ty], color=GRID, lw=0.7, zorder=2,
                    solid_capstyle="round")
        ax.text(x_anchor, ty, text, fontsize=fontsize, color=INK,
                ha=ha, va="center", zorder=5)

def save(fig, path, pdf=True):
    if BARE:
        for ax in fig.axes:
            for loc in ("left", "center", "right"):
                ax.set_title("", loc=loc)
    if NOTEXT:
        strip_text(fig)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    import os
    print(f"  wrote {os.path.basename(path)}"
          + (f" (+ {os.path.basename(path).replace('.png', '.pdf')})" if pdf else ""))

def pstar(p):
    """Compact significance annotation."""
    if p is None or p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

