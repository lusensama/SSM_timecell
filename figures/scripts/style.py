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
    if BARE:
        return None
    return ax.text(x, y, s, **kw)

def callout(ax, *args, **kw):
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
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)

def chance_line(ax, y, label="chance", color=INK_MUTED, x=0.995, ha="right"):
    ax.axhline(y, color=color, lw=0.9, zorder=1)
    ax.text(x, y, f" {label} ", transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", fontsize=7, color=INK_2)

def title(fig, main, sub=None):
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
    if BARE:
        return
    if y is None:
        y = -0.16 / fig.get_figheight()
    fig.text(0.0, y, text, ha="left", va="top", fontsize=6.2, color=INK_MUTED)

def fan_labels(ax, points, x_anchor, side="right", min_gap_frac=0.052,
               fontsize=8, leader=True):
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
    if p is None or p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

