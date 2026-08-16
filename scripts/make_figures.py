"""Generate publication figures for the GMC manuscript.

Reads the per-method summary CSVs from Results/summaries/ (data-driven, matching
the manuscript tables exactly) and writes three figures to papers/figures/:

  fig1_framework.{pdf,svg,png}  — GMC model overview
                                  (similarity fusion → cold-start fill = solver
                                   initialization → ONE rank prior in two
                                   geometries: matrix (fused block) + tensor
                                   (per-modality slices) → one estimator → M̂,
                                   no post-hoc filter; GMC-E = score-level fusion
                                   with baselines + optional bilateral filter)
  fig2_main.{pdf,svg,png}       — AUPR comparison, GMC / GMC-E vs published
                                  baselines on all four datasets (2x2 panels)
  fig3_ablation.{pdf,svg,png}   — unified-config ablation (2x2, one panel per
                                  dataset, fresh validation folds)

SGLP is deliberately excluded: it is the predecessor of GMC (replaced, not a
comparison baseline). The ablation bars shown are GMC's three architectural
components only (cold-start fill / block view / tensor view); the rank-vs-raw
fusion check is neutral and reported in prose, not plotted.

Run:  python scripts/make_figures.py
"""
import os
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# LaTeX-style math (Computer Modern via mathtext — vector outlines, no local
# TeX distribution needed). Keeps math glyphs consistent across all figures.
matplotlib.rcParams["mathtext.fontset"] = "cm"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "Results", "summaries")
FIG = os.path.join(ROOT, "papers", "figures")
os.makedirs(FIG, exist_ok=True)

# ---- colour palette (validated: blue/red pair passes CVD + normal-vision
# floors; grey is the deliberately-recessive neutral baseline group) ---------
C_BASE = "#7f8c8d"    # published baselines / neutral
C_GMC = "#2a78d6"     # GMC (ours)
C_GMC_E = "#e34948"   # GMC-E (ours)
C_BLOCK = "#3b82f6"   # block completion view (blue)
C_TENSOR = "#8b5cf6"  # tensor completion view (violet)
C_FILL = "#14b8a6"    # cold-start KNN fill
C_INPUT = "#f59e0b"   # masked input M
C_DIS = "#16a34a"     # disease similarity (green) — red is reserved for GMC-E
DARK = "#1e293b"

DATASETS = ["Fdataset", "Cdataset", "CTDdataset2023", "Ydataset"]

# internal summary tag -> display name (published baselines under our folds)
BASELINES = [
    ("baseline_BNNR", "BNNR"),
    ("baseline_OMC", "OMC"),
    ("baseline_ITRPCA", "ITRPCA"),
    ("baseline_DNMFDDA", "DNMFDDA"),
    ("baseline_HGIMC", "HGIMC"),
    ("baseline_MSBMF", "MSBMF"),
    ("baseline_DDASKF", "DDA-SKF"),
    ("baseline_NMF-DR", "NMF-DR"),
    ("multiGMF_full", "multiGMF"),
]
# one unified GMC config on all four datasets (2026-08-11 adoption); the former
# per-dataset anchors (gmc_cs_filt37 / gmc_graph_trrank_a07 / gmc_trrank) are
# superseded — see memory [[unified-gmc-config-adopted]].
GMC_TAG = {ds: "gmc_unified" for ds in DATASETS}


def save_three(fig, base):
    """Save a figure as .pdf, .svg and .png (300 dpi) from the same figure."""
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    print(f"  {os.path.basename(base)}.{{pdf,svg,png}}")


def load_aupr(dataset, tag):
    """(AUPR, AUPR_std) for a dataset + internal tag, or None if absent."""
    p = os.path.join(RES, f"{dataset}_{tag}_summary.csv")
    if not os.path.exists(p):
        return None
    row = pd.read_csv(p).iloc[0]
    return float(row["AUPR"]), float(row.get("AUPR_std", 0.0))


# ============================================================================
# Figure 1 — GMC pipeline framework
# ============================================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    ax.axis("off")
    ax.set_xlim(-0.3, 16.6)
    ax.set_ylim(-0.9, 7.4)

    def rounded_box(cx, cy, w, h, face, edge, txt="", fs=10, lw=1.0,
                    color="black", weight="normal", sub=None, sub_fs=7.5,
                    sub_color=None):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=lw,
                                    zorder=2))
        if txt:
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                    color=color, fontweight=weight, zorder=4)
        if sub:
            ax.text(cx, cy - h / 2 - 0.16, sub, ha="center", va="top",
                    fontsize=sub_fs, color=sub_color or color, zorder=4)

    def small_matrix(cx, cy, size=0.62, face="#3b82f6", n=4, mask_frac=0.0,
                     seed=0, label="", diag_strength=True, noise=0.30,
                     label_dy=0.16, zero_row=None, zero_col=None):
        rng = random.Random(seed)
        base = mcolors.to_rgb(face)
        cell = size / n
        margin = (size - n * cell) / 2.0
        for i in range(n):
            for j in range(n):
                x0 = cx - size / 2 + margin + j * cell
                y0 = cy + size / 2 - margin - (i + 1) * cell
                if zero_row == i or zero_col == j:
                    # all-zero (novel) entity: no known entries at all
                    ax.add_patch(plt.Rectangle((x0, y0), cell * 0.92, cell * 0.92,
                                 facecolor="#ffffff", edgecolor="#cbd5e1",
                                 linewidth=0.5, zorder=2))
                    continue
                if mask_frac > 0 and rng.random() < mask_frac:
                    ax.add_patch(plt.Rectangle((x0, y0), cell * 0.92, cell * 0.92,
                                 facecolor="#f1f5f9", edgecolor="#94a3b8",
                                 linewidth=0.4, linestyle=":", zorder=2))
                    continue
                if diag_strength:
                    d = abs(i - j) / max(1, n - 1)
                    v = max(0.12, 1.0 - 0.70 * d + 0.12 * rng.random())
                else:
                    v = rng.uniform(0.25, 1.0)
                v += noise * (rng.random() - 0.5)
                v = max(0.08, min(1.0, v))
                fc = tuple(c * v + 0.86 * (1 - v) for c in base)
                ax.add_patch(plt.Rectangle((x0, y0), cell * 0.92, cell * 0.92,
                             facecolor=fc, edgecolor="#e2e8f0",
                             linewidth=0.3, zorder=2))
        ax.add_patch(plt.Rectangle((cx - size / 2, cy - size / 2), size, size,
                     fill=False, edgecolor=face, linewidth=1.0, zorder=3))
        if label:
            ax.text(cx, cy - size / 2 - label_dy, label, ha="center", va="top",
                    fontsize=8.5, color=face, fontweight="bold", zorder=4)

    def network_graph(cx, cy, n=6, marker='o', color="#3b82f6", size=1.0,
                      seed=0, label="", layout="ring"):
        rng = np.random.default_rng(seed)
        r = size / 2.0 - 0.12
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if layout == "ring":
            rr = np.full(n, r)
        else:
            rr = r * (0.82 + 0.18 * rng.random(n))
            angles = angles + 0.08 * (rng.random(n) - 0.5)
        xs = cx + rr * np.cos(angles)
        ys = cy + rr * np.sin(angles)
        edges = set()
        for i in range(n):
            edges.add((min(i, (i + 1) % n), max(i, (i + 1) % n)))
            edges.add((min(i, (i + 2) % n), max(i, (i + 2) % n)))
        if n >= 6:
            edges.add((0, n // 2))
        for i, j in edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color, lw=0.9,
                    alpha=0.5, zorder=1)
        ax.scatter(xs, ys, s=110, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=3)
        if label:
            ax.text(cx, cy - size / 2 - 0.14, label, ha="center", va="top",
                    fontsize=8, color=color, fontweight="bold", zorder=4)

    def block_matrix(cx, cy, size=0.78, labels=("Wdd", "F", "F$^T$", "Wrr")):
        """2x2 block [[Wdd, F], [F^T, Wrr]] used by the block completion."""
        half = size / 2.0
        # Wdd = disease sim, Wrr = drug sim (colour semantics match panel (a))
        quads = [
            (C_DIS, cx - half, cy),
            (C_FILL, cx, cy),
            (C_FILL, cx - half, cy - half),
            (C_BLOCK, cx, cy - half),
        ]
        pos = [(cx - half * 0.5, cy + half * 0.5),
               (cx + half * 0.5, cy + half * 0.5),
               (cx - half * 0.5, cy - half * 0.5),
               (cx + half * 0.5, cy - half * 0.5)]
        for (color, x0, y0) in quads:
            ax.add_patch(plt.Rectangle((x0, y0), half, half, facecolor=color,
                         edgecolor="white", linewidth=0.5, zorder=2))
        ax.add_patch(plt.Rectangle((cx - half, cy - half), size, size,
                     fill=False, edgecolor=DARK, linewidth=1.0, zorder=3))
        for (x, y), txt in zip(pos, labels):
            ax.text(x, y, txt, ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=4)

    def tensor_icon(cx, cy, w=1.0, h=0.62, n=3, face=C_TENSOR):
        """Stacked-slice icon for the per-similarity tensor completion."""
        for k in range(n):
            fc = tuple(c + (1 - c) * 0.35 * (k / n) for c in mcolors.to_rgb(face))
            ax.add_patch(plt.Rectangle(
                (cx - w / 2 + k * 0.15, cy - h / 2 - k * 0.12), w, h,
                facecolor=fc, edgecolor="#6d28d9", linewidth=0.6, zorder=2 + k))

    def arrow(x1, y1, x2, y2, color="#334155", lw=1.3, ms=13, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=ms, color=color,
                                     linewidth=lw, linestyle=ls, zorder=5))

    def group_box(x, y, w, h, title="", caption="", fs=11, cap_fs=9,
                  color="#64748b"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor="none", edgecolor=color,
                                    linestyle="--", linewidth=1.0, zorder=0))
        if title:
            ax.text(x + w / 2, y + h + 0.14, title, ha="center", va="bottom",
                    fontsize=fs, fontweight="bold", color=color, zorder=4)
        if caption:
            ax.text(x + w / 2, y - 0.18, caption, ha="center", va="top",
                    fontsize=cap_fs, color=color, zorder=4)

    # ------------------------------------------------------------------ title
    ax.text(8.15, 7.12, "GMC: Graph Multi-view Completion",
            ha="center", fontsize=15, fontweight="bold", color=DARK)

    # ------------------------------------------------------------------ (a)
    group_box(0.25, 1.15, 4.2, 5.35,
              title="(a) Similarity fusion",
              caption="mean-fuse 5+2 views → W")
    # drug row
    network_graph(1.6, 5.6, n=7, marker='^', color=C_BLOCK, seed=11, size=1.1,
                  label="5 drug sims")
    arrow(2.22, 5.6, 3.03, 5.6, color=C_BLOCK)
    ax.text(2.62, 5.6, r"$\Sigma$", ha="center", va="center", fontsize=11,
            color="#64748b", zorder=4)
    ax.text(2.62, 5.24, "fuse", ha="center", va="top", fontsize=6.4,
            color="#64748b", zorder=4)
    small_matrix(3.4, 5.6, size=0.66, face=C_BLOCK, n=4, seed=7,
                 label="W$_{rr}$", diag_strength=True, noise=0.25)
    # disease row
    network_graph(1.6, 2.8, n=5, marker='s', color=C_DIS, seed=29, size=1.1,
                  label="2 disease sims", layout="cluster")
    arrow(2.22, 2.8, 3.03, 2.8, color=C_DIS)
    ax.text(2.62, 2.8, r"$\Sigma$", ha="center", va="center", fontsize=11,
            color="#64748b", zorder=4)
    ax.text(2.62, 2.44, "fuse", ha="center", va="top", fontsize=6.4,
            color="#64748b", zorder=4)
    small_matrix(3.4, 2.8, size=0.66, face=C_DIS, n=4, seed=41,
                 label="W$_{dd}$", diag_strength=False, noise=0.35)

    # ------------------------------------------------------------------ (b)
    group_box(4.95, 1.15, 5.0, 5.35,
              title="(b) Cold-start fill (initialization)",
              caption="seed the completion solver in all-zero rows/cols only")
    # M (masked) — one all-zero row + one all-zero column drawn EMPTY and
    # dashed-outlined: these novel entities are the ONLY targets of the KNN fill
    mz_size, mz_cx, mz_cy, mz_n = 0.7, 5.7, 4.3, 5
    zero_row, zero_col = 1, 3
    small_matrix(mz_cx, mz_cy, size=mz_size, face=C_INPUT, n=mz_n, mask_frac=0.35,
                 seed=2, label="M (masked)", zero_row=zero_row, zero_col=zero_col)
    mcell = mz_size / mz_n
    # all-zero row (dashed outline + label)
    ry = mz_cy + mz_size / 2 - (zero_row + 1) * mcell
    ax.add_patch(plt.Rectangle((mz_cx - mz_size / 2, ry), mz_size, mcell,
                 fill=False, edgecolor="#0f172a", linewidth=1.3,
                 linestyle="--", zorder=4))
    ax.text(mz_cx - mz_size / 2 - 0.13, ry + mcell / 2, "all-zero row",
            ha="right", va="center", fontsize=6.6, fontweight="bold",
            color="#0f172a", zorder=4)
    # all-zero column (dashed outline + label)
    cx0 = mz_cx - mz_size / 2 + zero_col * mcell
    ax.add_patch(plt.Rectangle((cx0, mz_cy - mz_size / 2), mcell, mz_size,
                 fill=False, edgecolor="#0f172a", linewidth=1.3,
                 linestyle="--", zorder=4))
    ax.text(cx0 + mcell / 2, mz_cy + mz_size / 2 + 0.13, "all-zero col",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold",
            color="#0f172a", zorder=4)
    rounded_box(6.9, 4.3, 1.0, 0.9, "#fef3c7", C_INPUT, "KNN fill", fs=9,
                color=C_INPUT, sub="k=10 (unified)", sub_color="#b45309")
    # cold-start gate (diamond): only all-zero rows/cols pass through
    gx, gy, gs = 8.05, 4.3, 0.52
    ax.add_patch(Polygon(
        [(gx, gy + gs), (gx + gs * 0.62, gy), (gx, gy - gs), (gx - gs * 0.62, gy)],
        facecolor="#fff7ed", edgecolor="#b45309", linewidth=1.4, zorder=3))
    ax.text(gx, gy + 0.18, "cold-start", ha="center", va="center", fontsize=6.8,
            color="#b45309", fontweight="bold", zorder=4)
    ax.text(gx, gy - 0.12, "all-zero\nrow/col?", ha="center", va="center",
            fontsize=6.0, color="#b45309", zorder=4)
    ax.text(gx, 3.58, "only all-zero rows/cols get KNN fill",
            ha="center", va="top", fontsize=7.8, fontweight="bold",
            color="#b45309", zorder=4)
    small_matrix(9.35, 4.3, size=0.7, face=C_FILL, n=5, seed=53)
    ax.text(9.35, 4.3 - 0.45, "F", ha="center", va="top", fontsize=9,
            color=C_FILL, fontweight="bold", zorder=4)
    arrow(6.05, 4.3, 6.36, 4.3, color=C_INPUT)
    arrow(7.4, 4.3, 7.72, 4.3, color=C_INPUT)
    arrow(8.5, 4.3, 9.0, 4.3, color=C_FILL)
    # Wrr / Wdd feed the KNN propagation
    ax.annotate("", xy=(6.9, 4.78), xytext=(4.3, 4.3),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.1,
                                connectionstyle="arc3,rad=0.28", linestyle="--"),
                zorder=1)
    ax.text(5.6, 5.0, "W$_{rr}$/W$_{dd}$ graphs", ha="center", va="bottom",
            fontsize=6.8, color="#64748b", zorder=4)

    # ------------------------------------------------------------------ (c)
    group_box(10.55, 1.15, 5.85, 5.35,
              title="(c) Completion: one prior, two geometries",
              caption="two geometries of one rank prior → one estimator (no filter)")
    # shared stem: ONE low-rank completion prior — both geometries solve the same
    # masked completion from the same similarity data, fed the same fill
    rounded_box(13.0, 6.15, 2.2, 0.7, "#f8fafc", DARK,
                "one low-rank completion prior", fs=8.2, color=DARK,
                weight="bold")
    # matrix geometry — nuclear norm over the fused joint block (BNNR/SVT)
    rounded_box(12.5, 4.1, 2.2, 2.3, "#eff6ff", C_BLOCK, "", fs=9,
                color=C_BLOCK)
    block_matrix(12.5, 4.9, size=0.56)
    ax.text(12.5, 3.95, "matrix geometry", ha="center", fontsize=7.8,
            color="#1d4ed8", fontweight="bold", zorder=4)
    ax.text(12.5, 3.68, "nuclear norm on the fused joint block", ha="center",
            fontsize=6.2, color=C_BLOCK, zorder=4)
    ax.text(12.5, 3.42, "rank-capped ADMM", ha="center", fontsize=6.2,
            color="#1d4ed8", zorder=4)
    # tensor geometry — nuclear norm over the per-modality slices (ITRPCA)
    rounded_box(15.3, 4.1, 2.2, 2.3, "#f5f3ff", C_TENSOR, "", fs=9,
                color=C_TENSOR)
    tensor_icon(15.3, 4.9, w=1.0, h=0.5)
    ax.text(15.3, 3.95, "tensor geometry", ha="center", fontsize=7.8,
            color="#6d28d9", fontweight="bold", zorder=4)
    ax.text(15.3, 3.68, "nuclear norm on the per-modality slices", ha="center",
            fontsize=6.2, color=C_TENSOR, zorder=4)
    ax.text(15.3, 3.42, "FFT-domain t-SVD (5+2)", ha="center", fontsize=6.2,
            color="#6d28d9", zorder=4)
    # convergence into ONE estimator (rank-free scale, fixed equal weights)
    rounded_box(13.48, 2.0, 2.7, 1.0, "#ecfdf5", "#0f766e",
                "one estimator", fs=8.5, color="#0f766e", weight="bold",
                sub="rnorm01, w = 0.5/0.5", sub_fs=6.4, sub_color="#0f766e")
    # output (unified config applies no post-hoc filter to the estimator)
    small_matrix(16.1, 2.0, size=0.72, face="#d97706", n=5, seed=71,
                 diag_strength=False, noise=0.2)
    ax.text(16.1, 1.56, r"$\hat{M}$", ha="center", va="top", fontsize=11,
            color="#d97706", fontweight="bold", zorder=4)

    # arrows inside (c): one completion branches into two geometries of the same
    # similarity data, then converges into a single estimator
    arrow(9.72, 4.3, 12.0, 5.9, color=C_FILL, lw=1.1)      # fill (initialization) → shared prior
    arrow(12.9, 5.8, 12.5, 5.25, color="#64748b", lw=1.0)  # prior → matrix geometry
    arrow(13.3, 5.8, 15.3, 5.25, color="#64748b", lw=1.0)  # prior → tensor geometry
    arrow(12.5, 2.95, 13.1, 2.5, color=C_BLOCK, lw=1.1)    # matrix readout → estimator
    arrow(15.3, 2.95, 13.85, 2.5, color=C_TENSOR, lw=1.1)  # tensor readout → estimator
    arrow(14.83, 2.0, 15.74, 2.0, color="#0f766e", lw=1.3) # estimator → M̂
    ax.text(15.2, 2.28, "no post-hoc filter", ha="center", fontsize=5.8,
            color="#64748b", zorder=4)

    # ------------------------------------------------------------------ GMC-E
    # score-level combination below the pipeline; the OPTIONAL bilateral filter
    # lives in the GMC-E branch (GMC itself uses no post-hoc filter)
    rounded_box(12.5, 0.15, 2.7, 1.1, "#fdf2f8", C_GMC_E, "", fs=9,
                color=C_GMC_E)
    ax.text(12.5, 0.3, "GMC-E: score-level average", ha="center", fontsize=8.4,
            color="#be185d", fontweight="bold", zorder=4)
    ax.text(12.5, -0.26, "GMC + complementary baselines", ha="center",
            fontsize=6.8, color="#be185d", zorder=4)
    rounded_box(15.1, 0.15, 1.6, 1.05, "#ffedd5", "#d97706",
                "bilateral filter", fs=7.6, color="#b45309",
                sub="optional (k=5-sparsified)", sub_fs=6.4, sub_color="#b45309")
    ax.text(16.3, 0.15, r"$\hat{M}_E$", ha="center", va="center", fontsize=10,
            color="#d97706", fontweight="bold", zorder=4)
    ax.add_patch(FancyArrowPatch((16.1, 1.62), (13.7, 0.72), arrowstyle="-|>",
                                 mutation_scale=13, color="#d97706",
                                 linewidth=1.1, linestyle="-", zorder=5,
                                 connectionstyle="arc3,rad=0.12"))
    arrow(13.85, 0.3, 14.28, 0.3, color="#b45309", lw=1.0)
    arrow(15.9, 0.3, 16.08, 0.3, color="#b45309", lw=1.0)

    # known-entry restoration note
    ax.text(8.0, -0.7,
            r"train entries restored verbatim:  $M$ = where($A \neq 0$, $A$, $\hat{M}$)",
            ha="center", va="top", fontsize=7, color="#64748b", zorder=4)

    # in-figure colour legend (red is reserved for GMC-E = upper reference,
    # matching the red in fig2/fig3)
    lx, ly = 14.15, 7.0
    ax.add_patch(FancyBboxPatch((lx, ly - 1.58), 2.45, 1.58,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor="white", edgecolor="#cbd5e1",
                                linewidth=0.8, zorder=6))
    ax.text(lx + 0.12, ly - 0.12, "Colour legend", ha="left", va="center",
            fontsize=6.6, fontweight="bold", color=DARK, zorder=7)
    legend_items = [
        (C_BLOCK, "drug sims W$_{rr}$"),
        (C_DIS, "disease sims W$_{dd}$"),
        (C_INPUT, "masked input M"),
        (C_FILL, "cold-start fill F"),
        (C_TENSOR, "tensor geometry"),
        (C_GMC_E, "GMC-E (upper ref.)"),
        ("#d97706", "output $\\hat{M}$"),
    ]
    for k, (c, t) in enumerate(legend_items):
        yr = ly - 0.26 - k * 0.19
        ax.add_patch(plt.Rectangle((lx + 0.12, yr - 0.05), 0.13, 0.13,
                     facecolor=c, edgecolor="none", zorder=7))
        ax.text(lx + 0.34, yr, t, ha="left", va="center", fontsize=6.0,
                color=DARK, zorder=7)

    save_three(fig, os.path.join(FIG, "fig1_framework"))
    plt.close(fig)


# ============================================================================
# Figure 2 — main AUPR comparison (2x2, one panel per dataset)
# ============================================================================
def fig2_main():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4))
    axes = axes.ravel()
    for ax, ds in zip(axes, DATASETS):
        names, vals, errs, cols = [], [], [], []
        for tag, label in BASELINES:
            r = load_aupr(ds, tag)
            if r is None:
                continue
            names.append(label)
            vals.append(r[0])
            errs.append(r[1])
            cols.append(C_BASE)
        gmc_i = gmc_e_i = None
        for tag, label in [(GMC_TAG[ds], "GMC (ours)"), ("ensemble", "GMC-E (ours)")]:
            r = load_aupr(ds, tag)
            names.append(label)
            vals.append(r[0])
            errs.append(r[1])
            cols.append(C_GMC if label.startswith("GMC (") else C_GMC_E)
            if label == "GMC (ours)":
                gmc_i = len(names) - 1
            else:
                gmc_e_i = len(names) - 1
        ax.bar(np.arange(len(names)), vals, yerr=errs, color=cols,
               edgecolor="black", linewidth=0.5, capsize=2,
               error_kw=dict(lw=0.8))
        for i in (gmc_i, gmc_e_i):
            ax.text(i, vals[i] + errs[i] + 0.012, f"{vals[i]:.3f}",
                    ha="center", fontsize=7, fontweight="bold", color=cols[i])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=6.4, rotation=40, ha="right")
        ax.set_ylim(0, 0.92)
        ax.set_title(ds, fontsize=10.5, pad=8)
        ax.set_ylabel("AUPR", fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_BASE, edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_GMC, edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_GMC_E, edgecolor="black", linewidth=0.5),
    ]
    fig.legend(handles, ["Published baselines", "GMC (ours)", "GMC-E (ours)"],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=8.5, frameon=True, edgecolor="#cbd5e1")
    fig.suptitle("AUPR under 10-fold random-entry CVa (mean ± std)",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_three(fig, os.path.join(FIG, "fig2_main"))
    plt.close(fig)


# ============================================================================
# Figure 3 — remove-one-at-a-time ablation (2x2, one panel per dataset; data
# from the fresh validation folds, matching Table 2 + ablation narrative)
# ============================================================================
# Each ablation starts from the FULL unified model (mode 0 = uni_obs_rc400_t50:
# fill=knn + block + tensor + rank fusion) and removes EXACTLY ONE architectural
# component (2026-08-16 refactor from the old add-a-rung ladder):
#   mode A  −fill    = ablate_fill       (fill="none")
#   mode B  −block   = ablate_block      (w_bnnr=0, tensor-only)
#   mode C  −tensor  = uni_obs_rc400_nt  (w_tensor=0, block-only)
# The scale-free rank fusion is NOT an architectural component — it is a numerical
# detail (rank normalization before the fixed 0.5/0.5 weight). Its raw-vs-rank
# check (ablate_fusion) is neutral (|Δ|≤0.0003) and is reported in prose only,
# not as a 4th bar. uni_obs_rc400_nofill (removes BOTH fill and tensor) is a
# superseded control and no longer plotted.
ABLATION_MODES = [
    ("full", "GMC\n(full)", C_GMC),
    ("ablate_fill", "−fill\n(A)", "#8fb3c9"),
    ("ablate_block", "−block\n(B)", "#7f9cb5"),
    ("uni_obs_rc400_nt", "−tensor\n(C)", "#a895c9"),
]


def _fresh_aupr(ds, config):
    """AUPR for a unified-scope config on the fresh validation folds."""
    p = os.path.join(RES, f"{ds}_unified_fresh_summary.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    row = df[df["config"] == config]
    if row.empty:
        return None
    return float(row["AUPR"].iloc[0])


def _ablation_bar(ax, ds, vals, labels, cols, ymax):
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="black",
                  linewidth=0.5, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.025, f"{v:.4f}", ha="center", fontsize=7,
                fontweight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0, ymax)
    ax.set_title(ds, fontsize=10.5)
    ax.set_ylabel("AUPR", fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    return bars


def fig3_ablation():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.0))
    axes = axes.ravel()
    labels = [m[1] for m in ABLATION_MODES]
    cols = [m[2] for m in ABLATION_MODES]
    for ax, ds in zip(axes, DATASETS):
        vals = [_fresh_aupr(ds, m[0]) for m in ABLATION_MODES]
        if any(v is None for v in vals):
            ax.text(0.5, 0.5, f"incomplete ablation for {ds}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color="#dc2626")
            continue
        _ablation_bar(ax, ds, vals, labels, cols, 0.85)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_GMC, edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor="#8fb3c9", edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7f9cb5", edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor="#a895c9", edgecolor="black", linewidth=0.5),
    ]
    fig.legend(handles,
               ["GMC (full model)", "− cold-start KNN fill (mode A)",
                "− block completion view (mode B)",
                "− tensor completion view (mode C)"],
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=7.6, frameon=True, edgecolor="#cbd5e1")
    fig.suptitle("Remove-one-module ablation of GMC (AUPR, fresh validation folds)",
                 fontsize=11, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_three(fig, os.path.join(FIG, "fig3_ablation"))
    plt.close(fig)


# ============================================================================
# Figure 4 — parameter sensitivity (AUPR vs each hyperparameter, one panel per
# axis; data from Results/summaries/*_param_sweep_summary.csv, fresh folds)
# ============================================================================
# Equidistant 5-point grid (2026-08-16): all five axes run on all four
# datasets. Config names encode (axis, value): alpha01/03/07/09, iter20/30/50/60,
# rc200/300/500/600, k5/15/20/25, wt01/03/07/09; the single "center" config
# (the gmc_unified center on every axis) is reused across all five panels.
AXES = [
    ("alpha",   r"$\alpha$",   0.5),
    ("maxiter", "maxiter",     40),
    ("rc",      "rank cap",    400),
    ("k",       "k (fill)",    10),
    ("wt",      r"$w_t$",      0.5),
]
# config-name suffix -> numeric value on each axis (generalized decoder)
_AXIS_DECODE = {
    "alpha": lambda s: int(s) / 100.0,          # 01,03,07,09 -> 0.1..0.9
    "maxiter": lambda s: int(s),                # 20,30,50,60
    "rc": lambda s: int(s),                     # 200,300,500,600
    "k": lambda s: int(s),                      # 5,15,20,25
    "wt": lambda s: int(s) / 100.0,             # 01,03,07,09 -> 0.1..0.9
}
# config-name prefix -> axis key (iter -> maxiter; the rest are the axis name)
_AXIS_PREFIX = {"alpha": "alpha", "iter": "maxiter",
                "rc": "rc", "k": "k", "wt": "wt"}
DS_COLORS = {"Fdataset": C_GMC, "Cdataset": C_DIS,
             "CTDdataset2023": C_TENSOR, "Ydataset": C_INPUT}


def _param_pts(ds):
    """{(axis, value): AUPR} from the param-sweep summary for one dataset.

    The "center" config (== gmc_unified) maps to every axis's center value, so
    each 4-point grid line gains its shared center point.
    """
    p = os.path.join(RES, f"{ds}_param_sweep_summary.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        cfg, v = r["config"], r["AUPR"]
        if cfg == "center":
            for key, _, center in AXES:
                out[(key, center)] = v
            continue
        for prefix, axis in _AXIS_PREFIX.items():
            if cfg.startswith(prefix) and len(cfg) > len(prefix):
                out[(axis, _AXIS_DECODE[axis](cfg[len(prefix):]))] = v
    return out


def fig4_param():
    fig, axes = plt.subplots(1, len(AXES), figsize=(15.5, 3.2))
    for ax, (key, label, center) in zip(axes, AXES):
        for ds in DATASETS:
            pts = _param_pts(ds)
            xs = sorted([x for (k, x) in pts if k == key])
            ys = [pts[(key, x)] for x in xs]
            ax.plot(xs, ys, "o-", label=ds, color=DS_COLORS[ds],
                    lw=1.6, markersize=4)
        ax.axvline(center, color=DARK, ls="--", lw=1.0, alpha=0.6)
        ax.set_title(label, fontsize=9.5)
        ax.set_ylabel("AUPR" if ax is axes[0] else "", fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
    handles = [plt.Line2D([], [], color=DS_COLORS[d], lw=1.6, label=d)
               for d in DATASETS]
    handles.append(plt.Line2D([], [], color=DARK, lw=1.0, ls="--", label="center"))
    fig.legend(handles, DATASETS + ["center"], ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), fontsize=8, frameon=True,
               edgecolor="#cbd5e1")
    fig.suptitle("Parameter sensitivity of the unified configuration (AUPR, fresh folds)",
                 fontsize=11, y=1.12)
    fig.tight_layout()
    save_three(fig, os.path.join(FIG, "fig4_param"))
    plt.close(fig)


# ============================================================================
# Figure 5 — robustness to missing data (AUPR vs held-out fraction; data from
# Results/summaries/*_robust_mask_summary.csv)
# ============================================================================
def fig5_robust():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ds in DATASETS:
        p = os.path.join(RES, f"{ds}_robust_mask_summary.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p).sort_values("frac")
        ax.plot(df["frac"], df["AUPR"], "o-", label=ds, lw=1.8, markersize=5)
    ax.set_xlabel("Held-out fraction (mask rate)", fontsize=9)
    ax.set_ylabel("AUPR", fontsize=9)
    ax.set_xticks([0.05, 0.10, 0.20, 0.30])
    ax.grid(ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=True, edgecolor="#cbd5e1")
    ax.set_title("Robustness to missing data (fixed unified config)",
                 fontsize=10.5)
    fig.tight_layout()
    save_three(fig, os.path.join(FIG, "fig5_robust"))
    plt.close(fig)


if __name__ == "__main__":
    print("Generating GMC figures →", FIG)
    fig1_framework()
    fig2_main()
    fig3_ablation()
    fig4_param()
    fig5_robust()
    print("done.")
