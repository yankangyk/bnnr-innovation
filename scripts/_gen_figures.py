"""
Generate publication-quality figures for BNNR Innovation paper.

Fig 1: γ × λ heatmap (λ-insensitivity discovery)
Fig 2: GF-BNNR α sensitivity line plot
Fig 3: Inside-outside framework schematic
Fig 4: GBNNR + GF-BNNR stack bar chart

Output: papers/figures/fig_*.pdf and fig_*.png
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Style ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "papers", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def save(name):
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), format=ext)
    print(f"  Saved {name}")


# ═════════════════════════════════════════════════════════════════════════════════
# Fig 1: γ × λ Heatmap — λ-insensitivity discovery
# ═════════════════════════════════════════════════════════════════════════════════
def fig1_heatmap():
    gammas = [0.5, 1.0, 2.0, 3.0]
    lambdas = [0, 1e-3, 1e-2, 1e-1]
    lambda_labels = [r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]

    data = np.array([
        [0.3273, 0.3272, 0.3260, 0.2829],  # γ=0.5
        [0.3273, 0.3272, 0.3269, 0.2832],  # γ=1.0
        [0.3273, 0.3269, 0.3312, 0.2806],  # γ=2.0
        [0.3273, 0.3269, 0.3286, 0.2577],  # γ=3.0
    ])
    bnnr_baseline = 0.3071

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.255, vmax=0.332)

    for i in range(len(gammas)):
        for j in range(len(lambdas)):
            val = data[i, j]
            color = "white" if val > 0.315 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    # Dashed divider between inert region (λ=0..10⁻²) and degraded (λ=10⁻¹)
    ax.axvline(x=2.5, color="black", lw=1.6, ls="--", alpha=0.55)

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(lambda_labels, fontsize=11)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([rf"$\gamma = {g}$" for g in gammas], fontsize=11)
    ax.set_xlabel("Regularization strength " + r"$\lambda$", fontsize=12)
    ax.set_ylabel("Confidence parameter " + r"$\gamma$", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("AUPR", fontsize=11)

    # Title with built-in annotation — no extra boxes or arrows
    ax.set_title(
        f"GBNNR: " + r"$\gamma \times \lambda$" + f" sweep on Fdataset fold 1\n"
        f"(BNNR baseline AUPR = {bnnr_baseline:.4f}; "
        + r"$\lambda = 0$ to $10^{-2}$ inert, $\lambda = 10^{-1}$ degrades"
        + ")",
        fontsize=10.5, fontweight="bold", pad=12)

    plt.tight_layout()
    save("fig1_lambda_heatmap")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════════
# Fig 2: GF-BNNR α sensitivity line plot
# ═════════════════════════════════════════════════════════════════════════════════
def fig2_alpha_sensitivity():
    alphas = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    # Actual experimental results from _alpha_sweep.py (Fdataset/Cdataset/DNdataset, fold 1)
    f_aupr  = [0.3157, 0.3148, 0.3113, 0.3118, 0.3087, 0.3036]
    f_auroc = [0.9138, 0.9134, 0.9132, 0.9132, 0.9135, 0.9142]
    c_aupr  = [0.4090, 0.4072, 0.4054, 0.3983, 0.4047, 0.4003]
    c_auroc = [0.9516, 0.9540, 0.9561, 0.9568, 0.9569, 0.9560]
    dn_aupr = [0.3259, 0.3259, 0.3259, 0.3259, 0.3260, 0.3260]
    dn_auroc = [0.9311, 0.9658, 0.9722, 0.9735, 0.9740, 0.9743]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharex=True)

    datasets = [
        ("Fdataset (1.04%)", f_aupr, "o", "#2196F3"),
        ("Cdataset (0.93%)", c_aupr, "s", "#4CAF50"),
        ("DNdataset (0.015%)", dn_aupr, "^", "#FF5722"),
    ]

    for ax, (name, aupr, marker, color) in zip(axes, datasets):
        ax.plot(alphas, aupr, marker=marker, color=color, lw=2,
                markersize=8, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8)
        ax.axhline(y=aupr[0], color="gray", ls="--", lw=0.8, alpha=0.6,
                   label=f"α=0 (GIP only): {aupr[0]:.4f}")
        best_idx = np.argmax(aupr)
        ax.scatter([alphas[best_idx]], [aupr[best_idx]], s=120, color="red",
                   zorder=10,
                   label=f"Best α={alphas[best_idx]:.1f}: {aupr[best_idx]:.4f}")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("AUPR", fontsize=11)
        ax.set_xlabel("Filter strength α_f", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("GF-BNNR: Effect of Filter Strength on AUPR (fold 1)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save("fig2_alpha_sensitivity")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════════
# Fig 3: Inside-Outside Framework Schematic
# ═════════════════════════════════════════════════════════════════════════════════
def fig3_framework():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    c_bnnr   = "#607D8B"
    c_gbnnr  = "#FF9800"
    c_gfbnnr = "#2196F3"
    c_manifold = "#4CAF50"
    c_data   = "#37474F"
    c_arrow  = "#555555"
    c_filter_box = "#E3F2FD"

    def box(ax, x, y, w, h, text, color, fontsize=9.5, fontweight="normal",
            text_color="white", edge_color=None, lw=1.5):
        if edge_color is None:
            edge_color = color
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor=edge_color, linewidth=lw, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=text_color)

    def arrow(ax, x1, y1, x2, y2, color=c_arrow, lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    def label(ax, x, y, text, fontsize=8.5, color="black", ha="center",
              fontweight="normal", fontstyle="normal"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
                color=color, fontweight=fontweight, fontstyle=fontstyle)

    # ── Title ──
    ax.text(5.5, 6.55, "Inside-Outside Framework for Manifold-Aware Matrix Completion",
            ha="center", va="top", fontsize=14, fontweight="bold")

    # ── Left: Input data ──
    box(ax, 1.2, 4.8, 1.8, 0.9, "Similarity\nNetworks", c_data, fontsize=9, fontweight="bold")
    box(ax, 1.2, 3.3, 1.8, 0.9, "Association\nMatrix", c_data, fontsize=9, fontweight="bold")
    # Input arrows (curved into BNNR core)
    arrow(ax, 2.1, 4.8, 2.7, 4.8)
    arrow(ax, 2.1, 3.3, 2.7, 3.3)

    # ── Center: BNNR core container ──
    box(ax, 4.2, 4.05, 2.8, 2.8, "", "#ECEFF1", edge_color="#90A4AE", lw=1.5)
    label(ax, 4.2, 5.2, "BNNR ADMM", color=c_bnnr, fontsize=12, fontweight="bold")
    label(ax, 4.2, 4.4, "W-update (LS)\nX-update (SVT)\nY-update (dual)",
          fontsize=9, color="#455A64")

    # ── GBNNR (inside BNNR, at bottom) ──
    box(ax, 4.2, 3.2, 1.9, 0.7, "GBNNR", c_gbnnr, fontsize=10, fontweight="bold")
    label(ax, 4.2, 2.75, "kNN Laplacian + Inner GD", fontsize=7.5, color="#BF360C")
    # "inside" bracket on left of BNNR core
    ax.plot([2.65, 2.65], [2.75, 3.55], color=c_gbnnr, lw=2.2)
    ax.plot([2.35, 2.65], [3.55, 3.55], color=c_gbnnr, lw=2.2)
    label(ax, 1.70, 3.55, "inside", fontsize=9, color=c_gbnnr, ha="right", fontweight="bold")

    # ── Right: Output side ──
    # Completed matrix
    box(ax, 7.1, 4.05, 1.9, 1.1, "Completed\nMatrix " + r"$\mathbf{M}_{raw}$",
        "#78909C", fontsize=9, fontweight="bold")
    arrow(ax, 5.6, 4.05, 6.15, 4.05)
    label(ax, 5.88, 4.35, "output", fontsize=7.5, color="#777")

    # GF Filter with formula as subtitle inside the box
    box(ax, 9.1, 5.25, 2.6, 0.85,
        "Graph Low-Pass Filter\n" + r"$(\mathbf{I}+\alpha_f\mathbf{L}_d)^{-1} \mathbf{M} (\mathbf{I}+\alpha_f\mathbf{L}_r)^{-1}$",
        c_filter_box, fontsize=8.0, fontweight="bold", text_color="#0D47A1",
        edge_color=c_gfbnnr)

    # Filtered matrix (below GF filter)
    box(ax, 9.1, 3.55, 2.2, 1.0, "Filtered\nMatrix " + r"$\mathbf{M}_{\rm filt}$",
        c_gfbnnr, fontsize=9, fontweight="bold")

    # Arrows: completed → filter → filtered
    arrow(ax, 7.9, 4.6, 7.9, 4.825)     # up to filter box bottom
    arrow(ax, 9.1, 4.825, 9.1, 4.05)    # down from filter to filtered

    # "outside" bracket on right (encompassing filter box + formula)
    ax.plot([10.35, 10.35], [3.1, 5.78], color=c_gfbnnr, lw=2.2)
    ax.plot([10.35, 10.75], [5.78, 5.78], color=c_gfbnnr, lw=2.2)
    label(ax, 10.95, 5.78, "outside", fontsize=9, color=c_gfbnnr, ha="left", fontweight="bold")

    # ── Bottom: Manifold signal + RA-BNNR ──
    label(ax, 5.2, 1.80, "Both strategies draw from the same manifold signal",
          fontsize=9.5, color=c_manifold, fontweight="bold")
    label(ax, 5.2, 1.35, "Stacking yields no additive gain (AUPR 0.3237 vs 0.3269)",
          fontsize=8, color="#666")

    # Curved manifold arrows (bottom connecting path)
    ax.annotate("", xy=(4.2, 2.85), xytext=(7.1, 1.95),
                arrowprops=dict(arrowstyle="->", color=c_manifold, lw=1.3,
                               connectionstyle="arc3,rad=-0.20"))
    ax.annotate("", xy=(7.1, 1.80), xytext=(9.1, 3.05),
                arrowprops=dict(arrowstyle="->", color=c_manifold, lw=1.3,
                               connectionstyle="arc3,rad=-0.20"))

    # RA-BNNR supplementary bar
    box(ax, 4.2, 0.85, 3.2, 0.55, "RA-BNNR: rank-adaptive " + r"$\beta$" + " (complementary)",
        "#F5F5F5", fontsize=8, text_color="#424242", edge_color="#BDBDBD", lw=1.0)

    plt.tight_layout(pad=0.5)
    save("fig3_framework_schematic")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════════
# Fig 4: GBNNR + GF-BNNR Stack Bar Chart
# ═════════════════════════════════════════════════════════════════════════════════
def fig4_stack_bars():
    methods = ["BNNR", "GBNNR\nλ=0", "GBNNR\nλ=1e-3", "GF-BNNR", "GBNNR\n+GF"]
    aupr_vals = [0.3071, 0.3273, 0.3269, 0.3118, 0.3237]
    auroc_vals = [0.9109, 0.9097, 0.9094, 0.9132, 0.9084]
    colors = ["#607D8B", "#FFB74D", "#FF9800", "#2196F3", "#AB47BC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # AUPR
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, aupr_vals, color=colors, edgecolor="white", lw=0.8,
                    width=0.65)
    ax1.axhline(y=aupr_vals[0], color="gray", ls="--", lw=1.0, alpha=0.5)
    for bar, val in zip(bars1, aupr_vals):
        delta = val - aupr_vals[0]
        sign = "+" if delta >= 0 else ""
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}\n({sign}{delta:.4f})",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("AUPR", fontsize=11, fontweight="bold")
    ax1.set_title("AUPR Comparison", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.30, 0.345)
    ax1.grid(axis="y", alpha=0.3)

    # AUROC
    bars2 = ax2.bar(x, auroc_vals, color=colors, edgecolor="white", lw=0.8,
                    width=0.65)
    ax2.axhline(y=auroc_vals[0], color="gray", ls="--", lw=1.0, alpha=0.5)
    for bar, val in zip(bars2, auroc_vals):
        delta = val - auroc_vals[0]
        sign = "+" if delta >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                 f"{val:.4f}\n({sign}{delta:.4f})",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel("AUROC", fontsize=11, fontweight="bold")
    ax2.set_title("AUROC Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.905, 0.918)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("GBNNR + GF-BNNR Stacking Experiment (Fdataset, fold 1)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save("fig4_stack_bars")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig1_heatmap()
    fig2_alpha_sensitivity()
    fig3_framework()
    fig4_stack_bars()
    print(f"Done. Figures saved to {OUT_DIR}")
