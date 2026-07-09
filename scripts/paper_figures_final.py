"""
BADGE paper — publication-quality figures (v2, corrected)
=========================================================
Generates 5 figures for the BADGE manuscript:
  Fig 1: BADGE framework schematic
  Fig 2: Density-adaptive shrinkage function lambda(rho)
  Fig 3: Main AUPR comparison (4 methods x 3 datasets)
  Fig 4: Convergence analysis (N=1,2,3 with SD bands)
  Fig 5: GF-BNNR filter strength alpha_f sensitivity

Output: PDF (vector, fonttype=42) + 300 DPI PNG, Arial font.
All data verified against Results/BADGE/*/E*_summary.json.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ===========================================================================
# Nature/Bioinformatics rcParams
# ===========================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "mathtext.fontset": "cm",
})

# ===========================================================================
# Colorblind-safe palette (Okabe-Ito derived)
# ===========================================================================
C_DN    = "#0072B2"
C_F     = "#E69F00"
C_C     = "#D55E00"
C_BADGE = "#009E73"
C_GRAY  = "#767676"
C_BLACK = "#272727"

DATASET_STYLE = {
    "DNdataset": {"color": C_DN,    "marker": "o", "label": "DNdataset"},
    "Fdataset":  {"color": C_F,     "marker": "s", "label": "Fdataset"},
    "Cdataset":  {"color": C_C,     "marker": "D", "label": "Cdataset"},
}
METHOD_ORDER = ["BNNR", "GBNNR", "GF-BNNR", "BADGE\n(N=2)"]
METHOD_COLOR = {
    "BNNR":          C_GRAY,
    "GBNNR":         "#56B4E9",
    "GF-BNNR":       "#9AA0A6",
    "BADGE\n(N=2)":  C_BADGE,
}

# ===========================================================================
# Experimental data (10-fold CVa) — verified against summary.json files
# ===========================================================================
AUPR_MEAN = {
    "DNdataset": np.array([0.2564, 0.2539, 0.3166, 0.3207]),
    "Fdataset":  np.array([0.3061, 0.3199, 0.3153, 0.3233]),
    "Cdataset":  np.array([0.2772, 0.4006, 0.3958, 0.4051]),
}
AUPR_SD = {
    "DNdataset": np.array([0.1345, 0.1331, 0.0207, 0.0227]),
    "Fdataset":  np.array([0.0240, 0.0273, 0.0251, 0.0280]),
    "Cdataset":  np.array([0.1212, 0.0198, 0.0195, 0.0215]),
}

# BADGE convergence: N=1 (=GF-BNNR), N=2, N=3 — with SD
CONV_AUPR = {
    "DNdataset": [(0.3166, 0.0207), (0.3207, 0.0227), (0.3211, 0.0219)],
    "Fdataset":  [(0.3153, 0.0251), (0.3233, 0.0280), (0.3199, 0.0287)],
    "Cdataset":  [(0.3958, 0.0195), (0.4051, 0.0215), (0.4037, 0.0227)],
}

# Shrinkage weights — VERIFIED from shrinkage_lambda_mean in summary.json
LAMBDA_VAL = {"DNdataset": 0.0521, "Fdataset": 0.9623, "Cdataset": 0.9561}
DENSITY_PCT = {"DNdataset": 0.015, "Fdataset": 1.04, "Cdataset": 0.93}

# Shrinkage function parameters (Section 2.3.2)
MU, TAU = -3.0, 0.3

# Alpha_f sweep (from Results/alpha_sweep/alpha_sweep_results.csv)
ALPHA_SWEEP = {
    "Fdataset":  {"alpha": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                  "auroc": [0.9138, 0.9134, 0.9132, 0.9132, 0.9135, 0.9142],
                  "aupr":  [0.3157, 0.3148, 0.3113, 0.3118, 0.3087, 0.3036]},
    "Cdataset":  {"alpha": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                  "auroc": [0.9516, 0.9540, 0.9561, 0.9568, 0.9569, 0.9560],
                  "aupr":  [0.4090, 0.4072, 0.4054, 0.3983, 0.4047, 0.4003]},
    "DNdataset": {"alpha": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                  "auroc": [0.9311, 0.9658, 0.9722, 0.9735, 0.9740, 0.9743],
                  "aupr":  [0.3259, 0.3259, 0.3259, 0.3259, 0.3260, 0.3260]},
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================
def save_fig(fig, name):
    for fmt, dpi in [("pdf", None), ("png", 300)]:
        path = os.path.join(OUT_DIR, f"{name}.{fmt}")
        kw = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.02} if dpi \
             else {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(path, **kw)
        print(f"  -> {path}")
    plt.close(fig)


def panel_label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


def shrinkage(rho):
    return 1.0 / (1.0 + np.exp(-(np.log10(rho) - MU) / TAU))


# ===========================================================================
# FIGURE 1 — BADGE Framework Schematic (single-column vertical layout)
# ===========================================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(3.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    C_IN   = "#F5F5F5"
    C_GIP  = "#E6F0FF"
    C_CORE = "#D4EDDA"
    C_BNNR = "#EEEEEE"
    C_FILT = "#FFE8CC"
    C_OUT  = "#F0F0F0"

    def box(x, y, w, h, text, fc, ec="#999", lw=1.2, fs=7.5, bold=False, tc=C_BLACK):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(rect)
        fw = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=3)

    def arrow(x1, y1, x2, y2, color=C_GRAY, lw=1.3, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    linestyle=ls, connectionstyle="arc3,rad=0"),
                    zorder=1)

    # INPUT block
    box(0.5, 13.3, 9.0, 1.4,
        "Input\n$\\mathbf{S}_{rr}, \\mathbf{S}_{dd}, \\mathbf{W}_{dr}$",
        C_IN, ec=C_GRAY, fs=8.5, bold=True)

    # Prior GIP + Shrinkage
    box(0.5, 11.0, 4.2, 1.8, "Prior GIP\n$\\mathbf{G}^{prior}$",
        C_GIP, ec="#9DB8D8", fs=8)
    box(5.3, 11.0, 4.2, 1.8,
        "Density-adaptive\nshrinkage $\\lambda(\\rho)$\n[Core]",
        C_CORE, ec=C_BADGE, lw=2.5, fs=8, bold=True, tc=C_BADGE)
    arrow(4.7, 11.9, 5.3, 11.9, color=C_BADGE, lw=1.6)
    arrow(5.0, 13.3, 2.6, 12.8)
    arrow(5.0, 13.3, 7.4, 12.8, color=C_BADGE)

    # Iterative loop boundary
    loop = FancyBboxPatch((0.3, 3.5), 9.4, 6.9, boxstyle="round,pad=0.2",
                          facecolor="none", edgecolor=C_GRAY, linewidth=1.2,
                          linestyle="--", zorder=0)
    ax.add_patch(loop)
    ax.text(5.0, 10.15, "Iterative Refinement  ($N=2$)",
            fontsize=8.5, color=C_GRAY, ha="center", fontweight="bold",
            style="italic")

    # Fused similarity -> BNNR -> filter
    box(0.6, 8.4, 2.6, 1.4, "Fused $\\tilde{\\mathbf{S}}$\n$w\\mathbf{G}+(1-w)\\mathbf{S}$",
        C_GIP, ec="#9DB8D8", fs=7)
    box(3.7, 8.4, 2.6, 1.4, "BNNR\nADMM + SVT", C_BNNR, ec="#BBB", fs=7.5)
    box(6.8, 8.4, 2.6, 1.4, "Graph Filter\n$(\\mathbf{I}+\\alpha_f\\mathbf{L})^{-1}$",
        C_FILT, ec="#E0C090", fs=7)
    arrow(3.2, 9.1, 3.7, 9.1)
    arrow(6.3, 9.1, 6.8, 9.1)
    arrow(2.6, 11.0, 1.9, 9.8)

    # M_cur output
    box(6.8, 6.6, 2.6, 1.2, "$\\mathbf{M}_{cur}$\n(completed)", C_BNNR,
        ec="#BBB", fs=7.5, bold=True)
    arrow(8.1, 8.4, 8.1, 7.8)

    # Empirical GIP + Bayesian fusion
    box(0.6, 6.6, 2.6, 1.2, "Empirical GIP\n$\\mathbf{G}^{emp}$",
        C_GIP, ec="#9DB8D8", fs=7)
    arrow(6.8, 7.2, 3.2, 7.2)

    box(3.7, 6.4, 2.6, 1.6,
        "Bayesian Fusion\n$\\mathbf{G}=\\lambda\\mathbf{G}^{emp}+(1-\\lambda)\\mathbf{G}^{prior}$",
        C_CORE, ec=C_BADGE, lw=2.0, fs=7, bold=True, tc=C_BADGE)
    arrow(3.2, 7.2, 3.7, 7.2)
    arrow(7.4, 11.0, 5.0, 8.0, color=C_BADGE, lw=1.2, ls="--")

    # Feedback arc
    ax.annotate("", xy=(1.9, 9.8), xytext=(5.0, 6.4),
                arrowprops=dict(arrowstyle="->", color=C_BADGE, lw=1.6,
                                connectionstyle="arc3,rad=-0.45"), zorder=4)
    ax.text(0.55, 8.2, "update\n$\\tilde{\\mathbf{S}}$", fontsize=6.5,
            color=C_BADGE, ha="left", va="center", fontweight="bold",
            style="italic")
    ax.text(5.0, 5.9, "if $t<N$ and $\\lambda>0.01$",
            fontsize=6.5, color=C_BADGE, ha="center", style="italic",
            bbox=dict(facecolor="white", edgecolor=C_BADGE,
                      boxstyle="round,pad=0.2", lw=0.8))

    # OUTPUT block
    box(0.5, 1.8, 9.0, 1.3, "Output: $\\mathbf{M}_{cur}$ (predicted associations)",
        C_OUT, ec=C_GRAY, fs=8, bold=True)
    arrow(8.1, 6.6, 8.1, 3.8, lw=1.5)
    ax.text(8.4, 5.2, "if $t=N$", fontsize=6.5, color=C_GRAY,
            ha="left", style="italic")

    # Module color legend
    legend_items = [
        ("Input",          C_IN,   C_GRAY),
        ("GIP module",     C_GIP,  "#9DB8D8"),
        ("BNNR ADMM",      C_BNNR, "#BBB"),
        ("Graph filter",   C_FILT, "#E0C090"),
        ("Core innovation", C_CORE, C_BADGE),
    ]
    for i, (lbl, fc, ec) in enumerate(legend_items):
        x0 = 0.5 + i * 1.85
        rect = FancyBboxPatch((x0, 0.6), 1.7, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + 0.85, 0.9, lbl, fontsize=6, ha="center", va="center",
                fontweight="bold" if "Core" in lbl else "normal")

    fig.tight_layout(pad=0.4)
    save_fig(fig, "fig1_framework")
    print("Fig 1 done.\n")


# ===========================================================================
# FIGURE 2 — Density-adaptive shrinkage function lambda(rho)
# ===========================================================================
def fig2_shrinkage():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    rho = np.logspace(-5, -0.5, 800)
    lam = shrinkage(rho)
    ax.plot(rho * 100, lam, color=C_BADGE, lw=2.2, zorder=3,
            label=r"$\lambda(\rho) = \sigma\!\left(\frac{\log_{10}\rho - \mu}{\tau}\right)$")

    # Regime shading
    rho_ultra = np.logspace(-5, np.log10(0.001), 200)
    ax.fill_between(rho_ultra * 100, shrinkage(rho_ultra), alpha=0.12,
                    color=C_DN, zorder=1, label="Ultra-sparse\n(prior dominates)")
    rho_mod = np.logspace(np.log10(0.003), -0.5, 200)
    ax.fill_between(rho_mod * 100, shrinkage(rho_mod), alpha=0.10,
                    color="#F4A0A0", zorder=1,
                    label="Moderate density\n(empirical trusted)")

    # lambda=0.5 threshold
    ax.axhline(0.5, color=C_GRAY, lw=0.8, ls="--", alpha=0.6, zorder=2)
    ax.text(0.012, 0.52, r"$\lambda=0.5$", fontsize=7, color=C_GRAY,
            va="bottom", ha="left")

    # mu marker
    ax.axvline(0.1, color=C_GRAY, lw=0.6, ls=":", alpha=0.5, zorder=2)
    ax.text(0.13, 0.04, r"$\mu=-3.0$" "\n" r"($\rho=0.1\%$)",
            fontsize=6.5, color=C_GRAY, va="bottom", ha="left")

    # Dataset markers — labels outside the curve
    label_specs = [
        ("DNdataset", 0.015,  0.04,  0.22, "left"),
        ("Cdataset",  0.93,   3.5,   0.92, "left"),
        ("Fdataset",  1.04,   3.5,   0.80, "left"),
    ]
    for name, pct, lx, ly, ha in label_specs:
        cfg = DATASET_STYLE[name]
        lam_val = LAMBDA_VAL[name]
        ax.plot([pct], [lam_val], marker=cfg["marker"], color=cfg["color"],
                markersize=8, markeredgecolor="white", markeredgewidth=0.9, zorder=6)
        ax.annotate("", xy=(pct, lam_val), xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="-", color=cfg["color"],
                                    lw=0.7, connectionstyle="arc3,rad=0.0"), zorder=5)
        ax.text(lx + (0.05 if ha == "left" else -0.05), ly,
                f"{cfg['label']}\n" + r"$\rho=$" + f"{pct:.3f}%\n"
                r"$\lambda=$" + f"{lam_val:.3f}",
                fontsize=6.8, color=cfg["color"], fontweight="bold",
                va="center", ha=ha, zorder=7,
                bbox=dict(facecolor="white", edgecolor=cfg["color"],
                          boxstyle="round,pad=0.2", lw=0.6, alpha=0.95))

    ax.set_xscale("log")
    ax.set_xlabel(r"Association density $\rho$ (%)", labelpad=4)
    ax.set_ylabel(r"Shrinkage weight $\lambda(\rho)$", labelpad=4)
    ax.set_xlim(0.008, 8)
    ax.set_ylim(-0.02, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2,
              handletextpad=0.4, labelspacing=0.3, borderpad=0.3)

    fig.tight_layout(pad=0.5)
    save_fig(fig, "fig2_shrinkage")
    print("Fig 2 done.\n")


# ===========================================================================
# FIGURE 3 — Main AUPR comparison (3 panels, per-dataset y-axis)
# ===========================================================================
def fig3_main_aupr():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    datasets = ["DNdataset", "Fdataset", "Cdataset"]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        means = AUPR_MEAN[ds]
        sds = AUPR_SD[ds]
        x = np.arange(len(METHOD_ORDER))

        y_min = max(0.0, min(means - sds) * 0.85)
        y_max = max(means + sds) * 1.15
        ax.set_ylim(y_min, y_max)

        for xi, m, s, mname in zip(x, means, sds, METHOD_ORDER):
            color = METHOD_COLOR[mname]
            is_badge = "BADGE" in mname
            edge_c = C_BLACK if is_badge else "none"
            edge_w = 1.8 if is_badge else 0
            ax.bar(xi, m, 0.62, color=color, edgecolor=edge_c,
                   linewidth=edge_w, zorder=3, alpha=0.95)

        ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=C_BLACK,
                    elinewidth=0.9, capsize=3, capthick=0.9, zorder=4)

        for xi, m, s in zip(x, means, sds):
            ax.text(xi, m + s + (y_max - y_min) * 0.025,
                    f"{m:.4f}", ha="center", va="bottom",
                    fontsize=6.8, fontweight="bold", color=C_BLACK)

        gf_val = means[2]
        ax.axhline(gf_val, color=C_GRAY, lw=0.7, ls="--", alpha=0.5, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, fontsize=7)
        ax.set_title(f"{ds}\n" + r"($\rho=$" + f"{DENSITY_PCT[ds]}%)",
                     fontsize=8.5, fontweight="bold", pad=8)
        if i == 0:
            ax.set_ylabel("AUPR", fontsize=9)
        ax.yaxis.grid(True, lw=0.4, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    legend_elements = [Patch(facecolor=METHOD_COLOR[m], edgecolor=C_BLACK
                             if "BADGE" in m else "none",
                             linewidth=1.5 if "BADGE" in m else 0,
                             label=m.replace("\n", " ")) for m in METHOD_ORDER]
    legend_elements.append(Line2D([0], [0], color=C_GRAY, lw=0.8, ls="--",
                                  label="GF-BNNR baseline"))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=1.0, rect=[0, 0.06, 1, 1])
    save_fig(fig, "fig3_main_aupr")
    print("Fig 3 done.\n")


# ===========================================================================
# FIGURE 4 — Convergence analysis (N=1,2,3) with SD bands
# ===========================================================================
def fig4_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    datasets = ["DNdataset", "Fdataset", "Cdataset"]
    N_vals = np.array([1, 2, 3])
    N_labels = ["N=1\n(GF-BNNR)", "N=2\n(BADGE)", "N=3"]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        cfg = DATASET_STYLE[ds]
        means = np.array([v[0] for v in CONV_AUPR[ds]])
        sds = np.array([v[1] for v in CONV_AUPR[ds]])

        ax.fill_between(N_vals, means - sds, means + sds,
                        color=cfg["color"], alpha=0.18, zorder=1)
        ax.plot(N_vals, means, color=cfg["color"], lw=1.8,
                marker=cfg["marker"], markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        gf_val = means[0]
        ax.axhline(gf_val, color=C_GRAY, lw=0.7, ls="--", alpha=0.5, zorder=1)

        for xi, m, s in zip(N_vals, means, sds):
            ax.text(xi, m + s + (means.max() - means.min() + sds.max()) * 0.15,
                    f"{m:.4f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=cfg["color"])

        ax.scatter([2], [means[1]], marker="*", s=110, color=C_BADGE,
                   edgecolor="white", linewidth=0.8, zorder=6)
        ax.text(2, means[1] - sds[1] - (means.max() - means.min() + sds.max()) * 0.18,
                "optimal", ha="center", va="top",
                fontsize=6.5, color=C_BADGE, fontweight="bold", style="italic")

        imp = (means[1] - means[0]) / means[0] * 100
        sign = "+" if imp >= 0 else ""
        ax.text(0.97, 0.06, f"{sign}{imp:.1f}% vs GF-BNNR",
                transform=ax.transAxes, fontsize=7.5, color=C_BADGE,
                fontweight="bold", ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor=C_BADGE,
                          boxstyle="round,pad=0.25", lw=0.7, alpha=0.9))

        ax.text(0.03, 0.94, r"$\lambda=$" + f"{LAMBDA_VAL[ds]:.3f}",
                transform=ax.transAxes, fontsize=7, color=C_GRAY,
                ha="left", va="top", style="italic")

        ax.set_xticks(N_vals)
        ax.set_xticklabels(N_labels, fontsize=7)
        ax.set_xlim(0.7, 3.3)
        all_vals = np.concatenate([means - sds, means + sds])
        margin = (all_vals.max() - all_vals.min()) * 0.35 + 0.005
        ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)
        if i == 0:
            ax.set_ylabel("AUPR", fontsize=9)
        ax.set_xlabel("Refinement iteration", fontsize=8.5, labelpad=4)
        ax.set_title(ds, fontsize=9, fontweight="bold", pad=6)
        ax.yaxis.grid(True, lw=0.4, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=1.2)
    save_fig(fig, "fig4_convergence")
    print("Fig 4 done.\n")


# ===========================================================================
# FIGURE 5 — GF-BNNR filter strength alpha_f sensitivity
# ===========================================================================
def fig5_alpha_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    datasets = ["DNdataset", "Fdataset", "Cdataset"]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        cfg = DATASET_STYLE[ds]
        data = ALPHA_SWEEP[ds]
        alphas = data["alpha"]
        auroc = data["auroc"]
        aupr = data["aupr"]

        l1, = ax.plot(alphas, auroc, color=cfg["color"], lw=1.8,
                      marker=cfg["marker"], markersize=6,
                      markeredgecolor="white", markeredgewidth=0.6,
                      label="AUROC", zorder=4)
        ax.set_ylabel("AUROC", color=cfg["color"], fontsize=8.5)
        ax.tick_params(axis="y", labelcolor=cfg["color"])
        ax.set_xlabel(r"Filter strength $\alpha_f$", fontsize=8.5, labelpad=3)

        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        l2, = ax2.plot(alphas, aupr, color=C_BADGE, lw=1.8,
                       marker="^", markersize=6,
                       markeredgecolor="white", markeredgewidth=0.6,
                       label="AUPR", zorder=4)
        ax2.set_ylabel("AUPR", color=C_BADGE, fontsize=8.5)
        ax2.tick_params(axis="y", labelcolor=C_BADGE)

        ax.axvline(0.5, color=C_GRAY, lw=0.7, ls=":", alpha=0.6, zorder=1)
        ax.text(0.52, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                r"$\alpha_f=0.5$" "\n(default)",
                fontsize=6.5, color=C_GRAY, va="bottom", ha="left", style="italic")

        ax.set_title(f"{ds}\n" + r"($\rho=$" + f"{DENSITY_PCT[ds]}%)",
                     fontsize=8.5, fontweight="bold", pad=8)
        ax.set_xticks(alphas)

        if i == 0:
            ax.legend(handles=[l1, l2], loc="lower center", fontsize=7,
                      frameon=False, bbox_to_anchor=(0.5, -0.35), ncol=2)

    fig.tight_layout(pad=1.2, rect=[0, 0.05, 1, 1])
    save_fig(fig, "fig5_alpha_sweep")
    print("Fig 5 done.\n")


# ===========================================================================
if __name__ == "__main__":
    print(f"BADGE paper figures -> {OUT_DIR}\n")
    fig1_framework()
    fig2_shrinkage()
    fig3_main_aupr()
    fig4_convergence()
    fig5_alpha_sweep()
    print(f"All figures saved to: {OUT_DIR}")
