"""Plotting utilities for PWAS figures."""

import numpy as np
from scipy import stats


def grouped_barplot(stats_df, y_col, se_col, label_col=None, label_fontsize=6, ax=None):
    """Two-level grouped barplot from a MultiIndex DataFrame."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    level1_values = stats_df.index.get_level_values(0).unique()
    level2_values = stats_df.index.get_level_values(1).unique()

    bar_width = 0.25
    positions = np.arange(len(level1_values))
    palette = plt.cm.Blues(np.linspace(0.2, 0.7, len(level2_values)))

    for i, level2 in enumerate(level2_values):
        subset = stats_df.xs(level2, level=1, drop_level=False)
        container = ax.bar(
            x=positions + i * bar_width - bar_width,
            height=subset[y_col],
            yerr=subset[se_col],
            width=bar_width,
            label=f"{level2}",
            color=palette[i],
            capsize=0.0,
            error_kw={"linewidth": 0.7, "color": "black"},
        )
        if label_col is not None:
            ax.bar_label(
                container, labels=subset[label_col].values,
                fontsize=label_fontsize, label_type="center",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(level1_values)
    ax.legend(fontsize=8, columnspacing=0.5, handletextpad=0.5, frameon=False, loc="upper right")


def genome_wide_manhattan(chrom, pos, z, annot=None, ax=None, log10p_height=-np.log10(5e-8)):
    """Manhattan plot for genome-wide data."""
    import matplotlib.pyplot as plt

    log10_p_values = -(stats.norm.logsf(np.abs(z)) + np.log(2)) / np.log(10)

    sorted_indices = np.lexsort((pos, chrom))
    z_sorted = np.array(z)[sorted_indices]
    chrom_sorted = np.array(chrom)[sorted_indices]
    pos_sorted = np.array(pos)[sorted_indices]
    log10_p_values_sorted = log10_p_values[sorted_indices]

    if annot is not None:
        annot_sorted = np.array(annot)[sorted_indices]

    x = np.arange(len(z_sorted))
    if ax is None:
        ax = plt.gca()

    colors = ["#1b9e77", "#d95f02"]
    unique_chroms = np.unique(chrom_sorted)

    for i, c in enumerate(unique_chroms):
        mask = chrom_sorted == c
        ax.scatter(x[mask], log10_p_values_sorted[mask], s=1, color=colors[i % 2], rasterized=True)

    if annot is not None:
        mask = np.array([a != "" for a in annot_sorted])
        if np.any(mask):
            ax.scatter(
                x[mask], np.minimum(log10_p_values_sorted[mask], 50),
                s=10, color="red", zorder=5,
            )
            for i in np.where(mask)[0]:
                ax.annotate(
                    f"{annot_sorted[i]}",
                    (x[i], np.minimum(log10_p_values_sorted[i], 50)),
                    xytext=(0, 0), textcoords="offset points",
                    fontsize=8, ha="center", va="bottom",
                )

    chrom_boundaries = []
    for c in range(1, 23):
        if c in chrom_sorted:
            chrom_boundaries.append(np.min(x[chrom_sorted == c]))
    chrom_boundaries.append(x[-1])

    chrom_midpoints = [
        (chrom_boundaries[i] + chrom_boundaries[i + 1]) / 2
        for i in range(len(chrom_boundaries) - 1)
    ]

    ax.set_xticks(chrom_boundaries, minor=False)
    ax.set_xticklabels([])
    ax.set_xticks(chrom_midpoints, minor=True)
    ax.set_xticklabels(range(1, 23), minor=True, fontsize=6)
    ax.axhline(log10p_height, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel(r"-$\log_{10}$(p)")
    return ax


def cis_trans_miami(
    df,
    chrom_col="CHROM",
    cis_z_col="CIS_Z",
    trans_z_col="TRANS_Z",
    yclip=120,
    ylim=130,
    pvalue_threshold=1e-5,
    annot_dict=None,
    figsize=(9, 3),
):
    """Miami plot with cis (top) and trans (bottom, inverted) PWAS z-scores."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects

    if annot_dict is None:
        annot_dict = {}

    chrom_breaks = [0] + df.groupby(chrom_col).size().cumsum().tolist()
    chrom_centers = [
        (chrom_breaks[i] + chrom_breaks[i + 1]) / 2
        for i in range(len(chrom_breaks) - 1)
    ]

    abs_z_threshold = stats.norm.isf(pvalue_threshold)
    cis_absz = np.abs(df[cis_z_col]).clip(0, yclip)
    trans_absz = np.abs(df[trans_z_col]).clip(0, yclip)

    cis_signif = cis_absz > abs_z_threshold
    trans_signif = trans_absz > abs_z_threshold
    both_signif = cis_signif & trans_signif
    cis_only = cis_signif & ~trans_signif
    trans_only = ~cis_signif & trans_signif

    fig, (cis_ax, trans_ax) = plt.subplots(2, 1, sharex=True, figsize=figsize, dpi=150)

    for x, y, color, size, alpha in [
        (np.arange(len(df)), cis_absz, "gray", 5, 0.6),
        (np.arange(len(df))[cis_only], cis_absz[cis_only], "red", 10, 0.5),
        (np.arange(len(df))[both_signif], cis_absz[both_signif], "purple", 30, 0.8),
    ]:
        cis_ax.scatter(x, y, color=color, s=size, alpha=alpha, edgecolor="none")
    cis_ax.set_ylabel(r"Cis $|Z|$")
    cis_ax.set_ylim(0, ylim)

    for x, y, color, size, alpha in [
        (np.arange(len(df)), trans_absz, "gray", 5, 0.1),
        (np.arange(len(df))[trans_only], trans_absz[trans_only], "blue", 10, 0.5),
        (np.arange(len(df))[both_signif], trans_absz[both_signif], "purple", 30, 0.8),
    ]:
        trans_ax.scatter(x, y, color=color, s=size, alpha=alpha, edgecolor="none")
    trans_ax.set_ylabel("Trans $|Z|$")
    trans_ax.invert_yaxis()
    trans_ax.set_ylim(ylim, 0)
    trans_ax.spines["bottom"].set_visible(False)
    trans_ax.spines["top"].set_visible(True)
    trans_ax.xaxis.tick_top()
    trans_ax.xaxis.set_tick_params(pad=-1)

    trans_ax.set_xticks(chrom_breaks)
    trans_ax.set_xticklabels([])
    for i, center in enumerate(chrom_centers):
        trans_ax.text(
            center, 1.13, str(i + 1), fontsize=8,
            ha="center", va="top", transform=trans_ax.get_xaxis_transform(),
        )

    trans_ax.set_xlabel("Chromosome", fontsize=10, labelpad=2, x=-0.02)
    trans_ax.xaxis.set_label_position("top")

    for pid, where in annot_dict:
        xloc = df.index.get_loc(pid)
        color = {"both": "purple", "cis": "red", "trans": "blue"}[where]

        if where == "both":
            text_ax = trans_ax
            yloc = trans_absz.values[xloc]
        elif where == "cis":
            text_ax = cis_ax
            yloc = cis_absz.values[xloc]
        else:
            text_ax = trans_ax
            yloc = trans_absz.values[xloc]

        text_ax.text(
            xloc + annot_dict[(pid, where)]["x"] + 20,
            yloc + annot_dict[(pid, where)]["y"],
            annot_dict[(pid, where)]["text"],
            fontsize=8, ha="left", va="center", color=color, fontstyle="italic",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="w")],
        )

    return fig, (cis_ax, trans_ax)
