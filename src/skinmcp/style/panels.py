"""Figure panels ported from the reference notebook (cell 10, 22, 35, 46, 47, 127).

Behavioural fidelity matters here: these produce the lab's existing figures, so
the layout constants, the adjust_text tuning, and the tile colour maths are
reproduced exactly rather than re-derived.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from . import palettes as P
from .rcparams import savefig, style

logger = logging.getLogger(__name__)


def slug(label: str) -> str:
    """Filesystem-safe slug. Handles MΦ, DC/Mono, Endo. (reference `_slug`)."""
    s = str(label).replace("φ", "phi").replace("Φ", "phi")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "cell"


def pick_labels(df_side: Any, n: int, ascending_lfc: bool = False) -> Any:
    """Half by smallest padj, half by largest |LFC|, deduped (reference cell 10)."""
    import pandas as pd

    if df_side.empty:
        return df_side.iloc[:0]
    half = max(n // 2, 1)
    by_sig = df_side.nsmallest(half, "padj")
    by_lfc = (df_side.nsmallest(half, "lfc") if ascending_lfc
              else df_side.nlargest(half, "lfc"))
    return pd.concat([by_sig, by_lfc]).drop_duplicates("gene").head(n)


# --------------------------------------------------------------------------- #
# volcano grid
# --------------------------------------------------------------------------- #

def volcano_grid(
    de_tables: dict[str, Any],
    out_path: Any,
    *,
    group_a: str = "Burn",
    group_b: str = "Sham",
    ncols: int | None = None,
    fdr: float = 0.05,
    lfc: float = 0.5,
    n_label: int = 9,
    must_label: Sequence[str] = (),
    highlight_genes: Sequence[str] = (),
    xlim: tuple[float, float] = (-8.0, 8.0),
    title_suffix: str = "",
    profile: str = "large",
) -> dict[str, Any]:
    """Grid of volcano panels, one per label.

    `de_tables` maps label -> DataFrame with columns `gene`, `lfc`, `padj`.
    Positive LFC is `group_a`. Returns the saved paths plus per-panel counts.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from adjustText import adjust_text

    labels = [k for k, v in de_tables.items() if v is not None and not v.empty]
    if not labels:
        raise ValueError("volcano_grid: no non-empty DE tables")

    a_col = P.CONDITION_COLORS.get(str(group_a).lower(), P.BURN)
    b_col = P.CONDITION_COLORS.get(str(group_b).lower(), P.SHAM)
    highlight = set(highlight_genes)
    counts: dict[str, dict[str, int]] = {}

    with style(profile) as A:
        nc = ncols or min(4, len(labels))
        nr = int(np.ceil(len(labels) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 7 * nr))
        axes_flat = np.atleast_1d(axes).flatten()
        n_draw = min(n_label, 12)

        for i, lab in enumerate(labels):
            ax = axes_flat[i]
            df = de_tables[lab].copy()
            df["nlp"] = -np.log10(df["padj"].clip(lower=1e-300))
            sig_a = (df["padj"] < fdr) & (df["lfc"] > lfc)
            sig_b = (df["padj"] < fdr) & (df["lfc"] < -lfc)
            ns = ~sig_a & ~sig_b

            ax.scatter(df.loc[ns, "lfc"], df.loc[ns, "nlp"], c=P.NS_GREY, s=5,
                       alpha=0.4, linewidths=0, rasterized=True)
            ax.scatter(df.loc[sig_b, "lfc"], df.loc[sig_b, "nlp"], c=b_col, s=10,
                       alpha=0.8, linewidths=0, rasterized=True)
            ax.scatter(df.loc[sig_a, "lfc"], df.loc[sig_a, "nlp"], c=a_col, s=10,
                       alpha=0.8, linewidths=0, rasterized=True)

            ax.axhline(-np.log10(fdr), color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)
            ax.axvline(lfc, color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)
            ax.axvline(-lfc, color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)

            df_sig = df[(df["padj"] < fdr) & (df["lfc"].abs() > lfc)]
            top_a = pick_labels(df_sig[df_sig.lfc > 0], n_draw, ascending_lfc=False)
            top_b = pick_labels(df_sig[df_sig.lfc < 0], n_draw, ascending_lfc=True)
            forced = df[df["gene"].isin(list(must_label))]
            label_df = pd.concat([top_a, top_b, forced]).drop_duplicates("gene")

            ax.set_xlim(*xlim)
            ymax = float(df["nlp"].max()) if len(df) else 1.0
            ax.set_ylim(-0.5, max(ymax * 1.15, 1.0))

            na, nb = int(sig_a.sum()), int(sig_b.sum())
            counts[lab] = {f"{group_a}_up": na, f"{group_b}_up": nb,
                           "n_tested": int(len(df))}
            t_a = ax.text(0.03, 0.97, f"{group_a}↑ {na}", transform=ax.transAxes,
                          fontsize=A["count_fs"], va="top", ha="left", color=a_col,
                          fontweight="bold")
            t_b = ax.text(0.03, 0.91, f"{group_b}↑ {nb}", transform=ax.transAxes,
                          fontsize=A["count_fs"], va="top", ha="left", color=b_col,
                          fontweight="bold")

            texts = []
            for _, row in label_df.iterrows():
                col = "red" if row["gene"] in highlight else "black"
                texts.append(ax.text(row["lfc"], -np.log10(max(row["padj"], 1e-300)),
                                     row["gene"], fontsize=A["annot_fs"], color=col,
                                     fontweight="bold", ha="center"))
            if texts:
                adjust_text(
                    texts, ax=ax, objects=[t_a, t_b],
                    arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.6),
                    expand=(1.3, 1.5), force_text=(0.6, 0.8), force_static=(0.2, 0.3),
                    force_pull=(0.05, 0.05), max_move=6, min_arrow_len=3,
                    only_move={"text": "xy", "static": "xy", "explode": "xy"},
                    ensure_inside_axes=True, time_lim=3.0,
                )

            ax.set_xlabel("Log$_2$ Fold Change", fontsize=28, fontweight="bold")
            ax.set_ylabel("$-$Log$_{10}$(padj)", fontsize=28, fontweight="bold")
            ax.set_title(f"{lab}{title_suffix}", fontsize=30, fontweight="bold")
            ax.grid(False)

        for ax in axes_flat[len(labels):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        paths = savefig(fig, out_path, hero=(len(labels) == 1))
        plt.close(fig)

    return {"paths": paths, "counts": counts, "n_panels": len(labels),
            "ncols": ncols or min(4, len(labels))}


# --------------------------------------------------------------------------- #
# enrichment tile
# --------------------------------------------------------------------------- #

def enrichment_tile(
    df: Any,
    out_path: Any,
    *,
    title: str = "",
    tile_size: float = 0.7,
    limit: int = 40,
    direction_order: Sequence[str] = ("Sham", "Burn"),
    profile: str = "large",
) -> dict[str, Any]:
    """Pathway × direction tile (reference `make_enrichment_tile`).

    `df` needs columns `pathway_clean`, `padj`, `FoldEnrichment`, `Count`,
    `directionality`. Fill is `-log10(padj)` on a blue→red map; alpha encodes
    gene Count; the bold number in each tile is fold enrichment.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if df is None or len(df) == 0:
        return {"paths": {}, "skipped": True, "reason": "no enriched terms to plot"}

    df = df.copy()
    df["logp"] = -np.log10(df["padj"].clip(lower=1e-300))
    avail = df["directionality"].unique().tolist()
    dir_order = [d for d in direction_order if d in avail] + \
                [d for d in avail if d not in direction_order]
    df["directionality"] = pd.Categorical(df["directionality"], categories=dir_order, ordered=True)
    df = df.sort_values(["directionality", "logp"], ascending=[True, False])

    pathways = df["pathway_clean"].unique().tolist()
    pathways_rev = pathways[::-1]
    pathway_to_y = {p: i for i, p in enumerate(pathways_rev)}
    n_rows, n_cols = len(pathways), len(dir_order)

    trunc = [p[:limit] + "..." if len(p) > limit else p for p in pathways]
    trunc_rev = trunc[::-1]
    max_len = max(len(p) for p in trunc)

    with style(profile) as A:
        fig_w = max_len * 0.12 + 1.0 + n_cols * tile_size + 2.5
        fig_h = 2.5 + n_rows * tile_size + 1.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = mcolors.LinearSegmentedColormap.from_list("br", ["#377EB8", "#E41A1C"])
        norm_lp = mcolors.Normalize(vmin=float(df["logp"].min()), vmax=float(df["logp"].max()))
        c_min, c_max = float(df["Count"].min()), float(df["Count"].max())

        def alpha_of(c: float) -> float:
            return 0.4 + 0.6 * (c - c_min) / (c_max - c_min) if c_max > c_min else 1.0

        lookup = {(r["directionality"], r["pathway_clean"]): r for _, r in df.iterrows()}
        for col_idx, grp in enumerate(dir_order):
            for pathway, row_idx in pathway_to_y.items():
                row = lookup.get((grp, pathway))
                if row is None:
                    continue
                ax.add_patch(patches.Rectangle(
                    (col_idx - 0.5, row_idx - 0.5), 1, 1,
                    color=cmap(norm_lp(row["logp"])), alpha=alpha_of(row["Count"]), ec=None))
                ax.text(col_idx, row_idx, f"{row['FoldEnrichment']}", ha="center",
                        va="center", color="black", fontsize=A["tile_value_fs"],
                        fontweight="bold")

        ax.set_aspect("equal")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(dir_order, rotation=45, fontsize=A["tile_tick_fs"],
                           color="black", fontweight="bold", ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(trunc_rev, fontsize=A["tile_tick_fs"], color="black")
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("black")
            sp.set_linewidth(1.2)
        ax.set_facecolor("white")
        ax.grid(False)
        if title:
            ax.set_title(title, fontsize=19, fontweight="bold", pad=12)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_lp)
        sm.set_array([])
        fig.canvas.draw()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.25, 0.018, pos.height * 0.5])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r"$-\log_{10}(p_{adj})$", fontsize=A["cbar_label_fs"],
                       fontweight="bold", labelpad=12)
        cbar.ax.tick_params(labelsize=A["cbar_tick_fs"])
        cbar.ax.yaxis.label.set_fontweight("bold")

        fig.canvas.draw()
        paths = savefig(fig, out_path, hero=True)
        plt.close(fig)

    return {"paths": paths, "n_terms": n_rows, "directions": list(dir_order)}


# --------------------------------------------------------------------------- #
# dotplot
# --------------------------------------------------------------------------- #

def cluster_gene_order(adata: Any, genes: Sequence[str], groupby: str,
                       group_order: Sequence[str], method: str = "average") -> list[str]:
    """Order genes by similarity of BOTH mean expression and % expressing.

    Reference cell 35: genes are blocked by the group that expresses them in the
    most cells, then hierarchically ordered within each block so neighbours on
    the dotplot are neighbours in profile space.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.cluster.hierarchy import leaves_list, linkage

    g = [x for x in genes if x in adata.var_names]
    if len(g) < 3:
        return g
    X = adata[:, g].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    grp = adata.obs[groupby].astype(str).values
    Xdf = pd.DataFrame(X, columns=g)

    M = Xdf.assign(_grp=grp).groupby("_grp").mean().reindex(group_order)
    F = (Xdf > 0).assign(_grp=grp).groupby("_grp").mean().reindex(group_order)

    def sc01(D: Any) -> Any:
        return ((D - D.min()) / (D.max() - D.min())).fillna(0)

    Mz, Fz = sc01(M), sc01(F)
    feat = pd.concat([Mz, Fz], axis=0)
    dom = Fz.values.argmax(0)

    ordered: list[str] = []
    for gi in range(len(group_order)):
        block = [g[j] for j in range(len(g)) if dom[j] == gi]
        if len(block) > 2:
            Z = linkage(feat[block].T.values, method=method, metric="euclidean",
                        optimal_ordering=True)
            block = [block[k] for k in leaves_list(Z)]
        ordered += block
    # Genes whose dominant group fell outside group_order still have to appear.
    ordered += [x for x in g if x not in ordered]
    return ordered


def dotplot(
    adata: Any,
    genes: Sequence[str] | dict[str, Sequence[str]],
    groupby: str,
    out_path: Any,
    *,
    cmap: str = "Reds",
    dot_max: float = 1.0,
    standard_scale: str | None = "var",
    dendrogram: bool = False,
    title: str = "",
    profile: str = "standard",
) -> dict[str, Any]:
    """Styled scanpy dotplot (§8.4)."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    flat = ([g for v in genes.values() for g in v] if isinstance(genes, dict) else list(genes))
    present = [g for g in flat if g in adata.var_names]
    if not present:
        raise ValueError("dotplot: none of the requested genes are in var_names")
    if isinstance(genes, dict):
        genes = {k: [g for g in v if g in adata.var_names] for k, v in genes.items()}
        genes = {k: v for k, v in genes.items() if v}
    else:
        genes = present

    n_groups = int(adata.obs[groupby].astype(str).nunique())
    with style(profile):
        axd = sc.pl.dotplot(
            adata, genes, groupby=groupby, use_raw=False,
            standard_scale=standard_scale, cmap=cmap, dot_max=dot_max,
            dendrogram=dendrogram,
            figsize=(0.34 * len(present) + 2, 0.5 * n_groups + 2),
            colorbar_title="Scaled\nexpression", size_title="% expressing",
            title=title, show=False,
        )
        main = axd["mainplot_ax"]
        main.tick_params(axis="x", labelsize=15)
        main.tick_params(axis="y", labelsize=18)
        for lbl in main.get_xticklabels():
            lbl.set_fontweight("bold")
        for lbl in main.get_yticklabels():
            lbl.set_fontweight("bold")
        if isinstance(genes, dict):
            for txt in main.figure.texts:
                txt.set_fontsize(17)
                txt.set_fontweight("bold")
        fig = plt.gcf()
        paths = savefig(fig, out_path)
        plt.close(fig)
    return {"paths": paths, "n_genes": len(present), "n_groups": n_groups}


# --------------------------------------------------------------------------- #
# UMAP helpers
# --------------------------------------------------------------------------- #

def style_umap_axis(ax: Any, title: str = "", label_fs: int = 24, title_fs: int = 20) -> None:
    """Reference cell 5 `style_umap`: bold labels, no ticks (coords are arbitrary)."""
    if title:
        ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP 1", fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=label_fs, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def legend_only(mapping: dict[str, str], out_path: Any, *, orientation: str = "vertical",
                title: str = "", ncol: int = 1, profile: str = "standard") -> dict[str, Any]:
    """Standalone legend figure, for panel assembly in Illustrator (cells 23/34)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with style(profile):
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                          markersize=13, label=k) for k, c in mapping.items()]
        n = len(handles)
        cols = ncol if orientation == "vertical" else max(1, n)
        fig = plt.figure(figsize=(3.2 * cols, 0.42 * (n / cols) + 1.0))
        fig.legend(handles=handles, loc="center", frameon=False, ncol=cols,
                   title=title or None, fontsize=15, title_fontsize=16)
        paths = savefig(fig, out_path)
        plt.close(fig)
    return {"paths": paths, "n_entries": len(mapping)}


def significance_stars(p: float) -> str:
    """Reference `stars` helper."""
    import numpy as np

    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
