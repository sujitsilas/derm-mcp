"""`skin.enrich.*` — over-representation, GSEA, TF activity, signature scoring.

Two design choices worth stating:

- `exclude_terms` is a *parameter*, never a hardcoded list. The reference
  notebook filters 28 GO terms out of every result; some of those are defensible
  in skin (neuronal, cartilage), others are not (`Epithelial To Mesenchymal
  Transition` is plausibly real in wound fibroblasts). Baking it in would
  propagate one project's judgement call to every user, so it ships as the
  opt-in preset `skin_irrelevant_v1` and every dropped term is reported.
- Fold enrichment is `(k/N) / (n/M_BG)` with `M_BG` configurable and reported,
  because it materially changes the number.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import knowledge as K
from .. import registry
from ..config import CONFIG
from ..errors import BadParam, NetworkUnavailable, NotFound
from ..memory import store
from ._base import Ctx, tool

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("broad_biology", "coherent_programs", "signaling", "immune_specific",
                  "tf_activity", "metabolism", "cell_identity", "disease")


def _clean_term(t: str) -> str:
    s = str(t).split("(")[0].strip().replace("_", " ")
    return (s[0].upper() + s[1:]) if s else s


def _resolve_exclusions(exclude_terms: list[str] | None,
                        exclude_preset: str) -> tuple[list[str], str]:
    terms = list(exclude_terms or [])
    if exclude_preset:
        terms += K.exclude_preset(exclude_preset)
    return list(dict.fromkeys(terms)), exclude_preset


def _require_network(what: str) -> None:
    if CONFIG.offline:
        raise NetworkUnavailable(
            f"{what} needs network access and the server was started with --offline",
            remedy=("Use skin.enrich.score_signature with a shipped gene set from "
                    "knowledge/genesets.yaml, which works offline, or restart without "
                    "--offline."),
            suggested_tool="skin.enrich.score_signature",
        )


@tool("skin.enrich.list_libraries", category="enrich",
      summary="Curated enrichment-library recommendations for a question type.")
def list_libraries(question_type: str = "broad_biology", organism: str = "mouse",
                   project_id: str = "", dry_run: bool = False, seed: int = 0,
                   *, ctx: Ctx) -> None:
    """Which gene-set library should you use? Returns curated candidates with guidance.

    Library names in the shipped catalogue were verified against the live Enrichr
    index. Note that ImmuneSigDB is NOT an Enrichr library under any name; for
    that one the recommendation routes through gseapy.Msigdb instead.

    Args:
        question_type: "broad_biology", "coherent_programs", "signaling",
            "immune_specific", "tf_activity", "metabolism", "cell_identity",
            or "disease".
        organism: "mouse" or "human". Filters the organism-specific entries.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    if question_type not in QUESTION_TYPES:
        raise BadParam(f"question_type must be one of {list(QUESTION_TYPES)}")
    org = K._check_organism(organism)
    cat = K.enrich_libraries()
    q = cat["question_types"][question_type]

    def keep(name: str) -> bool:
        n = name.lower()
        if "mouse" in n:
            return org == "mouse"
        if "human" in n:
            return org == "human"
        return True

    rec = [x for x in q.get("recommended", []) if keep(x)]
    ctx.summary = {
        "question_type": question_type, "organism": org,
        "recommended": rec or q.get("recommended", []),
        "alternatives": [x for x in q.get("alternatives", []) if keep(x)],
        "mouse_native": q.get("mouse_native"),
        "msigdb_only": q.get("msigdb_only"),
        "prefer": q.get("prefer"),
        "why": q.get("why"),
        "caveats": cat["caveats"],
        "exclusion_presets": {k: v["description"][:180]
                              for k, v in cat["exclude_presets"].items()},
    }
    ctx.suggest("skin.enrich.ora", "skin.enrich.gsea")


@tool("skin.enrich.ora", category="enrich", needs_network=True,
      summary="Over-representation of a DE result's up/down gene sets (gseapy.enrichr).")
def ora(de_run_id: str, label: str, library: str = "GO_Biological_Process_2025",
        direction: str = "both", top_n_terms: int = 5, fdr: float = 0.05, lfc: float = 0.5,
        exclude_terms: list[str] | None = None, exclude_preset: str = "",
        background_size: int = 15000, organism: str = "", project_id: str = "",
        dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Run Enrichr over-representation on the up and down gene sets separately.

    Fold enrichment = (k/N) / (n/M_BG) with M_BG = `background_size`. That
    denominator materially changes the number, so it is reported alongside.

    Args:
        de_run_id: run_id from skin.de.pseudobulk or skin.de.wilcoxon.
        label: Which cell label's DE table to enrich.
        library: Enrichr library name. skin.enrich.list_libraries recommends one.
        direction: "up", "down", or "both".
        top_n_terms: Terms to keep per direction.
        fdr: Adjusted-p cutoff for calling a gene significant.
        lfc: Absolute log2FC cutoff for calling a gene significant.
        exclude_terms: Terms to drop. Every dropped term is reported.
        exclude_preset: Named preset from knowledge/enrich_libraries.yaml, e.g.
            "skin_irrelevant_v1". OFF by default — it is dataset-specific
            judgement, not a general truth, and it is recorded in the figure
            metadata whenever used.
        background_size: M_BG for the fold-enrichment denominator.
        organism: "mouse" or "human". Defaults to the dataset's organism.
        project_id: Defaults to the active project.
        dry_run: Report the gene-set sizes without calling Enrichr.
        seed: RNG seed.
    """
    import pandas as pd

    if direction not in ("up", "down", "both"):
        raise BadParam("direction must be up|down|both")
    run = store.get_run(ctx.project_id, de_run_id)
    if run is None:
        raise NotFound(f"unknown de_run_id {de_run_id!r}",
                       remedy="skin.memory.brief lists recent runs.")
    tables = run["result"].get("tables", {})
    if label not in tables:
        raise NotFound(f"label {label!r} is not in this DE run",
                       remedy=f"Labels available: {sorted(tables)}")

    contrast = run["params"].get("contrast", ["group_a", "group_b"])
    a, b = str(contrast[0]), str(contrast[1])
    org = K._check_organism(organism or registry.get_organism(
        registry.load(ctx.project_id, run["dataset_id"])))

    df = pd.read_csv(tables[label])
    sig = df[df["padj"] < fdr]
    sets: dict[str, list[str]] = {}
    if direction in ("up", "both"):
        sets[a] = sig.loc[sig["lfc"] > lfc, "gene"].astype(str).tolist()
    if direction in ("down", "both"):
        sets[b] = sig.loc[sig["lfc"] < -lfc, "gene"].astype(str).tolist()

    excl, preset_name = _resolve_exclusions(exclude_terms, exclude_preset)
    ctx.code = (
        "import gseapy as gp\n"
        f"sig = de[de['padj'] < {fdr}]\n"
        f"gene_sets = {{'{a}': sig[sig.lfc > {lfc}].gene.tolist(),\n"
        f"              '{b}': sig[sig.lfc < -{lfc}].gene.tolist()}}\n"
        f"enr = gp.enrichr(gene_list=glist, gene_sets={library!r}, "
        f"organism={org!r}, outdir=None)\n"
        f"# fold enrichment = (k/N) / (n/{background_size})\n"
    )
    if dry_run:
        ctx.summary = {"gene_set_sizes": {k: len(v) for k, v in sets.items()},
                       "library": library, "organism": org,
                       "n_exclusions": len(excl), "exclusion_preset": preset_name}
        return
    _require_network("Enrichr")

    try:
        import gseapy as gp
    except ImportError as e:
        from ..errors import DependencyMissing

        raise DependencyMissing("gseapy is not installed", remedy="uv pip install gseapy") from e

    rows, dropped_terms, per_direction = [], [], {}
    for dirname, glist in sets.items():
        per_direction[dirname] = {"n_genes": len(glist)}
        if len(glist) < 5:
            per_direction[dirname]["skipped"] = "fewer than 5 significant genes"
            ctx.warn(f"{label}/{dirname}: only {len(glist)} significant genes; ORA needs more "
                     f"to be meaningful. Consider skin.enrich.gsea, which uses the full "
                     f"ranked list and no cutoff.")
            continue
        try:
            enr = gp.enrichr(gene_list=glist, gene_sets=library, organism=org, outdir=None)
            res = enr.res2d.sort_values("Adjusted P-value").copy()
        except Exception as e:  # noqa: BLE001 - network/service errors are expected
            ctx.warn(f"{label}/{dirname}: Enrichr failed ({type(e).__name__}: {str(e)[:120]})")
            per_direction[dirname]["error"] = str(e)[:160]
            continue

        res["term_clean"] = res["Term"].map(_clean_term)
        if excl:
            low = [e.lower() for e in excl]
            hit = res["term_clean"].str.lower().apply(
                lambda t, _low=low: any(e in t for e in _low))
            dropped_terms += res.loc[hit, "term_clean"].tolist()
            res = res[~hit]
        top = res.head(top_n_terms)
        N = len(glist)
        for _, r in top.iterrows():
            k, n = (int(x) for x in str(r.get("Overlap", "1/1")).split("/"))
            fe = (k / N) / (n / background_size) if N and n else 1.0
            rows.append({"pathway_clean": r["term_clean"],
                         "padj": float(r["Adjusted P-value"]),
                         "FoldEnrichment": round(fe, 1), "Count": k,
                         "directionality": dirname,
                         "genes": str(r.get("Genes", ""))[:120]})
        per_direction[dirname]["n_terms"] = int(len(res))
        per_direction[dirname]["n_kept"] = int(len(top))

    if not rows:
        ctx.warn(f"No enriched terms survived for {label}. Nothing to plot.")

    out = pd.DataFrame(rows)
    run_id = store.new_id("enr", 8)
    p = ctx.tabledir() / f"enrichment_{run_id}.csv"
    out.to_csv(p, index=False)
    ctx.add_artifact("table", p, caption=f"ORA {label} ({library})",
                     params={"library": library, "exclusion_preset": preset_name,
                             "excluded_terms": excl, "background_size": background_size})

    if excl:
        ctx.warn(f"{len(set(dropped_terms))} terms were dropped by "
                 f"{preset_name or 'exclude_terms'}: {sorted(set(dropped_terms))[:6]}. "
                 f"This exclusion is recorded in the result and will appear in the caption "
                 f"metadata of any figure built from it.")

    store.record_run(ctx.project_id, run_id, "enrich", run["dataset_id"],
                     {"de_run_id": de_run_id, "label": label, "library": library,
                      "direction": direction, "fdr": fdr, "lfc": lfc,
                      "background_size": background_size, "exclusion_preset": preset_name,
                      "excluded_terms": excl, "contrast": [a, b], "method": "ora"},
                     {"table": str(p), "n_rows": int(len(out)),
                      "directions": list(sets.keys())})

    ctx.summary = {
        "run_id": run_id, "method": "ora", "label": label, "library": library,
        "organism": org, "background_size": background_size,
        "per_direction": per_direction,
        "top_terms": out.sort_values("padj").head(10).to_dict("records") if len(out) else [],
        "exclusion_preset": preset_name or None,
        "n_terms_excluded": len(set(dropped_terms)),
        "table_path": str(p),
    }
    ctx.suggest("skin.plot.enrichment_tile", "skin.enrich.gsea", "skin.plot.de_panel")


@tool("skin.enrich.gsea", category="enrich", needs_network=True,
      summary="GSEA prerank on the full ranked list. Preferred over ORA when power allows.")
def gsea(de_run_id: str, label: str, library: str = "MSigDB_Hallmark_2020",
         ranking: str = "stat", min_size: int = 10, max_size: int = 500,
         permutations: int = 1000, organism: str = "", project_id: str = "",
         dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Rank-based enrichment over the whole DE result, with no arbitrary cutoff.

    Prefer this over ORA when the DE has adequate power: ORA throws away the
    ranking and its answer depends on where you put the FDR/LFC thresholds.

    Args:
        de_run_id: run_id from a skin.de.* tool.
        label: Which cell label's DE table to use.
        library: Gene-set library, or a path to a local .gmt.
        ranking: "stat" (Wald statistic, preferred), "lfc", or "signed_logp".
        min_size: Minimum gene-set size.
        max_size: Maximum gene-set size.
        permutations: Permutations for the null.
        organism: "mouse" or "human". Defaults to the dataset's organism.
        project_id: Defaults to the active project.
        dry_run: Report the ranking without running.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd

    run = store.get_run(ctx.project_id, de_run_id)
    if run is None:
        raise NotFound(f"unknown de_run_id {de_run_id!r}")
    tables = run["result"].get("tables", {})
    if label not in tables:
        raise NotFound(f"label {label!r} not in this run", remedy=f"Available: {sorted(tables)}")
    df = pd.read_csv(tables[label])
    org = K._check_organism(organism or registry.get_organism(
        registry.load(ctx.project_id, run["dataset_id"])))

    if ranking == "stat" and "stat" in df.columns:
        rank = df.set_index("gene")["stat"]
    elif ranking == "signed_logp":
        rank = (np.sign(df["lfc"]) * -np.log10(df["padj"].clip(lower=1e-300)))
        rank.index = df["gene"]
    else:
        rank = df.set_index("gene")["lfc"]
        if ranking == "stat":
            ctx.warn("no 'stat' column in this DE table (cell-wise runs have none); "
                     "ranked by LFC instead.")
    rank = rank.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    ctx.code = (f"import gseapy as gp\n"
                f"rank = de.set_index('gene')[{ranking!r}].sort_values(ascending=False)\n"
                f"pre = gp.prerank(rnk=rank, gene_sets={library!r}, "
                f"min_size={min_size}, max_size={max_size}, permutation_num={permutations}, "
                f"seed={seed}, outdir=None)\n")
    if dry_run:
        ctx.summary = {"n_genes_ranked": int(len(rank)), "ranking": ranking,
                       "library": library, "organism": org}
        return
    _require_network("GSEA gene-set download")

    import gseapy as gp

    gene_sets: Any = library
    if org == "mouse" and library == "MSigDB_Hallmark_2020":
        try:
            gene_sets = gp.Msigdb().get_gmt(category="mh.all", dbver="2024.1.Mm")
            ctx.warn("Used the mouse-native Hallmark GMT (mh.all, 2024.1.Mm) rather than "
                     "Enrichr's human Hallmark, which would have mapped your symbols onto "
                     "human orthologs first.")
        except Exception as e:  # noqa: BLE001
            ctx.warn(f"mouse-native Hallmark fetch failed ({e}); using {library} with "
                     f"Enrichr's internal ortholog mapping.")

    pre = gp.prerank(rnk=rank, gene_sets=gene_sets, min_size=min_size, max_size=max_size,
                     permutation_num=permutations, seed=seed, outdir=None, threads=2)
    res = pre.res2d.copy()
    for c in ("NES", "FDR q-val", "NOM p-val"):
        if c in res:
            res[c] = pd.to_numeric(res[c], errors="coerce")
    res = res.sort_values("FDR q-val")

    run_id = store.new_id("gsea", 8)
    p = ctx.tabledir() / f"gsea_{run_id}.csv"
    res.to_csv(p, index=False)
    ctx.add_artifact("table", p, caption=f"GSEA {label} ({library})")

    sig = res[res["FDR q-val"] < 0.25]
    store.record_run(ctx.project_id, run_id, "enrich", run["dataset_id"],
                     {"de_run_id": de_run_id, "label": label, "library": library,
                      "ranking": ranking, "method": "gsea"},
                     {"table": str(p), "n_sig": int(len(sig))})
    ctx.summary = {
        "run_id": run_id, "method": "gsea", "label": label, "library": library,
        "n_gene_sets": int(len(res)), "n_sig_fdr25": int(len(sig)),
        "top_up": sig.nlargest(6, "NES")[["Term", "NES", "FDR q-val"]].to_dict("records"),
        "top_down": sig.nsmallest(6, "NES")[["Term", "NES", "FDR q-val"]].to_dict("records"),
        "table_path": str(p),
    }
    ctx.suggest("skin.plot.enrichment_tile", "skin.enrich.score_signature")


@tool("skin.enrich.score_signature", category="enrich",
      summary="Score a gene signature per cell (score_genes / AUCell / ssGSEA).")
def score_signature(dataset_id: str, name: str, genes: list[str] | None = None,
                    method: str = "score_genes", label: str = "", project_id: str = "",
                    dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Score one signature per cell and write it to obs. Works offline.

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        name: Signature name from knowledge/genesets.yaml, or a name for your
            own list passed in `genes`.
        genes: Explicit gene list. Omit to use the shipped signature `name`.
        method: "score_genes" (scanpy, default), "aucell", or "ssgsea" (both decoupler).
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report gene coverage without scoring.
        seed: RNG seed.
    """
    import scanpy as sc

    if method not in ("score_genes", "aucell", "ssgsea"):
        raise BadParam("method must be score_genes|aucell|ssgsea")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    gl = list(genes) if genes else K.get_signature(name, organism)
    present = K.present(adata, gl)

    ctx.code = (f"sc.tl.score_genes(adata, {present[:10]!r}, score_name='score_{name}',\n"
                f"                  use_raw=False, random_state={seed})\n")
    if dry_run:
        ctx.summary = {"name": name, "n_genes": len(gl), "n_present": len(present)}
        return
    if len(present) < 3:
        raise BadParam(f"only {len(present)} of {len(gl)} signature genes are in var_names",
                       remedy="The object may be subset to HVGs. Score on the full gene space.")
    if len(present) < 0.5 * len(gl):
        ctx.warn(f"only {len(present)}/{len(gl)} genes present for {name!r}; the score is "
                 f"based on a partial signature.")
    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")
        ctx.warn("X was z-scaled; restored layers['lognorm'] for scoring.")

    col = f"score_{name}"
    if method == "score_genes":
        sc.tl.score_genes(adata, present, score_name=col, use_raw=False, random_state=seed)
    else:
        import decoupler as dc
        import pandas as pd

        net = pd.DataFrame({"source": name, "target": present, "weight": 1.0})
        fn = dc.mt.aucell if method == "aucell" else dc.mt.gsva
        fn(adata, net, verbose=False)
        key = next((k for k in adata.obsm if k.startswith("score_") or "aucell" in k.lower()
                    or "gsva" in k.lower()), None)
        if key is None:
            raise BadParam(f"decoupler {method} produced no score matrix")
        adata.obs[col] = adata.obsm[key][name].to_numpy()

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="enrich.score_signature",
                         params={"name": name, "method": method, "n_genes": len(present)},
                         label=label)
    ctx.dataset_id = dsid
    s = adata.obs[col]
    ctx.summary = {"dataset_id": dsid, "score_column": col, "method": method,
                   "n_genes_used": len(present), "n_genes_requested": len(gl),
                   "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
    ctx.suggest("skin.plot.score_umap_grid", "skin.plot.state_space",
                "skin.enrich.score_panel")


@tool("skin.enrich.score_panel", category="enrich",
      summary="Score a whole shipped panel of signatures at once. Works offline.")
def score_panel(dataset_id: str, panel: str = "skin_wound_v1", label: str = "",
                project_id: str = "", dry_run: bool = False, seed: int = 0,
                *, ctx: Ctx) -> None:
    """Score every signature in a shipped panel in one call.

    `skin_wound_v1` covers glycolysis, OXPHOS, hypoxia, type-I and type-II IFN,
    cGAS-STING, inflammation, resolution/repair, LAM/SAM, phagocytosis,
    efferocytosis, ECM remodeling and proliferation.

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        panel: "skin_wound_v1", "keratinocyte_v1", or "fibroblast_v1".
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report coverage per signature.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    sets = K.get_panel(panel, organism)
    coverage = {n: (len(K.present(adata, g)), len(g)) for n, g in sets.items()}

    ctx.code = ("import scanpy as sc\n"
                "from skinmcp import knowledge as K\n"
                f"PANEL = K.get_panel({panel!r}, {organism!r})\n"
                "for name, genes in PANEL.items():\n"
                "    present = [g for g in genes if g in adata.var_names]\n"
                "    if len(present) >= 3:\n"
                "        sc.tl.score_genes(adata, present, score_name=f'score_{name}',\n"
                f"                          use_raw=False, random_state={seed})\n")
    if dry_run:
        ctx.summary = {"panel": panel, "n_signatures": len(sets),
                       "coverage": {k: f"{v[0]}/{v[1]}" for k, v in coverage.items()}}
        return

    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")

    written, thin = [], []
    for name, gl in sets.items():
        present = K.present(adata, gl)
        if len(present) < 3:
            thin.append(f"{name} ({len(present)}/{len(gl)})")
            continue
        sc.tl.score_genes(adata, present, score_name=f"score_{name}", use_raw=False,
                          random_state=seed)
        written.append(f"score_{name}")
    if thin:
        ctx.warn(f"skipped {len(thin)} signatures with fewer than 3 genes present: {thin[:6]}")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="enrich.score_panel",
                         params={"panel": panel, "n_scored": len(written)}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "panel": panel, "score_columns": written,
                   "coverage": {k: f"{v[0]}/{v[1]}" for k, v in coverage.items()}}
    ctx.suggest("skin.plot.score_umap_grid", "skin.plot.state_space")


@tool("skin.enrich.score_hallmark", category="enrich", needs_network=True,
      summary="Score MSigDB Hallmark sets per cell, mouse-native where it matters.")
def score_hallmark(dataset_id: str, sets: list[str] | None = None, organism: str = "",
                   label: str = "", project_id: str = "", dry_run: bool = False,
                   seed: int = 0, *, ctx: Ctx) -> None:
    """Score MSigDB Hallmark gene sets per cell.

    For mouse this fetches the mouse-native GMT (`mh.all`, 2024.1.Mm) rather than
    letting Enrichr map symbols onto human orthologs.

    Args:
        dataset_id: Handle or label.
        sets: Hallmark set names, e.g. ["HALLMARK_GLYCOLYSIS",
            "HALLMARK_HYPOXIA"]. Empty = a default metabolic/inflammatory subset.
        organism: "mouse" or "human". Defaults to the dataset's organism.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    org = K._check_organism(organism or registry.get_organism(adata))
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    want = sets or ["HALLMARK_GLYCOLYSIS", "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
                    "HALLMARK_HYPOXIA", "HALLMARK_INFLAMMATORY_RESPONSE",
                    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
                    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
                    "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_IL6_JAK_STAT3_SIGNALING"]
    ctx.code = ('import gseapy as gp\n'
                'gmt = gp.Msigdb().get_gmt(category=%r, dbver=%r)\n'
                'for s in SETS:\n'
                '    sc.tl.score_genes(adata, gmt[s], score_name=s, use_raw=False)\n'
                % (("mh.all", "2024.1.Mm") if org == "mouse" else ("h.all", "2024.1.Hs")))
    if dry_run:
        ctx.summary = {"organism": org, "n_sets": len(want), "sets": want}
        return
    _require_network("MSigDB Hallmark download")

    import gseapy as gp

    cat, ver = ("mh.all", "2024.1.Mm") if org == "mouse" else ("h.all", "2024.1.Hs")
    gmt = gp.Msigdb().get_gmt(category=cat, dbver=ver)
    if not gmt:
        raise NetworkUnavailable("MSigDB returned no gene sets",
                                 remedy="Use skin.enrich.score_panel, which ships offline.")
    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")

    written, missing = [], []
    for s in want:
        gl = gmt.get(s)
        if not gl:
            missing.append(s)
            continue
        present = K.present(adata, gl)
        if len(present) < 5:
            missing.append(f"{s} (only {len(present)} genes present)")
            continue
        sc.tl.score_genes(adata, present, score_name=s, use_raw=False, random_state=seed)
        written.append(s)
    if missing:
        ctx.warn(f"{len(missing)} sets could not be scored: {missing[:6]}")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="enrich.score_hallmark",
                         params={"sets": written, "organism": org, "gmt": f"{cat}@{ver}"},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "organism": org, "gmt": f"{cat} {ver}",
                   "scored": written, "not_scored": missing[:8]}
    ctx.suggest("skin.plot.state_space", "skin.plot.score_umap_grid")


@tool("skin.enrich.tf_activity", category="enrich", needs_network=True,
      summary="TF activity from a DE result via decoupler + CollecTRI (preferred over ORA).")
def tf_activity(de_run_id: str, label: str, organism: str = "", top_n: int = 20,
                project_id: str = "", dry_run: bool = False, seed: int = 0,
                *, ctx: Ctx) -> None:
    """Infer transcription-factor activity from the signed DE statistics.

    Preferred over ORA against TF-target libraries: this uses the effect sizes
    and the regulon signs rather than a thresholded gene list.

    Args:
        de_run_id: run_id from a skin.de.* tool.
        label: Which cell label's DE table to use.
        organism: "mouse" or "human". Defaults to the dataset's organism.
        top_n: TFs to report per direction.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd

    run = store.get_run(ctx.project_id, de_run_id)
    if run is None:
        raise NotFound(f"unknown de_run_id {de_run_id!r}")
    tables = run["result"].get("tables", {})
    if label not in tables:
        raise NotFound(f"label {label!r} not in this run", remedy=f"Available: {sorted(tables)}")
    org = K._check_organism(organism or registry.get_organism(
        registry.load(ctx.project_id, run["dataset_id"])))
    ctx.code = ("import decoupler as dc\n"
                f"net = dc.op.collectri(organism={org!r})\n"
                "acts = dc.mt.ulm(data=stat_matrix, net=net)\n")
    if dry_run:
        ctx.summary = {"organism": org, "method": "decoupler ULM over CollecTRI"}
        return
    _require_network("CollecTRI download")

    import decoupler as dc

    df = pd.read_csv(tables[label]).dropna(subset=["padj"])
    stat = df["stat"] if "stat" in df.columns else (
        np.sign(df["lfc"]) * -np.log10(df["padj"].clip(lower=1e-300)))
    mat = pd.DataFrame([stat.to_numpy()], columns=df["gene"].astype(str), index=[label])

    net = dc.op.collectri(organism=org)
    res = dc.mt.ulm(data=mat, net=net, verbose=False)
    scores = res[0] if isinstance(res, tuple) else res
    pvals = res[1] if isinstance(res, tuple) and len(res) > 1 else None
    s = scores.iloc[0].sort_values(ascending=False)

    run_id = store.new_id("tf", 8)
    p = ctx.tabledir() / f"tf_activity_{run_id}.csv"
    out = pd.DataFrame({"tf": s.index, "score": s.to_numpy()})
    if pvals is not None:
        out["pval"] = pvals.iloc[0].reindex(s.index).to_numpy()
    out.to_csv(p, index=False)
    ctx.add_artifact("table", p, caption=f"TF activity (CollecTRI/ULM) {label}")
    store.record_run(ctx.project_id, run_id, "enrich", run["dataset_id"],
                     {"de_run_id": de_run_id, "label": label, "method": "collectri_ulm"},
                     {"table": str(p)})
    ctx.summary = {"run_id": run_id, "method": "decoupler ULM / CollecTRI", "organism": org,
                   "top_activated": out.head(top_n).to_dict("records"),
                   "top_repressed": out.tail(top_n).iloc[::-1].to_dict("records"),
                   "table_path": str(p)}
    ctx.suggest("skin.plot.enrichment_tile", "skin.enrich.gsea")
