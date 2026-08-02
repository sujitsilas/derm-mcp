"""`skin.atlas.*` — reference atlases, label transfer, ortholog mapping.

The honest position on skin atlases, surfaced in the tool returns rather than
buried in docs:

- **There is no mouse skin CellTypist model.** The default for mouse is
  marker-based annotation. Cross-species transfer is offered but flagged
  low-confidence, and `train_model` — building a model from the lab's own
  annotated mouse skin data — is the highest-value path and is documented as
  first-class.
- **`Adult_Human_Skin` is healthy adult skin.** It has no burn, wound, LAM or
  MDM states. Applied to wound data it will confidently map everything onto
  homeostatic labels. Any query whose metadata indicates injury or disease gets
  a `domain_shift_warning`.
- **Census version is pinned**, not `"latest"` — Census releases change cell
  counts, and an analysis must not silently shift underneath you.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from .. import knowledge as K
from .. import registry
from ..config import CONFIG
from ..errors import BadParam, DependencyMissing, NetworkUnavailable, NotFound
from ..memory import store
from ._base import Ctx, require_obs, tool

logger = logging.getLogger(__name__)

CELLTYPIST_MODELS_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"

#: obs values that mean "this is not healthy tissue".
INJURY_TOKENS = ("burn", "wound", "injur", "lesion", "disease", "tumor", "tumour",
                 "psoria", "derm", "scar", "infect", "inflam", "treated")

HEALTHY_TISSUE_MODELS = ("Adult_Human_Skin.pkl", "Fetal_Human_Skin.pkl")


def requires_network(what: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Degrade to the shipped snapshot with a warning under --offline."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            if CONFIG.offline:
                raise NetworkUnavailable(
                    f"{what} needs network access and the server is in --offline mode",
                    remedy=("Use skin.atlas.marker_lookup or skin.annotate.score_lineages, "
                            "which use the shipped marker snapshots and work air-gapped."),
                    suggested_tool="skin.annotate.score_lineages")
            return fn(*a, **kw)

        return wrapper

    return deco


def _http_json(url: str) -> Any:
    import requests

    r = requests.get(url, timeout=CONFIG.network_timeout_s)
    r.raise_for_status()
    return r.json()


def _detect_domain_shift(adata: Any) -> dict[str, Any]:
    """Does the query's metadata say this is not healthy tissue?"""
    hits = []
    for c in adata.obs.columns:
        try:
            vals = set(map(str, adata.obs[c].astype(str).unique()[:40]))
        except Exception:  # noqa: BLE001
            continue
        for v in vals:
            lv = v.lower()
            if any(tok in lv for tok in INJURY_TOKENS):
                hits.append(f"{c}={v}")
    return {"injury_like_metadata": sorted(set(hits))[:10], "shifted": bool(hits)}


@tool("skin.atlas.list_models", category="atlas", needs_network=True,
      summary="List CellTypist models, highlighting the skin-relevant ones.")
def list_models(filter_skin: bool = True, project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Fetch the live CellTypist model index.

    Args:
        filter_skin: Only show skin/immune-relevant models.
        project_id: Defaults to the active project.
        dry_run: Report the endpoint only.
        seed: Unused.
    """
    if dry_run:
        ctx.summary = {"url": CELLTYPIST_MODELS_URL}
        return
    fetch = requires_network("the CellTypist model index")(_http_json)
    try:
        data = fetch(CELLTYPIST_MODELS_URL)
    except NetworkUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        raise NetworkUnavailable(
            f"could not reach the CellTypist index: {type(e).__name__}",
            remedy="Check connectivity, or stay marker-based with "
                   "skin.annotate.score_lineages.",
            suggested_tool="skin.annotate.score_lineages") from e

    models = data.get("models", [])
    rows = [{"filename": m.get("filename"), "details": (m.get("details") or "")[:120],
             "version": m.get("version"), "date": m.get("date")} for m in models]
    if filter_skin:
        rows = [r for r in rows
                if any(k in (r["filename"] or "").lower() + (r["details"] or "").lower()
                       for k in ("skin", "immune", "pan_fetal", "healthy_adult"))]
    ctx.summary = {
        "n_models_total": len(models), "n_shown": len(rows), "models": rows[:20],
        "skin_models": {
            "Adult_Human_Skin.pkl": "34 cell types, healthy adult human skin "
                                    "(Reynolds et al., Science 2021).",
            "Fetal_Human_Skin.pkl": "14 types, developing human fetal skin.",
        },
        "mouse_note": ("There is NO mouse skin CellTypist model. For mouse: stay "
                       "marker-based (skin.annotate.score_lineages, the default), or "
                       "train your own with skin.atlas.train_model on an annotated "
                       "in-house object — that is the highest-value option and it "
                       "compounds across projects."),
    }
    ctx.suggest("skin.atlas.celltypist", "skin.atlas.train_model")


@tool("skin.atlas.celltypist", category="atlas", needs_network=True,
      summary="CellTypist label transfer. Human skin only; mouse needs ortholog mapping.")
def celltypist(dataset_id: str, model: str = "Adult_Human_Skin.pkl", majority_voting: bool = True,
               over_clustering: str = "", label: str = "", project_id: str = "",
               dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Annotate cells with a pretrained CellTypist model.

    For a mouse object this maps mouse symbols onto human orthologs first and
    flags the whole result as low-confidence cross-species transfer.

    Args:
        dataset_id: Handle or label. Must be log1p(CP10K) normalized.
        model: Model filename, e.g. "Adult_Human_Skin.pkl", "Immune_All_Low.pkl",
            or the name of a model you trained with skin.atlas.train_model.
        majority_voting: Refine per-cell calls over an over-clustering. Recommended.
        over_clustering: obs column to vote within. Empty = CellTypist's own.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan and the domain-shift check.
        seed: RNG seed.
    """
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    shift = _detect_domain_shift(adata)

    ctx.code = ("import celltypist\n"
                f"pred = celltypist.annotate(adata, model={model!r}, "
                f"majority_voting={majority_voting})\n"
                "adata = pred.to_adata()\n")
    if dry_run:
        ctx.summary = {"model": model, "organism": organism,
                       "domain_shift": shift, "cross_species": organism == "mouse"}
        return

    if model in HEALTHY_TISSUE_MODELS and shift["shifted"]:
        ctx.warn(
            f"domain_shift_warning: {model} was trained on HEALTHY tissue and has no burn, "
            f"wound, LAM or monocyte-derived states, but this query's metadata says "
            f"{shift['injury_like_metadata'][:4]}. It will confidently map injury states "
            f"onto homeostatic labels. Treat every call as a nearest-homeostatic-neighbour, "
            f"not an identity, and keep the marker-based labels as the record.")

    try:
        import celltypist as ct
    except ImportError as e:
        raise DependencyMissing(
            "celltypist is not installed",
            remedy='uv pip install "skin-mcp[atlas]". Marker-based annotation via '
                   "skin.annotate.score_lineages needs nothing extra.",
            suggested_tool="skin.annotate.score_lineages") from e
    if CONFIG.offline:
        raise NetworkUnavailable(
            "CellTypist needs to download the model and --offline is set",
            remedy="Use skin.annotate.score_lineages (marker-based, offline).",
            suggested_tool="skin.annotate.score_lineages")

    query = adata
    ortho_report = None
    if organism == "mouse":
        rep = K.ortholog_report(list(map(str, adata.var_names)), "mouse", "human")
        keep = [g for g in map(str, adata.var_names) if g in rep["mapped"]]
        query = adata[:, keep].copy()
        query.var_names = [rep["mapped"][g] for g in keep]
        query.var_names_make_unique()
        ortho_report = {"n_mapped": len(keep), "n_input": int(adata.n_vars),
                        "n_unmapped": len(rep["unmapped"])}
        ctx.warn(
            f"CROSS-SPECIES TRANSFER: {len(keep)} of {adata.n_vars} mouse genes were mapped "
            f"to human orthologs so a human model could be applied. There is no mouse skin "
            f"CellTypist model. Every label from this run is LOW CONFIDENCE and must be "
            f"reported as cross-species. Prefer skin.annotate.score_lineages, or build a "
            f"mouse model with skin.atlas.train_model.")

    if registry.get_x_state(query) != "lognorm":
        if "lognorm" in query.layers:
            query.X = query.layers["lognorm"].copy()
        else:
            ctx.warn("CellTypist expects log1p(CP10K) input; this object is "
                     f"{registry.get_x_state(query)!r}. Results may be unreliable.")

    kw: dict[str, Any] = {"model": model, "majority_voting": bool(majority_voting)}
    if over_clustering:
        require_obs(query, over_clustering)
        kw["over_clustering"] = over_clustering
    pred = ct.annotate(query, **kw)
    out = pred.to_adata()

    key = "celltypist_majority" if majority_voting and "majority_voting" in out.obs \
        else "predicted_labels"
    src = "majority_voting" if key == "celltypist_majority" else "predicted_labels"
    adata.obs["celltypist_label"] = np.asarray(out.obs[src].astype(str))
    if "conf_score" in out.obs:
        adata.obs["celltypist_conf"] = np.asarray(out.obs["conf_score"], dtype=float)

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="atlas.celltypist",
                         params={"model": model, "majority_voting": majority_voting,
                                 "cross_species": organism == "mouse"}, label=label)
    ctx.dataset_id = dsid
    vc = adata.obs["celltypist_label"].value_counts()
    conf = (round(float(adata.obs["celltypist_conf"].mean()), 3)
            if "celltypist_conf" in adata.obs else None)
    ctx.summary = {
        "dataset_id": dsid, "model": model, "obs_key": "celltypist_label",
        "n_labels": int(vc.size), "top_labels": vc.head(20).to_dict(),
        "mean_confidence": conf, "cross_species": organism == "mouse",
        "ortholog_mapping": ortho_report,
        "domain_shift_warning": shift if shift["shifted"] else None,
        "provenance_tag": ("REFERENCE-BASED (CellTypist"
                           + (", CROSS-SPECIES" if organism == "mouse" else "")
                           + (", HEALTHY-TISSUE MODEL ON INJURED QUERY"
                              if (model in HEALTHY_TISSUE_MODELS and shift["shifted"]) else "")
                           + ")"),
    }
    ctx.suggest("skin.annotate.marker_report", "skin.memory.record_annotation",
                "skin.atlas.train_model")


@tool("skin.atlas.train_model", category="atlas",
      summary="Train a CellTypist model on your own annotated object. The high-value path.")
def train_model(dataset_id: str, label_key: str, name: str, feature_selection: bool = True,
                n_jobs: int = 4, project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Build a reusable CellTypist model from an object you have already annotated.

    This is how a lab's accumulated annotation work becomes reusable, and for
    mouse skin it is the only route to reference-based annotation, because no
    public mouse skin CellTypist model exists.

    Args:
        dataset_id: Handle or label with trustworthy labels. Should be log1p(CP10K).
        label_key: obs column holding the labels to learn.
        name: Model name. Saved to {project_root}/models/{name}.pkl.
        feature_selection: Two-pass training that keeps only informative genes.
        n_jobs: Threads.
        project_id: Defaults to the active project.
        dry_run: Report class balance without training.
        seed: RNG seed.
    """
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    organism = registry.get_organism(adata)
    vc = adata.obs[label_key].astype(str).value_counts()
    ctx.code = (f"import celltypist\n"
                f"model = celltypist.train(adata, labels={label_key!r}, "
                f"feature_selection={feature_selection}, n_jobs={n_jobs})\n"
                f"model.write({name!r} + '.pkl')\n")
    if dry_run:
        ctx.summary = {"n_labels": int(vc.size), "class_balance": vc.head(25).to_dict(),
                       "organism": organism}
        return

    tiny = vc[vc < 20]
    if len(tiny):
        ctx.warn(f"{len(tiny)} classes have fewer than 20 cells ({list(tiny.index)[:6]}); "
                 f"they will be learned poorly and will be over-predicted on new data.")
    try:
        import celltypist as ct
    except ImportError as e:
        raise DependencyMissing("celltypist is not installed",
                                remedy='uv pip install "skin-mcp[atlas]"') from e

    if registry.get_x_state(adata) != "lognorm":
        if "lognorm" in adata.layers:
            adata.X = adata.layers["lognorm"].copy()
            registry.set_x_state(adata, "lognorm")
        else:
            ctx.warn("CellTypist trains on log1p(CP10K); this object is "
                     f"{registry.get_x_state(adata)!r}.")

    models_dir = CONFIG.project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    model = ct.train(adata, labels=label_key, feature_selection=feature_selection,
                     n_jobs=n_jobs, check_expression=False)
    model.write(str(path))

    ctx.add_artifact("model", path, caption=f"CellTypist model {name} ({organism})",
                     params={"label_key": label_key, "n_labels": int(vc.size)})
    store.set_param(ctx.project_id, f"model.{name}", str(path), "global",
                    "skin.atlas.train_model",
                    f"CellTypist model trained on {label_key} ({organism}, "
                    f"{int(adata.n_obs)} cells, {int(vc.size)} classes)")
    ctx.summary = {"name": name, "path": str(path), "organism": organism,
                   "n_cells": int(adata.n_obs), "n_labels": int(vc.size),
                   "class_balance": vc.head(20).to_dict(),
                   "usage": f"skin.atlas.celltypist(dataset_id=..., model={str(path)!r})"}
    ctx.suggest("skin.atlas.celltypist", "skin.memory.note")


@tool("skin.atlas.census_celltypes", category="atlas", needs_network=True,
      summary="Inventory annotated skin cell types in CELLxGENE Census.")
def census_celltypes(organism: str = "human", tissue: str = "skin of body",
                     disease: str = "", project_id: str = "", dry_run: bool = False,
                     seed: int = 0, *, ctx: Ctx) -> None:
    """List cell types, counts and contributing datasets for a tissue in Census.

    The Census version is pinned in config, not "latest" — releases change cell
    counts and an analysis must not shift silently underneath you.

    Args:
        organism: "human" or "mouse".
        tissue: Census `tissue_general` value. "skin of body" for skin.
        disease: Optional disease filter, e.g. "normal".
        project_id: Defaults to the active project.
        dry_run: Report the query only.
        seed: Unused.
    """
    org = K._check_organism(organism)
    species = "Homo sapiens" if org == "human" else "Mus musculus"
    filt = f"tissue_general == '{tissue}'" + (f" and disease == '{disease}'" if disease else "")
    ctx.code = (f"import cellxgene_census\n"
                f"with cellxgene_census.open_soma(census_version={CONFIG.census_version!r}) as c:\n"
                f"    obs = c['census_data'][{species!r}].obs.read(\n"
                f"        value_filter={filt!r},\n"
                f"        column_names=['cell_type','dataset_id']).concat().to_pandas()\n")
    if dry_run:
        ctx.summary = {"census_version": CONFIG.census_version, "species": species,
                       "filter": filt}
        return
    if CONFIG.offline:
        raise NetworkUnavailable(
            "CELLxGENE Census needs network access and --offline is set",
            remedy="Use skin.atlas.marker_lookup, which uses the shipped snapshots.",
            suggested_tool="skin.atlas.marker_lookup")
    try:
        import cellxgene_census
    except ImportError as e:
        raise DependencyMissing('cellxgene-census is not installed',
                                remedy='uv pip install "skin-mcp[atlas]"') from e

    with cellxgene_census.open_soma(census_version=CONFIG.census_version) as census:
        obs = (census["census_data"][species].obs
               .read(value_filter=filt, column_names=["cell_type", "dataset_id"])
               .concat().to_pandas())
    if obs.empty:
        raise NotFound(f"no cells matched {filt!r} in Census {CONFIG.census_version}",
                       remedy="Verify the tissue_general ontology string; skin is "
                              "'skin of body'.")
    vc = obs["cell_type"].value_counts()
    ctx.summary = {"census_version": CONFIG.census_version, "species": species,
                   "tissue": tissue, "disease": disease or None,
                   "n_cells": int(len(obs)), "n_cell_types": int(vc.size),
                   "n_datasets": int(obs["dataset_id"].nunique()),
                   "cell_types": vc.head(30).to_dict()}
    ctx.suggest("skin.atlas.census_expression", "skin.atlas.census_reference")


@tool("skin.atlas.census_expression", category="atlas", needs_network=True,
      summary="Mean/pct expression of genes across skin cell types in Census.")
def census_expression(genes: list[str], organism: str = "human", tissue: str = "skin of body",
                      group_by: str = "cell_type", max_cells: int = 200000,
                      project_id: str = "", dry_run: bool = False, seed: int = 0,
                      *, ctx: Ctx) -> None:
    """External sanity check on a marker before you commit to a label.

    Args:
        genes: Gene symbols to check.
        organism: "human" or "mouse".
        tissue: Census `tissue_general` value.
        group_by: Census obs column to group by.
        max_cells: Cap on cells fetched.
        project_id: Defaults to the active project.
        dry_run: Report the query only.
        seed: RNG seed.
    """
    import numpy as np

    org = K._check_organism(organism)
    species = "Homo sapiens" if org == "human" else "Mus musculus"
    if dry_run:
        ctx.summary = {"genes": genes, "species": species, "tissue": tissue}
        return
    if CONFIG.offline:
        raise NetworkUnavailable("Census needs network access and --offline is set",
                                 remedy="Use skin.atlas.marker_lookup.",
                                 suggested_tool="skin.atlas.marker_lookup")
    try:
        import cellxgene_census
    except ImportError as e:
        raise DependencyMissing("cellxgene-census is not installed",
                                remedy='uv pip install "skin-mcp[atlas]"') from e

    with cellxgene_census.open_soma(census_version=CONFIG.census_version) as census:
        ad = cellxgene_census.get_anndata(
            census=census, organism=species,
            obs_value_filter=f"tissue_general == '{tissue}'",
            var_value_filter=f"feature_name in {list(map(str, genes))}",
            column_names={"obs": [group_by], "var": ["feature_name"]})
    if ad.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        ad = ad[rng.choice(ad.n_obs, max_cells, replace=False)].copy()
    ad.var_names = ad.var["feature_name"].astype(str)

    import pandas as pd
    import scipy.sparse as sp

    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    df = pd.DataFrame(X, columns=list(ad.var_names))
    df["_g"] = ad.obs[group_by].astype(str).to_numpy()
    mean = df.groupby("_g").mean().round(3)
    pct = (df.drop(columns="_g") > 0).assign(_g=df["_g"]).groupby("_g").mean().round(3)

    p = ctx.tabledir() / "census_expression.csv"
    mean.join(pct, lsuffix="_mean", rsuffix="_pct").to_csv(p)
    ctx.add_artifact("table", p, caption=f"Census expression of {genes} in {tissue}")
    ctx.summary = {"census_version": CONFIG.census_version, "species": species,
                   "n_cells": int(ad.n_obs), "genes_found": list(ad.var_names),
                   "top_groups": {g: mean.loc[g].to_dict()
                                  for g in mean.mean(1).nlargest(8).index},
                   "table_path": str(p)}
    ctx.suggest("skin.annotate.marker_report", "skin.atlas.marker_lookup")


@tool("skin.atlas.census_reference", category="atlas", needs_network=True,
      summary="Download a downsampled Census reference for label transfer.")
def census_reference(organism: str = "human", tissue: str = "skin of body",
                     cell_types: list[str] | None = None, max_cells: int = 50000,
                     label: str = "", project_id: str = "", dry_run: bool = False,
                     seed: int = 0, *, ctx: Ctx) -> None:
    """Fetch a reference AnnData from Census and register it as a handle.

    Args:
        organism: "human" or "mouse".
        tissue: Census `tissue_general` value.
        cell_types: Restrict to these cell types. Empty = all.
        max_cells: Downsample cap.
        label: Human alias for the reference handle.
        project_id: Defaults to the active project.
        dry_run: Report the query only.
        seed: RNG seed.
    """
    import numpy as np

    org = K._check_organism(organism)
    species = "Homo sapiens" if org == "human" else "Mus musculus"
    filt = f"tissue_general == '{tissue}'"
    if cell_types:
        filt += f" and cell_type in {list(map(str, cell_types))}"
    if dry_run:
        ctx.summary = {"census_version": CONFIG.census_version, "filter": filt,
                       "max_cells": max_cells}
        return
    if CONFIG.offline:
        raise NetworkUnavailable("Census needs network access and --offline is set",
                                 suggested_tool="skin.atlas.marker_lookup")
    try:
        import cellxgene_census
    except ImportError as e:
        raise DependencyMissing("cellxgene-census is not installed",
                                remedy='uv pip install "skin-mcp[atlas]"') from e

    with cellxgene_census.open_soma(census_version=CONFIG.census_version) as census:
        ad = cellxgene_census.get_anndata(
            census=census, organism=species, obs_value_filter=filt,
            column_names={"obs": ["cell_type", "disease", "sex", "dataset_id",
                                  "assay", "donor_id"],
                          "var": ["feature_name"]})
    if ad.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        ad = ad[rng.choice(ad.n_obs, max_cells, replace=False)].copy()
    ad.var_names = ad.var["feature_name"].astype(str)
    ad.var_names_make_unique()
    ad.layers["counts"] = ad.X.copy()
    registry.set_x_state(ad, "counts")
    registry.set_organism(ad, org)
    ad.obs["Sample"] = ad.obs["dataset_id"].astype(str)

    dsid = registry.mint(ctx.project_id, ad, parent_id=None, op="atlas.census_reference",
                         params={"census_version": CONFIG.census_version, "filter": filt,
                                 "max_cells": max_cells},
                         label=label or "census_reference")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "census_version": CONFIG.census_version,
                   "species": species, "n_obs": int(ad.n_obs), "n_vars": int(ad.n_vars),
                   "cell_types": ad.obs["cell_type"].value_counts().head(20).to_dict(),
                   "diseases": ad.obs["disease"].value_counts().head(8).to_dict()}
    ctx.suggest("skin.atlas.transfer_labels", "skin.atlas.train_model")


@tool("skin.atlas.transfer_labels", category="atlas",
      summary="kNN / ingest label transfer from a reference handle, with a confusion table.")
def transfer_labels(dataset_id: str, reference_id: str, reference_label_key: str = "cell_type",
                    method: str = "knn_harmony", n_neighbors: int = 15, n_hvg: int = 2000,
                    label: str = "", project_id: str = "", dry_run: bool = False,
                    seed: int = 0, *, ctx: Ctx) -> None:
    """Transfer labels from a reference object onto the query, with per-cell confidence.

    Args:
        dataset_id: Query handle.
        reference_id: Reference handle (e.g. from skin.atlas.census_reference).
        reference_label_key: obs column on the reference holding the labels.
        method: "knn_harmony" (joint PCA + Harmony + kNN vote) or "ingest" (scanpy).
        n_neighbors: Neighbours for the vote.
        n_hvg: HVGs for the joint embedding.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the shared-gene count only.
        seed: RNG seed.
    """
    import anndata as ad_mod
    import numpy as np
    import pandas as pd
    import scanpy as sc

    if method not in ("knn_harmony", "ingest"):
        raise BadParam("method must be knn_harmony|ingest")
    query = registry.load(ctx.project_id, dataset_id, copy=True)
    ref = registry.load(ctx.project_id, reference_id, copy=True)
    require_obs(ref, reference_label_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    q_org, r_org = registry.get_organism(query), registry.get_organism(ref)
    ortho = None
    if q_org != r_org:
        rep = K.ortholog_report(list(map(str, query.var_names)), q_org, r_org)
        keep = [g for g in map(str, query.var_names) if g in rep["mapped"]]
        query = query[:, keep].copy()
        query.var_names = [rep["mapped"][g] for g in keep]
        query.var_names_make_unique()
        ortho = {"n_mapped": len(keep), "from": q_org, "to": r_org}
        ctx.warn(f"CROSS-SPECIES: mapped {len(keep)} {q_org} genes to {r_org} orthologs. "
                 f"Labels transferred across species are low confidence by construction.")

    shared = sorted(set(map(str, query.var_names)) & set(map(str, ref.var_names)))
    if dry_run:
        ctx.summary = {"n_shared_genes": len(shared), "method": method,
                       "ortholog_mapping": ortho}
        return
    if len(shared) < 500:
        raise BadParam(f"only {len(shared)} genes shared between query and reference",
                       remedy="Check that both use the same symbol convention and that the "
                              "reference is not already subset to a marker panel.")

    shift = _detect_domain_shift(query)
    ref_shift = _detect_domain_shift(ref)
    if shift["shifted"] and not ref_shift["shifted"]:
        ctx.warn("domain_shift_warning: the query looks like injured/diseased tissue and the "
                 "reference looks healthy. Transferred labels will be nearest-homeostatic "
                 "matches, not identities. Keep your marker-based labels as the record.")

    q = query[:, shared].copy()
    r = ref[:, shared].copy()
    for a in (q, r):
        if "lognorm" in a.layers:
            a.X = a.layers["lognorm"].copy()
        elif registry.get_x_state(a) == "counts":
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
    q.obs["_batch"] = "query"
    r.obs["_batch"] = "reference"
    r.obs["_ref_label"] = r.obs[reference_label_key].astype(str)
    q.obs["_ref_label"] = "?"

    if method == "ingest":
        sc.pp.highly_variable_genes(r, n_top_genes=n_hvg)
        sc.pp.pca(r, random_state=seed)
        sc.pp.neighbors(r, random_state=seed)
        sc.tl.umap(r, random_state=seed)
        sc.tl.ingest(q, r, obs="_ref_label")
        pred = q.obs["_ref_label"].astype(str).to_numpy()
        conf = np.full(len(pred), np.nan)
    else:
        joint = ad_mod.concat([r, q], label="_src", keys=["reference", "query"],
                              index_unique="-")
        joint.layers["counts"] = joint.X.copy()
        sc.pp.highly_variable_genes(joint, n_top_genes=n_hvg)
        sc.pp.scale(joint, max_value=10)
        sc.tl.pca(joint, svd_solver="arpack", random_state=seed)
        try:
            import scanpy.external as sce

            sce.pp.harmony_integrate(joint, key="_src", basis="X_pca",
                                     adjusted_basis="X_pca_h", random_state=seed)
            rep_key = "X_pca_h"
        except Exception as e:  # noqa: BLE001
            ctx.warn(f"Harmony failed ({e}); transferring on uncorrected PCA.")
            rep_key = "X_pca"

        from sklearn.neighbors import NearestNeighbors

        is_ref = (joint.obs["_src"] == "reference").to_numpy()
        R = np.asarray(joint.obsm[rep_key])[is_ref]
        Q = np.asarray(joint.obsm[rep_key])[~is_ref]
        rlab = joint.obs["_ref_label"].astype(str).to_numpy()[is_ref]
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(R))).fit(R)
        _, ind = nn.kneighbors(Q)
        pred, conf = [], []
        for row in rlab[ind]:
            u, c = np.unique(row, return_counts=True)
            pred.append(u[c.argmax()])
            conf.append(c.max() / c.sum())
        pred, conf = np.array(pred), np.array(conf, dtype=float)

    query.obs["transferred_label"] = pd.Categorical(pred)
    query.obs["transfer_confidence"] = conf

    confusion = None
    for cand in ("cell_types", "celltype", "cell_type", "labels"):
        if cand in query.obs.columns:
            confusion = (pd.crosstab(query.obs[cand].astype(str), pd.Series(pred))
                         .head(15).to_dict())
            break

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    adata.obs["transferred_label"] = pd.Categorical(pred)
    adata.obs["transfer_confidence"] = conf
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="atlas.transfer_labels",
                         params={"reference_id": reference_id, "method": method,
                                 "n_neighbors": n_neighbors,
                                 "cross_species": q_org != r_org}, label=label)
    ctx.dataset_id = dsid
    low = float(np.mean(conf < 0.6)) if np.isfinite(conf).any() else None
    if low and low > 0.3:
        ctx.warn(f"{low:.0%} of cells were assigned with under 60% neighbour agreement. "
                 f"That usually means the query contains states the reference does not.")
    ctx.summary = {
        "dataset_id": dsid, "method": method, "n_shared_genes": len(shared),
        "obs_key": "transferred_label",
        "label_counts": pd.Series(pred).value_counts().head(20).to_dict(),
        "mean_confidence": (round(float(np.nanmean(conf)), 3)
                            if np.isfinite(conf).any() else None),
        "frac_low_confidence": (round(low, 3) if low is not None else None),
        "confusion_vs_existing": confusion, "ortholog_mapping": ortho,
        "domain_shift_warning": shift if shift["shifted"] else None,
        "provenance_tag": "REFERENCE-BASED (label transfer)"
                          + (", CROSS-SPECIES" if q_org != r_org else ""),
    }
    ctx.suggest("skin.memory.record_annotation", "skin.annotate.marker_report")


@tool("skin.atlas.ortholog_map", category="atlas",
      summary="Map gene symbols across mouse and human using the shipped MGI table. Offline.")
def ortholog_map(genes: list[str], from_organism: str = "mouse", to_organism: str = "human",
                 project_id: str = "", dry_run: bool = False, seed: int = 0,
                 *, ctx: Ctx) -> None:
    """Convert gene symbols between species, reporting what failed to map.

    Never uses `.upper()`. `Trp63` -> `TP63`, `Lyz2` -> `LYZ`, `H2-Aa` ->
    `HLA-DQA2`; `Ly6c2`, `Retnlg` and `Chil3` have no human ortholog at all and
    are reported as unmapped rather than silently dropped.

    Args:
        genes: Symbols to map.
        from_organism: "mouse" or "human".
        to_organism: "mouse" or "human".
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    rep = K.ortholog_report(list(map(str, genes)), from_organism, to_organism)
    ctx.summary = {
        **{k: v for k, v in rep.items() if k not in ("mapped", "ambiguous")},
        "mapped": dict(list(rep["mapped"].items())[:40]),
        "ambiguous": dict(list(rep["ambiguous"].items())[:10]),
    }
    if rep["unmapped"]:
        ctx.warn(f"{len(rep['unmapped'])} genes have no ortholog in the MGI table: "
                 f"{rep['unmapped'][:10]}. Some of these are genuinely species-specific "
                 f"(Ly6c1/2, Retnlg, Chil3, Ly6a in mouse); others are gaps in the table. "
                 f"Do not substitute an uppercase guess.")
    ctx.suggest("skin.atlas.celltypist", "skin.annotate.score_lineages")


@tool("skin.atlas.marker_lookup", category="atlas",
      summary="Look up canonical markers for a cell type. Offline-safe.")
def marker_lookup(cell_type: str, organism: str = "mouse", source: str = "local",
                  project_id: str = "", dry_run: bool = False, seed: int = 0,
                  *, ctx: Ctx) -> None:
    """Canonical markers for a cell type from the shipped knowledge base.

    Args:
        cell_type: Cell type name, matched loosely against the shipped sets.
        organism: "mouse" or "human".
        source: "local" (shipped, offline), "cellmarker", or "panglaodb".
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    from ..style.palettes import norm_key

    org = K._check_organism(organism)
    lin = K.lineages(org)
    subs: dict[str, list[str]] = {}
    for fam in K.markers(org)["subtypes"]:
        subs.update(K.subtype_sets(org, fam))
    everything = {**lin, **subs}

    nk = norm_key(cell_type)
    exact = [k for k in everything if norm_key(k) == nk]
    fuzzy = [k for k in everything if nk and (nk in norm_key(k) or norm_key(k) in nk)]
    hits = exact or fuzzy
    if not hits:
        raise NotFound(f"no marker set matches {cell_type!r}",
                       remedy=f"Available: {sorted(everything)[:30]}")
    if source != "local":
        ctx.warn(f"source={source!r} is not wired to a live API in this build; returned the "
                 f"shipped set instead. skin.enrich.list_libraries(question_type="
                 f"'cell_identity') points at CellMarker_2024 / PanglaoDB for an "
                 f"enrichment-based check.")
    ctx.summary = {"query": cell_type, "organism": org, "source": "local",
                   "matches": {k: everything[k] for k in hits[:4]},
                   "n_matches": len(hits)}
    ctx.suggest("skin.annotate.score_lineages", "skin.plot.dotplot")


@tool("skin.atlas.search_datasets", category="atlas", needs_network=True,
      summary="Discover public skin datasets in CELLxGENE. Metadata only.")
def search_datasets(query: str = "skin", organism: str = "human", project_id: str = "",
                    dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Search CELLxGENE Discover for datasets. Returns links, never bulk data.

    Args:
        query: Free-text match against dataset titles and collections.
        organism: "human" or "mouse".
        project_id: Defaults to the active project.
        dry_run: Report the endpoint only.
        seed: Unused.
    """
    url = "https://api.cellxgene.cziscience.com/curation/v1/collections"
    if dry_run:
        ctx.summary = {"endpoint": url, "query": query}
        return
    fetch = requires_network("CELLxGENE Discover")(_http_json)
    try:
        cols = fetch(url)
    except NetworkUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        raise NetworkUnavailable(f"could not reach CELLxGENE Discover: {type(e).__name__}",
                                 remedy="Search manually at cellxgene.cziscience.com") from e

    org = K._check_organism(organism)
    species = "Homo sapiens" if org == "human" else "Mus musculus"
    q = query.lower()
    hits = []
    for c in cols if isinstance(cols, list) else []:
        name = str(c.get("name", ""))
        if q not in name.lower():
            continue
        for d in c.get("datasets", []) or []:
            orgs = [o.get("label") for o in (d.get("organism") or [])]
            if species not in orgs:
                continue
            hits.append({"collection": name[:80], "title": str(d.get("title", ""))[:80],
                         "n_cells": d.get("cell_count"),
                         "url": f"https://cellxgene.cziscience.com/collections/"
                                f"{c.get('collection_id')}"})
    hits.sort(key=lambda h: -(h.get("n_cells") or 0))
    ctx.summary = {"query": query, "organism": org, "n_hits": len(hits),
                   "datasets": hits[:15],
                   "note": "Discovery only — this tool returns metadata and links, never "
                           "bulk downloads. Use skin.atlas.census_reference to fetch data."}
    ctx.suggest("skin.atlas.census_celltypes", "skin.atlas.census_reference")
