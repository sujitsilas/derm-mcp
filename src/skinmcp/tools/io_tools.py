"""`skin.io.*` — ingest, describe, lineage, save.

Ingest validation is deliberately loud. Almost every downstream surprise in a
skin dataset traces back to something that was visible at load time: duplicated
var_names, a declared organism that disagrees with the gene casing, Antibody
Capture rows sitting in the gene matrix, or an "raw" object that was already
log-normalized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import registry
from ..config import CONFIG
from ..errors import BadParam, NotFound, OrganismMismatch
from ..knowledge import _check_organism, infer_organism_from_genes
from ..memory import store
from ._base import Ctx, tool

logger = logging.getLogger(__name__)

CHEMISTRIES = ("10x_3prime_v3", "10x_3prime_v4", "10x_5prime", "10x_flex", "10x_flex_ffpe",
               "10x_multiome_gex", "snrna", "parse_evercode", "bd_rhapsody", "other")


# --------------------------------------------------------------------------- #
# shared ingest validation
# --------------------------------------------------------------------------- #

def validate_ingest(adata: Any, organism: str, ctx: Ctx, *, source: str,
                    expect_raw: bool = True) -> None:
    """Run every ingest check, appending warnings. Organism mismatch is fatal."""
    import numpy as np
    import pandas as pd

    # 1. duplicated var_names
    dup = pd.Index(adata.var_names).duplicated()
    if dup.any():
        n = int(dup.sum())
        adata.var_names_make_unique()
        ctx.warn(f"{n} duplicated gene symbols in {source} were made unique "
                 f"(Gene, Gene-1, ...). Check the reference GTF: duplicated symbols "
                 f"usually mean multiple Ensembl IDs collapsed onto one name, and the "
                 f"suffixed copies carry real counts that are now split.")

    # 2. organism vs gene casing — hard error, not a warning
    declared = _check_organism(organism)
    inferred = infer_organism_from_genes(adata.var_names)
    if inferred != "unknown" and inferred != declared:
        ex = list(map(str, adata.var_names[:8]))
        raise OrganismMismatch(
            f"declared organism {declared!r} but gene symbols look {inferred!r}",
            remedy=(f"First gene symbols: {ex}. Mouse symbols are Title-case (Actb), human "
                    f"are upper-case (ACTB). Every marker set, QC prefix and ortholog "
                    f"mapping downstream depends on this, so it cannot be a warning. "
                    f"Re-load with organism={inferred!r}, or fix the var_names."),
            details={"declared": declared, "inferred": inferred, "examples": ex},
        )

    # 3. non-GEX feature types split out
    ftcol = next((c for c in ("feature_types", "feature_type") if c in adata.var.columns), None)
    if ftcol is not None:
        types = adata.var[ftcol].astype(str)
        non_gex = types[~types.isin(["Gene Expression", "GEX"])].unique().tolist()
        if non_gex:
            for t in non_gex:
                mask = (types == t).values
                key = "protein" if "Antibody" in t else ("crispr" if "CRISPR" in t else t.lower())
                adata.obsm[f"X_{key}"] = adata[:, mask].X.copy()
                ctx.warn(f"{int(mask.sum())} {t!r} features were moved to obsm['X_{key}'] "
                         f"and removed from the gene matrix. They would otherwise dominate "
                         f"HVG selection and QC counts.")
            keep = types.isin(["Gene Expression", "GEX"]).values
            adata._inplace_subset_var(keep)

    # 4. X state
    state = registry.detect_x_state(adata)
    registry.set_x_state(adata, state)
    if expect_raw and state != "counts":
        ctx.warn(f"X in {source} looks {state!r}, not raw counts (max={float(_xmax(adata)):.2f}). "
                 f"Pseudobulk DE and re-normalization need integer counts. If a counts layer "
                 f"exists it will be used; otherwise expect skin.de.pseudobulk to refuse.")

    # 5. counts layer
    if "counts" not in adata.layers:
        if state == "counts":
            adata.layers["counts"] = adata.X.copy()
            ctx.warn("layers['counts'] was absent; X looked like raw counts and was copied "
                     "into it. Verify this is right before running DE.")
        else:
            ctx.warn("No layers['counts'] and X is not raw counts. Handles minted from this "
                     "object cannot be used for pseudobulk DE or subcluster "
                     "re-normalization.")

    # 6. empty cells / genes
    import scipy.sparse as sp

    X = adata.layers.get("counts", adata.X)
    tot = np.asarray(X.sum(1)).ravel() if sp.issparse(X) else np.asarray(X).sum(1)
    n_empty = int((tot == 0).sum())
    if n_empty:
        ctx.warn(f"{n_empty} cells have zero total counts. skin.qc.apply_filters will drop them.")

    registry.set_organism(adata, declared)
    u = registry.skinmcp_uns(adata)
    u["source"] = source

    # Remember where this came from so a later bare filename resolves.
    if Path(source).exists():
        CONFIG.register_data_dir(Path(source))


def _xmax(adata: Any) -> float:
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    sub = X[: min(2000, X.shape[0])]
    arr = sub.data if sp.issparse(sub) else np.asarray(sub)
    return float(np.nanmax(arr)) if arr.size else 0.0


def _describe_payload(adata: Any, dataset_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    """Compact object description that fits the 4 KB return budget.

    A real object has 30-40 obs columns. Emitting full value counts for each
    blows the budget, the whole summary spills to a resource, and the caller —
    which typically calls describe FIRST — gets nothing it can act on. So obs is
    a one-line type map for every column, plus the levels of the few columns
    that actually drive a design. The complete schema stays available at
    skin://dataset/{id}/obs_schema.
    """

    from . import _introspect as I

    types = {str(c): (f"cat[{adata.obs[c].astype(str).nunique()}]"
                      if I.is_categorical(adata.obs[c]) else str(adata.obs[c].dtype))
             for c in adata.obs.columns}
    cats = I.categoricals(adata)
    ranked = I.groupable(adata)
    levels = {c: cats[c][:15] for c in ranked[:8]}
    constant = I.constant_columns(adata)

    return {
        "dataset_id": dataset_id,
        "label": (row or {}).get("label") or "",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "organism": registry.get_organism(adata),
        "x_state": registry.get_x_state(adata),
        # zellkonverter/Seurat exports can yield a stray None key here; drop it
        # rather than report a layer the caller cannot address.
        "layers": [str(k) for k in adata.layers.keys() if k is not None],
        "obsm": [str(k) for k in adata.obsm.keys() if k is not None],
        "uns_keys": sorted(map(str, adata.uns.keys()))[:12],
        "var_columns": list(adata.var.columns)[:10],
        "obs_types": types,
        "obs_levels": levels,
        "groupable_columns": ranked,
        "constant_columns": constant,
        "n_obs_columns": int(len(types)),
        "full_obs_schema": f"skin://dataset/{dataset_id}/obs_schema",
        "parent_id": (row or {}).get("parent_id"),
        "op": (row or {}).get("op"),
    }


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #

@tool("skin.io.load_10x", category="io", summary="Load a CellRanger filtered matrix.")
def load_10x(path: str, sample_name: str, organism: str = "mouse",
             chemistry: str = "10x_3prime_v3", label: str = "", project_id: str = "",
             dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Load a CellRanger `filtered_feature_bc_matrix` (`.h5` file or mtx directory).

    Also accepts `raw_feature_bc_matrix` — pass that path to
    skin.qc.estimate_ambient later for empty-droplet profiling.

    Args:
        path: Path to the .h5 file or the mtx directory.
        sample_name: Written to obs["Sample"]. Use the name you will use everywhere.
        organism: "mouse" or "human". Validated against gene casing; a mismatch is fatal.
        chemistry: One of the presets in knowledge/platforms.yaml. Drives the QC defaults.
        label: Optional human alias for the resulting handle.
        project_id: Defaults to the active project.
        dry_run: Validate the path and return the code without loading.
        seed: RNG seed.
    """
    import scanpy as sc

    p = CONFIG.resolve_input(path)
    ctx.adopt_output_dir(p)
    if chemistry not in CHEMISTRIES:
        raise BadParam(f"unknown chemistry {chemistry!r}", remedy=f"One of {list(CHEMISTRIES)}")
    if not p.exists():
        raise NotFound(f"{p} does not exist", remedy="Check the path; it must be the "
                                                     "filtered_feature_bc_matrix .h5 or directory.")
    ctx.code = (f"import scanpy as sc\n"
                f"adata = sc.read_10x_{'h5' if p.suffix == '.h5' else 'mtx'}({str(p)!r})\n"
                f"adata.var_names_make_unique()\n"
                f"adata.obs['Sample'] = {sample_name!r}\n"
                f"adata.layers['counts'] = adata.X.copy()\n")
    if dry_run:
        ctx.summary = {"path": str(p), "sample": sample_name, "chemistry": chemistry}
        return

    adata = (sc.read_10x_h5(str(p)) if p.suffix == ".h5"
             else sc.read_10x_mtx(str(p), var_names="gene_symbols", cache=False))
    adata.var_names_make_unique()
    adata.obs["Sample"] = sample_name
    validate_ingest(adata, organism, ctx, source=str(p))
    registry.skinmcp_uns(adata)["chemistry"] = chemistry

    dsid = ctx.mint(adata, parent_id=None, op="load_10x",
                         params={"path": str(p), "sample": sample_name,
                                 "organism": organism, "chemistry": chemistry},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
                   "sample": sample_name, "chemistry": chemistry, "organism": organism}
    ctx.suggest("skin.io.build_multisample", "skin.qc.sample_stats", "skin.meta.annotate_samples")


@tool("skin.io.load_h5ad", category="io", summary="Load an existing .h5ad.")
def load_h5ad(path: str, organism: str = "mouse", counts_layer: str = "counts",
              label: str = "", allow_no_counts: bool = False, project_id: str = "",
              dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Load an `.h5ad`, validate it, and register it as a handle.

    Args:
        path: Path to the .h5ad.
        organism: "mouse" or "human". A casing mismatch is a hard error.
        counts_layer: Which layer holds raw counts. Renamed to "counts" internally.
        label: Optional human alias.
        allow_no_counts: Accept an object with no raw counts. DE and subcluster
            re-normalization will then be unavailable; only set this for a
            read-only look at someone else's processed object.
        project_id: Defaults to the active project.
        dry_run: Validate the path only.
        seed: RNG seed.
    """
    import anndata as ad

    p = CONFIG.resolve_input(path)
    ctx.adopt_output_dir(p)
    if not p.exists():
        raise NotFound(f"{p} does not exist")
    ctx.code = f"import anndata as ad\nadata = ad.read_h5ad({str(p)!r})\n"
    if dry_run:
        ctx.summary = {"path": str(p)}
        return

    registry.admit_external(p)
    adata = ad.read_h5ad(p)
    if counts_layer != "counts" and counts_layer in adata.layers:
        adata.layers["counts"] = adata.layers.pop(counts_layer)
        ctx.warn(f"layer {counts_layer!r} was renamed to 'counts' (the server's convention).")
    validate_ingest(adata, organism, ctx, source=str(p),
                    expect_raw="counts" not in adata.layers)

    dsid = ctx.mint(adata, parent_id=None, op="load_h5ad",
                         params={"path": str(p), "organism": organism},
                         label=label, allow_no_counts=allow_no_counts)
    ctx.dataset_id = dsid
    ctx.summary = _describe_payload(adata, dsid, None)
    ctx.suggest("skin.io.describe", "skin.qc.sample_stats", "skin.memory.brief")


@tool("skin.io.load_mtx_export", category="io",
      summary="Load the SCE mtx-export layout (matrix/genes/barcodes/metadata).")
def load_mtx_export(directory: str, organism: str = "mouse", label: str = "",
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Load an mtx export written by the R bridge or by `SingleCellExperiment`.

    Expects `matrix.mtx`, `genes.txt`, `barcodes.txt`, `metadata.csv`, and one
    `reducedDim_*.csv` per embedding (reference notebook cell 3).

    Args:
        directory: The export directory.
        organism: "mouse" or "human".
        label: Optional human alias.
        project_id: Defaults to the active project.
        dry_run: Check the directory layout without loading.
        seed: RNG seed.
    """
    from ..runtimes.bridge import read_mtx_export

    d = CONFIG.resolve_input(directory)
    ctx.adopt_output_dir(d)
    if not d.is_dir():
        raise NotFound(f"{d} is not a directory")
    missing = [f for f in ("matrix.mtx", "genes.txt", "barcodes.txt") if not (d / f).exists()]
    if missing:
        raise NotFound(f"missing {missing} in {d}",
                       remedy="Expected the SCE export layout: matrix.mtx, genes.txt, "
                              "barcodes.txt, metadata.csv, reducedDim_*.csv")
    ctx.code = (f"from skinmcp.runtimes.bridge import read_mtx_export\n"
                f"adata = read_mtx_export({str(d)!r})\n")
    if dry_run:
        ctx.summary = {"directory": str(d), "files": sorted(f.name for f in d.iterdir())[:20]}
        return

    adata = read_mtx_export(d)
    validate_ingest(adata, organism, ctx, source=str(d))
    dsid = ctx.mint(adata, parent_id=None, op="load_mtx_export",
                         params={"dir": str(d), "organism": organism}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = _describe_payload(adata, dsid, None)
    ctx.suggest("skin.io.describe", "skin.qc.sample_stats")


def _align_barcodes(target: Any, source: Any) -> tuple[Any, str]:
    """Match `source` cells onto `target` cells, tolerating the usual suffix drift.

    Barcodes routinely disagree between a processed object and the raw matrix it
    came from: `AAACCT-1` vs `AAACCT`, or `sample1_AAACCT-1` after a merge. Try
    the obvious normalisations in order and report which one worked, rather than
    failing on an exact-match test the user cannot see the result of.
    """
    import pandas as pd

    def strip_suffix(x: Any) -> Any:
        return pd.Index([str(b).rsplit("-", 1)[0] for b in x])

    def strip_prefix(x: Any) -> Any:
        return pd.Index([str(b).split("_", 1)[-1] for b in x])

    strategies = [
        ("exact", lambda i: i),
        ("ignoring the -1 suffix", strip_suffix),
        ("ignoring the sample prefix", strip_prefix),
        ("ignoring both prefix and suffix", lambda i: strip_suffix(strip_prefix(i))),
    ]
    best, best_n, best_how = None, 0, ""
    for how, fn in strategies:
        s_idx, t_idx = fn(source.obs_names), fn(target.obs_names)
        if not s_idx.is_unique or not t_idx.is_unique:
            continue
        n = len(t_idx.intersection(s_idx))
        if n > best_n:
            best, best_n, best_how = (s_idx, t_idx), n, how
        if n == target.n_obs:
            break
    if best is None or best_n == 0:
        return None, None, "no barcode overlap under any normalisation"
    s_idx, t_idx = best
    src = source.copy()
    src.obs_names = s_idx
    have = set(s_idx)
    # Positions in `target`, and the source rows in that same order, so the
    # caller can assign one into the other without re-deriving the mapping.
    rows = [i for i, b in enumerate(t_idx) if b in have]
    keep = [t_idx[i] for i in rows]
    return src[keep], rows, f"{best_n}/{target.n_obs} cells matched {best_how}"


@tool("skin.io.attach_counts", category="io",
      summary="Attach raw counts from a separate matrix to a counts-free handle.")
def attach_counts(dataset_id: str, path: str, counts_layer: str = "",
                  label: str = "", project_id: str = "", dry_run: bool = False,
                  seed: int = 0, *, ctx: Ctx) -> None:
    """Add `layers['counts']` to an object that arrived without them.

    Processed objects are often shared log-normalised with the raw matrix
    stripped, which blocks pseudobulk DE, subclustering and re-normalisation.
    If the user still has the raw counts — the CellRanger `.h5`/mtx directory,
    or an earlier `.h5ad` — this grafts them on by barcode.

    Cells are matched on barcode (tolerating `-1` suffixes and `sample_` prefixes)
    and genes on `var_names`. Counts must be integers; anything else is rejected,
    because silently treating normalised values as counts is how a DE result ends
    up wrong rather than absent.

    Args:
        dataset_id: The counts-free handle.
        path: Raw counts source — .h5ad, CellRanger .h5, or a 10x/mtx directory.
        counts_layer: Which layer of the source holds counts. Default: its X.
        label: Optional alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the overlap without writing a new handle.
        seed: RNG seed.
    """
    import numpy as np
    import scipy.sparse as sp

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    if "counts" in adata.layers:
        ctx.warn("this handle already has layers['counts']; it will be replaced.")
    p = CONFIG.resolve_input(path)
    if not p.exists():
        raise NotFound(f"{p} does not exist",
                       remedy="Ask the user where the raw counts live: the CellRanger "
                              "filtered_feature_bc_matrix .h5/directory, or the .h5ad "
                              "this object was derived from.")

    src = _read_counts_source(p)
    if counts_layer:
        if counts_layer not in src.layers:
            raise NotFound(f"{p} has no layer {counts_layer!r}",
                           remedy=f"Layers present: {list(src.layers)}")
        src.X = src.layers[counts_layer]

    aligned, rows, how = _align_barcodes(adata, src)
    if aligned is None:
        raise BadParam(
            f"cells in {p.name} do not match this dataset ({how})",
            remedy=("Check this is the raw matrix for THIS object. Sample-merged "
                    "objects usually need the per-sample matrices instead — load "
                    "those with skin.io.build_multisample."),
            details={"dataset_barcodes": list(map(str, adata.obs_names[:3])),
                     "source_barcodes": list(map(str, src.obs_names[:3]))},
        )

    genes = [g for g in adata.var_names if g in set(aligned.var_names)]
    if len(genes) < 0.5 * adata.n_vars:
        raise BadParam(
            f"only {len(genes)}/{adata.n_vars} genes are present in {p.name}",
            remedy="The raw matrix must cover the genes in this object. A different "
                   "reference build or a symbol/Ensembl mismatch would look like this.",
        )

    x = aligned[:, genes].X
    sample = (x[:20].toarray() if sp.issparse(x) else np.asarray(x[:20]))
    if sample.size and not np.allclose(sample, np.round(sample)):
        raise BadParam(
            f"{p.name} does not hold integer counts",
            remedy="This looks normalised too. Point at the raw matrix — counts must "
                   "be integers for pseudobulk DE to be valid.",
        )

    ctx.summary = {"matched": how, "genes_matched": len(genes), "source": str(p)}
    if dry_run:
        return

    gpos = {g: i for i, g in enumerate(adata.var_names)}
    cols = [gpos[g] for g in genes]
    block = x.toarray() if sp.issparse(x) else np.asarray(x)
    full = np.zeros(adata.shape, dtype="float32")
    full[np.ix_(rows, cols)] = block
    adata.layers["counts"] = sp.csr_matrix(full) if sp.issparse(x) else full

    if len(rows) < adata.n_obs:
        ctx.warn(f"{adata.n_obs - len(rows)} cells had no match and their counts are "
                 f"zero; drop them before DE or the pseudobulk sums will be wrong.")

    ctx.code = (f"import anndata as ad\n"
                f"raw = ad.read_h5ad({str(p)!r})\n"
                f"adata.layers['counts'] = raw[adata.obs_names, adata.var_names].X\n")
    dsid = ctx.mint(adata, parent_id=dataset_id, op="attach_counts",
                    params={"path": str(p), "matched": how}, label=label)
    ctx.dataset_id = dsid
    ctx.summary["dataset_id"] = dsid
    ctx.suggest("skin.io.describe", "skin.de.pseudobulk", "skin.sub.extract")


def _read_counts_source(p: Path) -> Any:
    """Read a raw-counts source in whichever of the three usual shapes it is."""
    import scanpy as sc

    if p.is_dir():
        return sc.read_10x_mtx(p)
    if p.suffix == ".h5ad":
        import anndata as ad

        return ad.read_h5ad(p)
    if p.suffix == ".h5":
        return sc.read_10x_h5(p)
    raise BadParam(f"cannot read counts from {p.name}",
                   remedy="Expected a .h5ad, a CellRanger .h5, or a 10x/mtx directory.")


@tool("skin.io.load_seurat_rds", category="io", needs_r=True,
      summary="Convert a Seurat .rds to h5ad via the R bridge.")
def load_seurat_rds(path: str, organism: str = "mouse", assay: str = "RNA", label: str = "",
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Convert a Seurat object to AnnData through the pinned R container.

    Args:
        path: Path to the .rds.
        organism: "mouse" or "human".
        assay: Seurat assay to export.
        label: Optional human alias.
        project_id: Defaults to the active project.
        dry_run: Report the plan without running R.
        seed: RNG seed.
    """
    from ..runtimes.bridge import seurat_to_h5ad

    p = CONFIG.resolve_input(path)
    ctx.adopt_output_dir(p)
    if not p.exists():
        raise NotFound(f"{p} does not exist")
    ctx.code = (f"# R bridge: Seurat -> SingleCellExperiment -> h5ad\n"
                f"# runtimes/r/scripts/seurat_to_h5ad.R  input={str(p)!r} assay={assay!r}\n")
    if dry_run:
        ctx.summary = {"path": str(p), "assay": assay, "backend": "r-bridge"}
        return

    adata, log_tail = seurat_to_h5ad(p, assay=assay, project_id=ctx.project_id)
    validate_ingest(adata, organism, ctx, source=str(p))
    dsid = ctx.mint(adata, parent_id=None, op="load_seurat_rds",
                         params={"path": str(p), "assay": assay, "organism": organism},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {**_describe_payload(adata, dsid, None), "r_log_tail": log_tail[-500:]}
    ctx.suggest("skin.io.describe", "skin.qc.sample_stats")


@tool("skin.io.build_multisample", category="io",
      summary="Concatenate per-sample inputs into one annotated object. The normal entry point.")
def build_multisample(
    inputs: list[dict[str, str]],
    organism: str = "mouse",
    chemistry: str = "10x_3prime_v3",
    label: str = "raw_merged",
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Load several samples and concatenate them with per-sample metadata attached.

    This is the normal way to start a project. Concatenation uses
    `batch_key="Sample"` and `index_unique="_"`, so barcodes stay traceable.

    Args:
        inputs: One dict per sample. Required keys: `path`, `sample`. Any other
            keys (`condition`, `timepoint`, `batch`, `sex`, `replicate`, ...)
            become obs columns on that sample's cells. Example:
            [{"path": "/data/B_D7_1/filtered_feature_bc_matrix.h5",
              "sample": "B_D7_1", "condition": "Burn", "timepoint": "D7"}]
        organism: "mouse" or "human".
        chemistry: Applied to every input unless a row overrides it.
        label: Human alias for the merged handle.
        project_id: Defaults to the active project.
        dry_run: Validate the inputs and return the plan.
        seed: RNG seed.
    """
    import anndata as ad
    import scanpy as sc

    if not inputs:
        raise BadParam("inputs is empty", remedy="Pass at least one {path, sample} entry.")
    for i, row in enumerate(inputs):
        for k in ("path", "sample"):
            if not row.get(k):
                raise BadParam(f"inputs[{i}] is missing {k!r}",
                               remedy="Every entry needs at least 'path' and 'sample'.")
    samples = [r["sample"] for r in inputs]
    if len(set(samples)) != len(samples):
        raise BadParam("duplicate sample names in inputs",
                       remedy=f"Sample names must be unique. Got: {samples}")

    meta_keys = sorted({k for r in inputs for k in r if k not in ("path", "sample", "chemistry")})
    ctx.code = (
        "import scanpy as sc, anndata as ad\n"
        f"inputs = {inputs!r}\n"
        "parts = []\n"
        "for row in inputs:\n"
        "    p = row['path']\n"
        "    a = sc.read_10x_h5(p) if p.endswith('.h5') else (\n"
        "        sc.read_h5ad(p) if p.endswith('.h5ad') else sc.read_10x_mtx(p, var_names='gene_symbols'))\n"
        "    a.var_names_make_unique()\n"
        "    for k, v in row.items():\n"
        "        if k != 'path':\n"
        "            a.obs[k.capitalize() if k in ('sample','condition','timepoint','batch') else k] = v\n"
        "    parts.append(a)\n"
        "adata = ad.concat(parts, join='outer', label='Sample', "
        "keys=[r['sample'] for r in inputs], index_unique='_')\n"
        "adata.layers['counts'] = adata.X.copy()\n"
    )
    if dry_run:
        ctx.summary = {"n_inputs": len(inputs), "samples": samples,
                       "metadata_columns": meta_keys, "organism": organism}
        return

    # Canonical capitalised names for the four columns the rest of the server expects.
    RENAME = {"sample": "Sample", "condition": "Type", "type": "Type",
              "timepoint": "Timepoint", "batch": "Batch", "sex": "Sex",
              "replicate": "Replicate"}
    parts, per_sample = [], []
    for row in inputs:
        p = Path(row["path"]).expanduser()
        if not p.exists():
            raise NotFound(f"{p} does not exist (sample {row['sample']!r})")
        if p.suffix == ".h5":
            a = sc.read_10x_h5(str(p))
        elif p.suffix == ".h5ad":
            a = sc.read_h5ad(str(p))
        else:
            a = sc.read_10x_mtx(str(p), var_names="gene_symbols", cache=False)
        a.var_names_make_unique()
        for k, v in row.items():
            if k == "path":
                continue
            a.obs[RENAME.get(k.lower(), k)] = str(v)
        parts.append(a)
        per_sample.append({"sample": row["sample"], "n_cells": int(a.n_obs),
                           "n_genes": int(a.n_vars)})

    adata = ad.concat(parts, join="outer", label="Sample",
                      keys=[r["sample"] for r in inputs], index_unique="_")
    adata.obs["Sample"] = adata.obs["Sample"].astype("category")
    validate_ingest(adata, organism, ctx, source=f"{len(inputs)} samples")
    registry.skinmcp_uns(adata)["chemistry"] = chemistry

    n_genes_each = {r["sample"]: int(p.n_vars) for r, p in zip(inputs, parts)}
    if len(set(n_genes_each.values())) > 1:
        ctx.warn(f"Samples have different gene counts {n_genes_each}; concat used "
                 f"join='outer', so genes absent from a sample are zero-filled there. "
                 f"That is correct only if the samples share a reference — verify.")

    dsid = ctx.mint(adata, parent_id=None, op="build_multisample",
                         params={"inputs": inputs, "organism": organism,
                                 "chemistry": chemistry}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
                   "n_samples": len(inputs), "per_sample": per_sample,
                   "obs_columns": list(adata.obs.columns), "organism": organism,
                   "chemistry": chemistry}
    ctx.suggest("skin.qc.sample_stats", "skin.meta.order_categorical", "skin.io.describe")


@tool("skin.io.describe", category="io", summary="Shape, obs schema, layers, embeddings.")
def describe(dataset_id: str, project_id: str = "", dry_run: bool = False, seed: int = 0,
             *, ctx: Ctx) -> None:
    """Describe a handle: dimensions, obs columns with cardinality, layers, obsm, X state.

    The full marker tables and any large obs level lists are exposed as
    resources rather than returned inline.

    Args:
        dataset_id: Handle or human label.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    # peek, not load: this reports obs/var/shape and never touches the matrix,
    # and reading a 78k x 20k object in full to list its columns was enough to
    # take the server down beside a resident local model.
    adata = registry.peek(ctx.project_id, dataset_id)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id) or dataset_id
    row = store.get_dataset(ctx.project_id, resolved)
    ctx.dataset_id = resolved
    ctx.inputs = {"dataset_id": resolved}
    ctx.summary = _describe_payload(adata, resolved, row)
    ctx.summary["obs_schema_uri"] = f"skin://dataset/{resolved}/obs_schema"
    ctx.suggest("skin.qc.sample_stats", "skin.io.lineage", "skin.help.workflow")


@tool("skin.io.lineage", category="io", summary="ASCII lineage tree of dataset handles.")
def lineage(dataset_id: str = "", project_id: str = "", dry_run: bool = False, seed: int = 0,
            *, ctx: Ctx) -> None:
    """Show how each handle was derived from its parent.

    Args:
        dataset_id: Focus on one handle's ancestry and descendants. Empty = whole forest.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    tree = registry.lineage_tree(ctx.project_id, dataset_id)
    ctx.summary = {"tree": tree, "n_datasets": len(store.list_datasets(ctx.project_id))}


@tool("skin.io.set_label", category="io", summary="Give a handle a human alias.")
def set_label(dataset_id: str, label: str, project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Alias a handle so you can refer to it as e.g. "macs_final" instead of ds_1a2b3c4d.

    Args:
        dataset_id: Handle to alias.
        label: The alias. Aliases resolve anywhere a dataset_id is accepted.
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if resolved is None:
        raise registry.bad_handle(ctx.project_id, dataset_id)
    if dry_run:
        ctx.summary = {"would_label": resolved, "as": label}
        return
    store.set_label(ctx.project_id, resolved, label)
    ctx.dataset_id = resolved
    ctx.summary = {"dataset_id": resolved, "label": label}


@tool("skin.io.save_h5ad", category="io", summary="Export a handle to an .h5ad path.")
def save_h5ad(dataset_id: str, path: str, project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Write a handle out to a path of your choosing, for use outside this server.

    Args:
        dataset_id: Handle to export.
        path: Destination .h5ad path.
        project_id: Defaults to the active project.
        dry_run: Report the destination without writing.
        seed: Unused.
    """
    p = CONFIG.resolve_input(path)
    ctx.code = f"adata.write_h5ad({str(p)!r}, compression='gzip')\n"
    if dry_run:
        ctx.summary = {"would_write": str(p)}
        return
    adata = registry.load(ctx.project_id, dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    for note in registry._sanitize_for_h5ad(adata):
        ctx.warn(note)
    adata.write_h5ad(p, compression="gzip")
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.add_artifact("h5ad", p, caption=f"export of {ctx.dataset_id}")
    ctx.summary = {"path": str(p), "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
                   "size_mb": round(p.stat().st_size / 1e6, 1)}


@tool("skin.io.cache_status", category="io", summary="In-memory AnnData cache state.")
def cache_status(clear: bool = False, project_id: str = "", dry_run: bool = False,
                 seed: int = 0, *, ctx: Ctx) -> None:
    """Report (and optionally clear) the LRU cache of loaded AnnData objects.

    Args:
        clear: Drop everything from the cache to free memory.
        project_id: Defaults to the active project.
        dry_run: Report only.
        seed: Unused.
    """
    before = registry.cache_info()
    if clear and not dry_run:
        registry.cache_clear()
    ctx.summary = {"before": before, "after": registry.cache_info() if clear else before}
