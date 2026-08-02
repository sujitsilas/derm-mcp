"""`skin.meta.*` — sample metadata, categorical ordering, palettes."""

from __future__ import annotations

import re
from typing import Any

from .. import registry
from ..errors import BadParam
from ..memory import store
from ..style import palettes as PAL
from ._base import Ctx, require_obs, tool


@tool("skin.meta.annotate_samples", category="meta",
      summary="Attach a per-sample metadata table to obs.")
def annotate_samples(dataset_id: str, table: list[dict[str, str]], sample_key: str = "Sample",
                     label: str = "", project_id: str = "", dry_run: bool = False,
                     seed: int = 0, *, ctx: Ctx) -> None:
    """Add condition / timepoint / batch / sex / replicate columns from a sample table.

    Errors on partial coverage rather than leaving NaN in a design variable — a
    missing condition on one sample silently drops it from every downstream
    contrast.

    Args:
        dataset_id: Handle or label.
        table: One dict per sample, each with the sample key plus any columns to
            add. Example: [{"sample": "B_D7_1", "condition": "Burn",
            "timepoint": "D7", "batch": "b1"}]
        sample_key: obs column the table's sample field maps onto.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Validate coverage without minting a handle.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, sample_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if not table:
        raise BadParam("table is empty")

    sample_field = next((k for k in table[0] if k.lower() in ("sample", "sample_id", "id")), None)
    if sample_field is None:
        raise BadParam("no sample column in the table",
                       remedy="Each row needs a 'sample' key matching obs[sample_key].")

    RENAME = {"condition": "Type", "type": "Type", "timepoint": "Timepoint",
              "batch": "Batch", "sex": "Sex", "replicate": "Replicate"}
    df = pd.DataFrame(table).set_index(sample_field)
    df.index = df.index.astype(str)
    df.columns = [RENAME.get(c.lower(), c) for c in df.columns]

    obs_samples = set(adata.obs[sample_key].astype(str).unique())
    covered = set(df.index)
    missing = sorted(obs_samples - covered)
    extra = sorted(covered - obs_samples)
    if missing:
        raise BadParam(
            f"{len(missing)} samples in the data have no metadata row: {missing[:10]}",
            remedy="Every sample must be covered; partial coverage puts NaN into design "
                   "variables and silently drops those cells from every contrast.",
        )
    if extra:
        ctx.warn(f"table has rows for samples not in the data: {extra[:10]} (ignored)")

    ctx.code = (f"meta = pd.DataFrame({table!r}).set_index({sample_field!r})\n"
                f"for col in meta.columns:\n"
                f"    adata.obs[col] = adata.obs[{sample_key!r}].astype(str).map(meta[col])\n")
    if dry_run:
        ctx.summary = {"columns": list(df.columns), "n_samples": len(obs_samples),
                       "coverage": "complete"}
        return

    key_str = adata.obs[sample_key].astype(str)
    for col in df.columns:
        adata.obs[col] = pd.Categorical(key_str.map(df[col]).astype(str))

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="meta.annotate_samples",
                         params={"table": table, "sample_key": sample_key}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "added_columns": list(df.columns),
                   "levels": {c: sorted(map(str, df[c].unique()))[:12] for c in df.columns}}
    ctx.suggest("skin.meta.order_categorical", "skin.meta.assign_palette", "skin.qc.sample_stats")


@tool("skin.meta.parse_sample_names", category="meta",
      summary="Regex named groups from sample names into obs columns.")
def parse_sample_names(dataset_id: str, pattern: str, sample_key: str = "Sample",
                       apply: bool = False, label: str = "", project_id: str = "",
                       dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Split structured sample names into obs columns with a named-group regex.

    Always previews first. Set apply=True once the preview looks right.

    Args:
        dataset_id: Handle or label.
        pattern: Regex with named groups, e.g.
            r"(?P<Type>[BS])_(?P<Timepoint>D\\d+)_(?P<Replicate>\\d+)".
        sample_key: obs column holding the sample names.
        apply: Write the columns and mint a handle. False = preview only.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Same as apply=False.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, sample_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise BadParam(f"invalid regex: {e}", remedy="Use named groups: (?P<Name>...)") from e
    if not rx.groupindex:
        raise BadParam("pattern has no named groups",
                       remedy=r"Use (?P<Type>...) syntax; group names become obs columns.")

    samples = sorted(map(str, adata.obs[sample_key].astype(str).unique()))
    parsed, failed = {}, []
    for s in samples:
        m = rx.match(s) or rx.search(s)
        if m:
            parsed[s] = m.groupdict()
        else:
            failed.append(s)

    ctx.code = (f"ext = adata.obs[{sample_key!r}].astype(str).str.extract(r{pattern!r})\n"
                f"for c in ext.columns: adata.obs[c] = ext[c]\n")
    preview = {"groups": list(rx.groupindex), "parsed": dict(list(parsed.items())[:12]),
               "failed": failed, "n_parsed": len(parsed), "n_failed": len(failed)}
    if failed:
        ctx.warn(f"{len(failed)} sample names did not match: {failed[:8]}")
    if dry_run or not apply:
        ctx.summary = {**preview, "applied": False,
                       "note": "Re-run with apply=True to write these columns."}
        return
    if failed:
        raise BadParam(f"{len(failed)} samples failed to parse: {failed[:8]}",
                       remedy="Fix the pattern, or use skin.meta.annotate_samples with an "
                              "explicit table.")

    key_str = adata.obs[sample_key].astype(str)
    for g in rx.groupindex:
        adata.obs[g] = pd.Categorical(key_str.map(lambda s, _g=g: parsed[s][_g]))
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="meta.parse_sample_names",
                         params={"pattern": pattern, "sample_key": sample_key}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {**preview, "applied": True, "dataset_id": dsid}
    ctx.suggest("skin.meta.order_categorical", "skin.meta.assign_palette")


@tool("skin.meta.order_categorical", category="meta",
      summary="Set the category order for an obs column (D7 < D10 < D14, not alphabetical).")
def order_categorical(dataset_id: str, key: str, order: list[str] | None = None,
                      natural: bool = True, label: str = "", project_id: str = "",
                      dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Fix the ordering of a categorical, so every plot and table agrees.

    With `natural=True` and no explicit order, timepoints sort by their numeric
    part: D7 < D10 < D14 < D19, which alphabetical sorting gets wrong.

    Args:
        dataset_id: Handle or label.
        key: obs column to order.
        order: Explicit order. Must cover every level present.
        natural: Use natural (numeric-aware) sorting when `order` is omitted.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the resolved order without minting.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    present = list(map(str, adata.obs[key].astype(str).unique()))

    if order:
        missing = [x for x in present if x not in order]
        if missing:
            raise BadParam(f"order does not cover {missing}",
                           remedy=f"Levels present: {sorted(present)}")
        final = [x for x in order if x in present]
    else:
        final = (PAL.order_timepoints(present) if natural else sorted(present))

    ctx.code = (f"adata.obs[{key!r}] = pd.Categorical(adata.obs[{key!r}].astype(str),\n"
                f"                                    categories={final!r}, ordered=True)\n")
    if dry_run:
        ctx.summary = {"key": key, "resolved_order": final}
        return

    adata.obs[key] = pd.Categorical(adata.obs[key].astype(str), categories=final, ordered=True)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="meta.order_categorical",
                         params={"key": key, "order": final}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "key": key, "order": final,
                   "counts": adata.obs[key].value_counts().reindex(final).to_dict()}
    ctx.suggest("skin.meta.assign_palette", "skin.meta.make_composite")


@tool("skin.meta.make_composite", category="meta",
      summary='Combine obs columns into one, e.g. Type_Timepoint = "Burn D7".')
def make_composite(dataset_id: str, keys: list[str], new_key: str, sep: str = " ",
                   label: str = "", project_id: str = "", dry_run: bool = False,
                   seed: int = 0, *, ctx: Ctx) -> None:
    """Build a composite categorical from several obs columns, ordered sensibly.

    Args:
        dataset_id: Handle or label.
        keys: Columns to combine, in order.
        new_key: Name of the new column.
        sep: Separator between components.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the levels without minting.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    for k in keys:
        require_obs(adata, k)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    vals = adata.obs[keys[0]].astype(str)
    for k in keys[1:]:
        vals = vals.str.cat(adata.obs[k].astype(str), sep=sep)

    # Order by the components' own category orders where they have one.
    def sort_key(v: str) -> tuple:
        parts = v.split(sep)
        out: list[Any] = []
        for i, k in enumerate(keys):
            s = adata.obs[k]
            cats = list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype) else []
            p = parts[i] if i < len(parts) else ""
            out.append(cats.index(p) if p in cats else 10_000)
            out.append(p)
        return tuple(out)

    levels = sorted(set(vals), key=sort_key)
    ctx.code = (f"adata.obs[{new_key!r}] = pd.Categorical(\n"
                f"    adata.obs[{keys!r}].astype(str).agg({sep!r}.join, axis=1),\n"
                f"    categories={levels!r}, ordered=True)\n")
    if dry_run:
        ctx.summary = {"new_key": new_key, "levels": levels[:24], "n_levels": len(levels)}
        return

    adata.obs[new_key] = pd.Categorical(vals, categories=levels, ordered=True)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="meta.make_composite",
                         params={"keys": keys, "new_key": new_key, "sep": sep}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "new_key": new_key, "n_levels": len(levels),
                   "counts": adata.obs[new_key].value_counts().reindex(levels).to_dict()}
    ctx.suggest("skin.meta.assign_palette", "skin.plot.umap_split")


@tool("skin.meta.assign_palette", category="meta",
      summary="Assign and persist a colour palette for a categorical obs column.")
def assign_palette(dataset_id: str, key: str, scheme: str = "celltype",
                   overrides: dict[str, str] | None = None, cmap: str = "Blues",
                   label: str = "", project_id: str = "", dry_run: bool = False,
                   seed: int = 0, *, ctx: Ctx) -> None:
    """Write `uns[f"{key}_colors"]` and record the palette in project memory.

    Cell-type colours come from a stable hash of the label string, so
    "Fibroblasts" is the same colour in every project and every figure.

    Args:
        dataset_id: Handle or label.
        key: Categorical obs column.
        scheme: "condition" (treated red / control blue), "timepoint"
            (sequential, ordered), "celltype" (stable 24-colour qualitative),
            "subtype" (the seeded macrophage palette), or "manual".
        overrides: Explicit {level: "#RRGGBB"} entries, applied last.
        cmap: Base colormap for the "timepoint" scheme.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the palette without writing.
        seed: RNG seed.
    """
    import pandas as pd

    if scheme not in PAL.SCHEMES:
        raise BadParam(f"scheme must be one of {list(PAL.SCHEMES)}, got {scheme!r}")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    s = adata.obs[key]
    levels = (list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype)
              else PAL.natural_order(s.astype(str)))
    pal = PAL.build(scheme, levels, overrides, cmap=cmap)
    unknown = [k for k, v in pal.items() if v == PAL.UNKNOWN]
    if unknown and scheme == "subtype":
        ctx.warn(f"{len(unknown)} labels are not in the seeded subtype palette and got grey "
                 f"(#999999): {unknown[:8]}. Pass `overrides` to colour them explicitly — "
                 f"they are never silently reassigned.")

    ctx.code = (f"adata.obs[{key!r}] = adata.obs[{key!r}].astype('category')\n"
                f"adata.uns['{key}_colors'] = {[pal[c] for c in levels]!r}\n")
    if dry_run:
        ctx.summary = {"key": key, "scheme": scheme, "palette": pal}
        return

    PAL.apply_to_adata(adata, key, pal)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="meta.assign_palette",
                         params={"key": key, "scheme": scheme, "overrides": overrides or {}},
                         label=label)
    store.set_param(ctx.project_id, f"palette.{key}", pal, "global", "skin.meta.assign_palette",
                    f"{scheme} palette, so figures across sessions match")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "key": key, "scheme": scheme, "palette": pal,
                   "recorded_as": f"palette.{key}"}
    ctx.suggest("skin.plot.umap", "skin.plot.legend_only")


@tool("skin.meta.get_palette", category="meta", summary="Read back a recorded palette.")
def get_palette(key: str, dataset_id: str = "", project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Fetch the palette for an obs key, from memory or from the object's uns.

    Args:
        key: Categorical obs column.
        dataset_id: Optional handle; falls back to the recorded parameter.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    p = store.get_param(ctx.project_id, f"palette.{key}")
    from_obj = {}
    if dataset_id:
        adata = registry.load(ctx.project_id, dataset_id)
        from_obj = PAL.get_from_adata(adata, key)
    ctx.summary = {"key": key, "from_memory": (p or {}).get("value"),
                   "from_object": from_obj,
                   "found": bool(p or from_obj)}
