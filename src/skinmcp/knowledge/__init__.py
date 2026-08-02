"""Loaders for the shipped knowledge base (markers, contamination patterns,
platform presets, gene sets, enrichment libraries, orthologs).

Everything is read once and cached; these files are static data, not
configuration, and are never written at runtime.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..errors import BadParam, OrganismMismatch

HERE = Path(__file__).parent
ORGANISMS = ("mouse", "human")


def _check_organism(organism: str) -> str:
    o = (organism or "").strip().lower()
    if o in ("mus musculus", "mm", "mouse"):
        return "mouse"
    if o in ("homo sapiens", "hs", "human"):
        return "human"
    raise OrganismMismatch(
        f"unsupported organism {organism!r}",
        remedy="skin-mcp v1 supports 'mouse' (Mus musculus) and 'human' (Homo sapiens) only.",
    )


@lru_cache(maxsize=16)
def _load_yaml(name: str) -> dict[str, Any]:
    p = HERE / name
    if not p.exists():
        raise BadParam(f"knowledge file {name} is missing from the install")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def markers(organism: str) -> dict[str, Any]:
    return _load_yaml(f"markers_{_check_organism(organism)}.yaml")


def lineages(organism: str) -> dict[str, list[str]]:
    return markers(organism)["lineages"]


def compartments(organism: str) -> dict[str, list[str]]:
    return markers(organism)["compartments"]


def gates(organism: str) -> dict[str, list[str]]:
    return markers(organism)["gates"]


def exclusive_pairs(organism: str) -> list[list[str]]:
    return markers(organism)["exclusive_pairs"]


def subtype_sets(organism: str, family: str) -> dict[str, list[str]]:
    subs = markers(organism)["subtypes"]
    if family not in subs:
        raise BadParam(
            f"unknown subtype family {family!r}",
            remedy=f"Available families: {sorted(subs)}",
        )
    return subs[family]


def qc_patterns(organism: str) -> dict[str, str]:
    return markers(organism)["qc_patterns"]


def cell_cycle(organism: str) -> dict[str, list[str]]:
    return markers(organism)["cell_cycle"]


def platforms() -> dict[str, Any]:
    return _load_yaml("platforms.yaml")


def platform_preset(organism: str, chemistry: str) -> dict[str, Any]:
    p = platforms()[_check_organism(organism)]
    if chemistry not in p:
        raise BadParam(
            f"unknown chemistry {chemistry!r}",
            remedy=f"Known chemistries: {sorted(p)}",
        )
    return dict(p[chemistry])


def platform_rules(chemistry: str) -> dict[str, Any]:
    return dict(platforms().get("rules", {}).get(chemistry, {}))


def qc_flag_thresholds() -> dict[str, Any]:
    return platforms()["flags"]


def expected_doublet_rate(n_cells: int) -> float:
    """Interpolate the 10x multiplet table for the recovered-cell count."""
    tbl = platforms()["multiplet_table"]
    xs = [r["cells_recovered"] for r in tbl]
    ys = [r["rate"] for r in tbl]
    if n_cells <= xs[0]:
        return ys[0]
    if n_cells >= xs[-1]:
        # Beyond the published table the rate is ~0.8% per 1000 cells loaded.
        return min(0.35, ys[-1] + (n_cells - xs[-1]) * 7.6e-6)
    for i in range(1, len(xs)):
        if n_cells <= xs[i]:
            f = (n_cells - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def contamination_groups(organism: str) -> dict[str, list[str]]:
    d = _load_yaml("contamination.yaml")
    return d[_check_organism(organism)]


def contamination_presets() -> dict[str, list[str]]:
    return _load_yaml("contamination.yaml")["presets"]


def resolve_gene_groups(organism: str, groups: list[str]) -> list[str]:
    """Expand preset names into concrete gene-group names."""
    known = contamination_groups(organism)
    presets = contamination_presets()
    out: list[str] = []
    for g in groups or []:
        if g in presets:
            out.extend(presets[g])
        elif g in known:
            out.append(g)
        else:
            raise BadParam(
                f"unknown gene group {g!r}",
                remedy=f"Gene groups: {sorted(known)}. Presets: {sorted(presets)}.",
            )
    return list(dict.fromkeys(out))


def match_gene_groups(organism: str, groups: list[str], var_names: Any) -> dict[str, list[str]]:
    """Which genes in `var_names` each named group actually matches."""
    known = contamination_groups(organism)
    resolved = resolve_gene_groups(organism, groups)
    names = list(map(str, var_names))
    out: dict[str, list[str]] = {}
    for g in resolved:
        pats = [re.compile(p) for p in known[g]]
        out[g] = [n for n in names if any(p.match(n) for p in pats)]
    return out


def genesets() -> dict[str, Any]:
    return _load_yaml("genesets.yaml")


def get_signature(name: str, organism: str = "mouse") -> list[str]:
    sigs = genesets()["signatures"]
    if name not in sigs:
        raise BadParam(f"unknown signature {name!r}", remedy=f"Available: {sorted(sigs)}")
    genes = list(sigs[name]["genes"])
    if _check_organism(organism) == "human":
        genes = map_orthologs(genes, "mouse", "human")
    return genes


def get_panel(name: str, organism: str = "mouse") -> dict[str, list[str]]:
    panels = genesets()["panels"]
    if name not in panels:
        raise BadParam(f"unknown panel {name!r}", remedy=f"Available: {sorted(panels)}")
    return {s: get_signature(s, organism) for s in panels[name]["signatures"]}


def subtype_priors(name: str) -> dict[str, Any]:
    pri = genesets()["subtype_priors"]
    if name not in pri:
        raise BadParam(f"unknown prior {name!r}", remedy=f"Available: {sorted(pri)}")
    return pri[name]


def enrich_libraries() -> dict[str, Any]:
    return _load_yaml("enrich_libraries.yaml")


def exclude_preset(name: str) -> list[str]:
    presets = enrich_libraries()["exclude_presets"]
    if name not in presets:
        raise BadParam(f"unknown exclusion preset {name!r}", remedy=f"Available: {sorted(presets)}")
    return list(presets[name]["terms"])


# --------------------------------------------------------------------------- #
# orthologs
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=2)
def _ortholog_maps() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    m2h: dict[str, list[str]] = {}
    h2m: dict[str, list[str]] = {}
    p = HERE / "orthologs_mm_hs.tsv"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("mouse_symbol"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        m, h = parts[0], parts[1]
        m2h.setdefault(m, []).append(h)
        h2m.setdefault(h, []).append(m)
    return m2h, h2m


def map_orthologs(genes: list[str], from_organism: str, to_organism: str) -> list[str]:
    """Map symbols across species using the shipped MGI table.

    Never `.upper()`. `Trp63` -> `TP63`, `Lyz2` -> `LYZ`, `H2-Aa` -> `HLA-DQA2`;
    genes with no ortholog are dropped, and `ortholog_report` tells you which.
    """
    f, t = _check_organism(from_organism), _check_organism(to_organism)
    if f == t:
        return list(genes)
    m2h, h2m = _ortholog_maps()
    table = m2h if f == "mouse" else h2m
    out: list[str] = []
    for g in genes:
        hits = table.get(g)
        if hits:
            out.append(hits[0])
    return list(dict.fromkeys(out))


def ortholog_report(genes: list[str], from_organism: str, to_organism: str) -> dict[str, Any]:
    f, t = _check_organism(from_organism), _check_organism(to_organism)
    m2h, h2m = _ortholog_maps()
    table = m2h if f == "mouse" else h2m
    mapped: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for g in genes:
        hits = table.get(g)
        if not hits:
            unmapped.append(g)
        elif len(hits) == 1:
            mapped[g] = hits[0]
        else:
            mapped[g] = hits[0]
            ambiguous[g] = hits
    return {
        "from": f, "to": t, "n_in": len(genes), "n_mapped": len(mapped),
        "mapped": mapped, "ambiguous": ambiguous, "unmapped": unmapped,
        "source": "MGI HOM_MouseHumanSequence.rpt (shipped snapshot)",
    }


def present(adata: Any, genes: list[str]) -> list[str]:
    """Subset a gene list to what is actually in `var_names`, order preserved."""
    have = set(map(str, adata.var_names))
    return [g for g in genes if g in have]


def infer_organism_from_genes(var_names: Any) -> str:
    """Mouse gene symbols are Title-case, human are ALL CAPS.

    Used only to *validate* a declared organism on ingest; a mismatch is a hard
    error, because every marker set and QC prefix downstream depends on casing.
    """
    names = [str(v) for v in list(var_names)[:4000]]
    if not names:
        return "unknown"
    upper = sum(1 for n in names if n.isupper() and len(n) > 1)
    title = sum(1 for n in names if n[:1].isupper() and n[1:].islower() is False
                and not n.isupper() and len(n) > 1)
    frac_upper = upper / len(names)
    if frac_upper > 0.6:
        return "human"
    if title / max(len(names), 1) > 0.3 or frac_upper < 0.25:
        return "mouse"
    return "unknown"
