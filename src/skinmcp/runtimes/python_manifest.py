"""Capture the exact package versions behind each step.

"Which version of scDblFinder produced this?" must be answerable at any point.
Version lookup is cached for the process lifetime — packages cannot change under
a running interpreter, so invalidation is unnecessary.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from functools import lru_cache

#: Packages whose version materially changes a result. Keep this list explicit;
#: dumping `pip freeze` into every step row makes the provenance log unreadable.
TRACKED = (
    "scanpy", "anndata", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
    "statsmodels", "harmonypy", "pydeseq2", "gseapy", "decoupler", "leidenalg",
    "igraph", "umap-learn", "pynndescent", "scikit-misc", "adjustText", "seaborn",
    "networkx", "h5py", "mcp", "liana", "celltypist", "cellrank", "cellxgene-census",
    "scanorama", "bbknn", "scvi-tools", "doubletdetection", "skin-mcp",
)

#: Which packages each tool namespace actually touches, so `step.versions_json`
#: records the relevant subset rather than everything.
NAMESPACE_PACKAGES: dict[str, tuple[str, ...]] = {
    "skin.io": ("scanpy", "anndata", "h5py", "numpy", "pandas"),
    "skin.qc": ("scanpy", "anndata", "numpy", "pandas", "scipy"),
    "skin.meta": ("anndata", "pandas", "matplotlib"),
    "skin.doublet": ("scanpy", "scikit-learn", "numpy", "doubletdetection"),
    "skin.integrate": ("scanpy", "harmonypy", "scikit-misc", "numpy", "scipy",
                       "scanorama", "bbknn", "scvi-tools"),
    "skin.cluster": ("scanpy", "leidenalg", "igraph", "umap-learn", "pynndescent",
                     "scikit-learn"),
    "skin.annotate": ("scanpy", "numpy", "pandas", "celltypist"),
    "skin.sub": ("scanpy", "harmonypy", "leidenalg", "numpy"),
    "skin.de": ("scanpy", "pydeseq2", "statsmodels", "numpy", "pandas"),
    "skin.enrich": ("gseapy", "decoupler", "pandas", "numpy"),
    "skin.abundance": ("statsmodels", "scanpy", "numpy", "scipy", "matplotlib"),
    "skin.traj": ("scanpy", "scipy", "numpy", "networkx", "cellrank"),
    "skin.ccc": ("liana", "scanpy", "pandas", "numpy"),
    "skin.plot": ("matplotlib", "seaborn", "adjustText", "scanpy", "numpy"),
    "skin.atlas": ("celltypist", "cellxgene-census", "scanpy", "anndata"),
    "skin.export": ("nbformat", "anndata", "scanpy"),
    "skin.runtime": ("mcp", "skin-mcp"),
    "skin.memory": (),
    "skin.help": (),
}


@lru_cache(maxsize=1)
def _all_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for pkg in TRACKED:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def capture(tool_name: str = "") -> dict[str, str]:
    """Version dict for the packages this tool's namespace actually touches."""
    allv = _all_versions()
    if not tool_name:
        return dict(allv)
    ns = ".".join(tool_name.split(".")[:2])
    wanted = NAMESPACE_PACKAGES.get(ns)
    if wanted is None:
        return dict(allv)
    d = {p: allv[p] for p in wanted if p in allv}
    d["python"] = platform.python_version()
    if "skin-mcp" in allv:
        d["skin-mcp"] = allv["skin-mcp"]
    return d


def full_manifest() -> dict[str, object]:
    """The methods-section-grade version table."""
    from ..config import CONFIG

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": _all_versions(),
        },
        "pinned": {
            "py_monocle": _py_monocle_provenance(),
            "cellxgene_census_version": CONFIG.census_version,
        },
    }


def _py_monocle_provenance() -> str:
    """Which trajectory implementation is actually available, resolved now.

    This used to publish a hardcoded `py_monocle_commit`, which was not a real
    commit in bioturing-org/py-monocle — and nothing ever fetched it, because
    skin.traj.monocle imports upstream py_monocle when installed and otherwise
    uses the shipped implementation. A commit hash that names nothing is worse
    than no hash at all here: `full_manifest` is what gets pasted into a methods
    section, so a reader would cite a commit that cannot be checked out.
    """
    try:
        return f"upstream py_monocle {importlib.metadata.version('py-monocle')}"
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import py_monocle  # type: ignore  # noqa: F401
    except ImportError:
        return ("shipped implementation (SimplePPT, Mao et al. SDM 2015; monocle3, "
                "Cao et al. Nature 2019) — upstream py-monocle not installed")
    return "upstream py_monocle (version not reported)"
