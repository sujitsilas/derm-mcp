"""The global matplotlib contract (§8.1).

Applied through a context manager, never by scattered `rcParams.update` calls,
so one tool cannot leak style into another. `pdf.fonttype: 42` is
non-negotiable: it keeps text editable in Illustrator, which is how these
figures get assembled for print.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_FONT_WARNED = False

BASE: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}

#: DE / enrichment panels. Deliberately large — these are rendered at figure
#: scale for print, not for on-screen reading.
PUB_LARGE: dict[str, Any] = {
    **BASE,
    "font.size": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "axes.labelsize": 28,
    "axes.labelweight": "bold",
    "axes.titlesize": 30,
    "axes.titleweight": "bold",
    "legend.fontsize": 15,
}

#: UMAPs, dotplots, QC panels.
PUB_STANDARD: dict[str, Any] = {
    **BASE,
    "font.size": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "legend.fontsize": 13,
}

PROFILES: dict[str, dict[str, Any]] = {
    "large": PUB_LARGE,
    "standard": PUB_STANDARD,
    "publication": PUB_STANDARD,
}

#: Extra sizes the panel code reads directly (they have no rcParam equivalent).
ANNOT: dict[str, dict[str, Any]] = {
    "large": {"annot_fs": 19, "count_fs": 15, "border_lw": 1.6, "tile_value_fs": 35,
              "tile_tick_fs": 53, "cbar_label_fs": 48, "cbar_tick_fs": 22},
    "standard": {"annot_fs": 13, "count_fs": 12, "border_lw": 1.4, "tile_value_fs": 20,
                 "tile_tick_fs": 20, "cbar_label_fs": 18, "cbar_tick_fs": 13},
}
ANNOT["publication"] = ANNOT["standard"]


def _resolve_font(params: dict[str, Any]) -> dict[str, Any]:
    """Fall back to DejaVu Sans with one warning if Arial is not installed."""
    global _FONT_WARNED
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    if params.get("font.family") == "Arial" and "Arial" not in have:
        out = dict(params)
        out["font.family"] = "DejaVu Sans"
        if not _FONT_WARNED:
            logger.warning("Arial not found; figures will use DejaVu Sans. "
                           "Install Arial for exact reference-figure parity.")
            _FONT_WARNED = True
        return out
    return params


def font_warning() -> str:
    """Non-empty when the requested font was substituted, for tool warnings."""
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    if "Arial" not in have:
        return ("Arial is not installed; figures rendered with DejaVu Sans. Text metrics "
                "differ slightly from the reference figures.")
    return ""


def annot(profile: str = "") -> dict[str, Any]:
    from ..config import CONFIG

    return ANNOT.get(profile or CONFIG.style_profile, ANNOT["standard"])


@contextmanager
def style(profile: str = "", **overrides: Any) -> Iterator[dict[str, Any]]:
    """Apply a style profile for the duration of a figure.

    Also forces the Agg backend: an MCP server has no display, and a stray
    interactive backend will block a tool call forever.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from ..config import CONFIG

    prof = profile or CONFIG.style_profile
    params = _resolve_font(PROFILES.get(prof, PUB_STANDARD))
    params = {**params, **overrides}
    with plt.rc_context(params):
        yield annot(prof)


def savefig(fig: Any, path: Any, *, dpi_png: int | None = None, hero: bool = False) -> dict[str, str]:
    """Write both `.pdf` (vector, editable text) and `.png`. Always both.

    Returns the two paths so the caller can register them as artifacts.
    """
    from pathlib import Path

    from ..config import CONFIG

    p = Path(path)
    pdf = p.with_suffix(".pdf")
    png = p.with_suffix(".png")
    dpi = dpi_png or (CONFIG.fig_dpi_hero if hero else CONFIG.fig_dpi_png)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    return {"pdf": str(pdf), "png": str(png)}
