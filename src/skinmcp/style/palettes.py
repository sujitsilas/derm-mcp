"""Condition / timepoint / celltype / subtype palettes (§8.2, §8.3).

Cell-type colours are assigned by a *stable hash of the label string*, so
"Fibroblasts" is the same colour in every project and every figure, forever.
Unknown labels get `#999999` — never a silent reassignment that makes two
figures disagree.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

# --- condition (§8.2) ------------------------------------------------------- #
BURN = "#C0392B"      # treated / injured arm, volcano + titles
SHAM = "#2471A3"      # control arm, volcano
SHAM_ALT = "#2980B9"  # control arm in trajectory / DA titles
NS_GREY = "#D5D8DC"   # non-significant points
CONTEXT_GREY = "#E6E6E6"  # cells outside the highlighted subset
UNKNOWN = "#999999"

CONDITION_COLORS: dict[str, str] = {
    "burn": BURN, "treated": BURN, "treatment": BURN, "wound": BURN, "wounded": BURN,
    "injury": BURN, "injured": BURN, "disease": BURN, "lesional": BURN, "case": BURN,
    "sham": SHAM, "control": SHAM, "ctrl": SHAM, "untreated": SHAM, "healthy": SHAM,
    "naive": SHAM, "unwounded": SHAM, "nonlesional": SHAM, "vehicle": SHAM, "wt": SHAM,
}

# --- macrophage subtypes (§8.3, reference cell 29) -------------------------- #
MAC_COLORS: dict[str, str] = {
    "MΦ-Inf": "#FA8072",
    "MΦ-Act": "#E31A1C",
    "MΦ-IFN/AS DCs": "#1F78B4",
    "Early MDM": "#6A3D9A",
    "MΦ-Res/Rep": "#33A02C",
    "LAM": "#FB9A99",
    "LAM-I": "#FDBF6F",
    "LAM-II": "#B15928",
    "Inf. Mono.": "#FF7F00",
}

MAC_IDENTITY_COLORS: dict[str, str] = {
    "Inflammatory Monocytes": "#D62728",
    "MΦ-Recruited": "#F39C12",
    "MΦ-Resident/Repair": "#2CA02C",
}

#: 24-colour qualitative palette, colourblind-checked (Okabe-Ito extended with
#: Paired/Dark2 entries chosen for luminance separation).
QUAL24: tuple[str, ...] = (
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2",
    "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC", "#1F78B4", "#33A02C",
    "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A",
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#8DD3C7", "#BC80BD",
)


def norm_key(s: str) -> str:
    """Label matching tolerant to Φ/φ/M, spaces, dots and slashes (§8.3)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("Φ", "").replace("φ", ""))


def _lookup(table: dict[str, str], label: str) -> str | None:
    nk = norm_key(label)
    for k, v in table.items():
        if norm_key(k) == nk:
            return v
    return None


def stable_color(label: str, palette: Iterable[str] = QUAL24) -> str:
    """Deterministic colour for a label, stable across projects and processes.

    Uses blake2b rather than `hash()`, which is salted per interpreter and would
    give a different colour on every restart.
    """
    pal = list(palette)
    h = hashlib.blake2b(norm_key(label).encode(), digest_size=8).hexdigest()
    return pal[int(h, 16) % len(pal)]


def condition_palette(levels: Iterable[str]) -> dict[str, str]:
    """Map condition levels onto the diverging treated/control pair."""
    levels = list(levels)
    out: dict[str, str] = {}
    unmatched: list[str] = []
    for lv in levels:
        c = CONDITION_COLORS.get(str(lv).strip().lower())
        if c:
            out[lv] = c
        else:
            unmatched.append(lv)
    if len(unmatched) == 2 and not out:
        # Two unrecognised levels: assume (control, treated) in sorted order.
        a, b = sorted(unmatched, key=str)
        out[a], out[b] = SHAM, BURN
    else:
        for lv in unmatched:
            out[lv] = stable_color(lv)
    return out


def timepoint_palette(levels: Iterable[str], cmap: str = "Blues") -> dict[str, str]:
    """Sequential, ordered by the numeric part of the label (D7 < D10 < D14)."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    lv = order_timepoints(levels)
    n = max(len(lv), 2)
    m = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else __import__(
        "matplotlib.pyplot", fromlist=["get_cmap"]).get_cmap(cmap)
    # Skip the palest 30% so the earliest timepoint is still visible on white.
    return {t: mcolors.to_hex(m(0.30 + 0.70 * i / (n - 1))) for i, t in enumerate(lv)}


def celltype_palette(labels: Iterable[str]) -> dict[str, str]:
    """Stable-hash colours, with the seeded subtype tables taking precedence.

    Collisions are resolved by walking the qualitative palette, so two distinct
    labels never share a colour inside a single figure.
    """
    out: dict[str, str] = {}
    used: set[str] = set()
    for lb in labels:
        c = _lookup(MAC_COLORS, lb) or _lookup(MAC_IDENTITY_COLORS, lb) or stable_color(lb)
        if c in used:
            for cand in QUAL24:
                if cand not in used:
                    c = cand
                    break
        out[lb] = c
        used.add(c)
    return out


def subtype_palette(labels: Iterable[str]) -> dict[str, str]:
    """Macrophage subtype palette; unknown labels get grey, never a reassignment."""
    out: dict[str, str] = {}
    for lb in labels:
        out[lb] = _lookup(MAC_COLORS, lb) or _lookup(MAC_IDENTITY_COLORS, lb) or UNKNOWN
    return out


SCHEMES = ("condition", "timepoint", "celltype", "subtype", "manual")


def build(scheme: str, levels: Iterable[str], overrides: dict[str, str] | None = None,
          cmap: str = "Blues") -> dict[str, str]:
    levels = list(levels)
    if scheme == "condition":
        pal = condition_palette(levels)
    elif scheme == "timepoint":
        pal = timepoint_palette(levels, cmap=cmap)
    elif scheme == "celltype":
        pal = celltype_palette(levels)
    elif scheme == "subtype":
        pal = subtype_palette(levels)
    elif scheme == "manual":
        pal = {lv: UNKNOWN for lv in levels}
    else:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    for k, v in (overrides or {}).items():
        pal[k] = v
    return {lv: pal.get(lv, UNKNOWN) for lv in levels}


# --------------------------------------------------------------------------- #
# categorical ordering
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def order_timepoints(levels: Iterable[str]) -> list[str]:
    """Natural ordering: D7 < D10 < D14 < D19, not alphabetical."""
    lv = [str(x) for x in levels]

    def key(s: str) -> tuple[int, float, str]:
        m = _NUM.search(s)
        return (0, float(m.group(1)), s) if m else (1, 0.0, s)

    return sorted(dict.fromkeys(lv), key=key)


def natural_order(levels: Iterable[str]) -> list[str]:
    """Natural sort over mixed alphanumeric labels (cluster ids, sample names)."""
    def key(s: str) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

    return sorted(dict.fromkeys(str(x) for x in levels), key=key)


def apply_to_adata(adata: Any, key: str, palette: dict[str, str]) -> None:
    """Write `uns[f"{key}_colors"]` in the category order scanpy expects."""
    import pandas as pd

    s = adata.obs[key]
    if not isinstance(s.dtype, pd.CategoricalDtype):
        adata.obs[key] = s.astype("category")
        s = adata.obs[key]
    cats = list(s.cat.categories)
    adata.uns[f"{key}_colors"] = [palette.get(c, UNKNOWN) for c in cats]


def get_from_adata(adata: Any, key: str) -> dict[str, str]:
    import pandas as pd

    s = adata.obs.get(key)
    if s is None:
        return {}
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("category")
    cols = adata.uns.get(f"{key}_colors")
    if cols is None:
        return {}
    return {str(c): str(col) for c, col in zip(s.cat.categories, cols)}
