"""skin-mcp — dermatology / skin single-cell RNA-seq analysis as MCP tools.

Importing this package hardens the numeric stack *before* anything touches
pandas. See `_harden_pandas`: the settings below are the difference between a
recoverable error and the whole server process disappearing mid-analysis.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


def _harden_pandas() -> None:
    """Keep pyarrow out of the string/categorical path.

    pandas 3.0 defaults `mode.string_storage` to "auto", which resolves to
    pyarrow whenever pyarrow is installed — and it always is here, because
    several dependencies pull it in. Every `.astype(str)` on an obs column and
    every categorical read out of an .h5ad then runs through pyarrow's C++
    allocator, and there are 170+ such call sites in this package.

    That allocator does not raise MemoryError when it cannot get memory: it
    takes down the process with SIGSEGV. This server is normally hosted by
    LM Studio, beside a resident local model that may hold 30 GB of a 51 GB
    machine, so allocation failure is a routine condition rather than an
    exotic one. Two real crashes, both on the same 78k x 20k object:

        io.describe  -> read_h5ad -> read_categorical
                     -> ArrowStringArray._from_sequence            SIGSEGV
        sub.extract  -> obs[key].astype(str) -> pyarrow.compute.take SIGSEGV

    A segfault never reaches Python, so none of the careful error handling in
    tools/_base.py runs, no step is recorded, and the client sees only
    "MCP error -32000: Connection closed" with the whole session lost.

    "python" storage keeps pandas 3 `str` dtype semantics — it is still a
    StringDtype column, not object — but backs it with numpy, whose allocation
    failure is an ordinary MemoryError that the tool wrapper catches, reports
    and survives. Slower on huge string columns; obs columns here are
    categorical and small, so the trade is heavily in our favour.

    Set SKINMCP_ALLOW_ARROW_STRINGS=1 to opt back out.
    """
    if os.environ.get("SKINMCP_ALLOW_ARROW_STRINGS", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        return
    try:
        pd.set_option("mode.string_storage", "python")
    except (KeyError, ValueError) as e:
        # pandas < 2.1 has no such option, and a future pandas may rename it.
        # Neither is worth failing to start over.
        logger.debug("could not pin pandas string storage: %s", e)


_harden_pandas()
