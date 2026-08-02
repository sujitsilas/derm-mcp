"""A closed, model-readable error taxonomy.

A small model cannot recover from a traceback. It *can* recover from
``{code, message, remedy, suggested_tool}``. Every failure path in this server
must raise a :class:`SkinMCPError`; anything else is a bug and gets wrapped as
``INTERNAL`` with the traceback logged to stderr, not returned.
"""

from __future__ import annotations

from typing import Any, Literal

ErrorCode = Literal[
    "INVALID_HANDLE",
    "MISSING_OBS_KEY",
    "MISSING_VAR",
    "MISSING_COUNTS",
    "INSUFFICIENT_REPLICATES",
    "INSUFFICIENT_CELLS",
    "CONFOUNDED_BATCH",
    "RUNTIME_UNAVAILABLE",
    "NETWORK_UNAVAILABLE",
    "ORGANISM_MISMATCH",
    "AMBIGUOUS_LABELS",
    "MISSING_EMBEDDING",
    "NO_PROJECT",
    "BAD_PARAM",
    "CONFIRMATION_REQUIRED",
    "DEPENDENCY_MISSING",
    "NOT_FOUND",
    "INTERNAL",
]


class SkinMCPError(Exception):
    """Base for every intentional failure. Carries a machine-actionable remedy."""

    code: ErrorCode = "INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        remedy: str = "",
        suggested_tool: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.suggested_tool = suggested_tool
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.remedy:
            d["remedy"] = self.remedy
        if self.suggested_tool:
            d["suggested_tool"] = self.suggested_tool
        if self.details:
            d["details"] = self.details
        return d


def _mk(name: str, code: ErrorCode) -> type[SkinMCPError]:
    return type(name, (SkinMCPError,), {"code": code})


InvalidHandle = _mk("InvalidHandle", "INVALID_HANDLE")
MissingObsKey = _mk("MissingObsKey", "MISSING_OBS_KEY")
MissingVar = _mk("MissingVar", "MISSING_VAR")
MissingCounts = _mk("MissingCounts", "MISSING_COUNTS")
InsufficientReplicates = _mk("InsufficientReplicates", "INSUFFICIENT_REPLICATES")
InsufficientCells = _mk("InsufficientCells", "INSUFFICIENT_CELLS")
ConfoundedBatch = _mk("ConfoundedBatch", "CONFOUNDED_BATCH")
RuntimeUnavailable = _mk("RuntimeUnavailable", "RUNTIME_UNAVAILABLE")
NetworkUnavailable = _mk("NetworkUnavailable", "NETWORK_UNAVAILABLE")
OrganismMismatch = _mk("OrganismMismatch", "ORGANISM_MISMATCH")
AmbiguousLabels = _mk("AmbiguousLabels", "AMBIGUOUS_LABELS")
MissingEmbedding = _mk("MissingEmbedding", "MISSING_EMBEDDING")
NoProject = _mk("NoProject", "NO_PROJECT")
BadParam = _mk("BadParam", "BAD_PARAM")
ConfirmationRequired = _mk("ConfirmationRequired", "CONFIRMATION_REQUIRED")
DependencyMissing = _mk("DependencyMissing", "DEPENDENCY_MISSING")
NotFound = _mk("NotFound", "NOT_FOUND")


# --------------------------------------------------------------------------- #
# Convenience constructors for the failures that fire most often. Centralising
# the remedy text means the model gets the same recovery advice every time.
# --------------------------------------------------------------------------- #

def missing_obs(key: str, available: list[str]) -> SkinMCPError:
    return MissingObsKey(
        f"obs column {key!r} not found",
        remedy=(
            f"Available obs columns: {', '.join(available[:25])}. "
            "Use skin.io.describe to list them, or skin.meta.annotate_samples to add metadata."
        ),
        suggested_tool="skin.io.describe",
        details={"requested": key, "available": available[:25]},
    )


def missing_counts(dataset_id: str) -> SkinMCPError:
    return MissingCounts(
        f"{dataset_id} has no layers['counts']",
        remedy=(
            "Raw integer counts must be preserved for pseudobulk DE and re-normalization. "
            "Re-load the object with skin.io.load_h5ad, or pass allow_no_counts=True to "
            "accept a counts-free handle (DE and subclustering will then be unavailable)."
        ),
        suggested_tool="skin.io.load_h5ad",
    )


def no_embedding(key: str, available: list[str]) -> SkinMCPError:
    return MissingEmbedding(
        f"obsm[{key!r}] not present",
        remedy=(
            f"Available embeddings: {available}. Run skin.integrate.preprocess (X_pca), "
            "then skin.integrate.harmony (X_pca_harmony), then skin.cluster.umap (X_umap)."
        ),
        suggested_tool="skin.integrate.preprocess",
        details={"available": available},
    )


def r_unavailable(reason: str, python_fallback: str) -> SkinMCPError:
    return RuntimeUnavailable(
        f"R runtime unavailable: {reason}",
        remedy=(
            f"Either build it with skin.runtime.create(kind='r'), or use the pure-Python "
            f"fallback `{python_fallback}`, which needs no container."
        ),
        suggested_tool=python_fallback,
    )


def needs_confirm(op: str, impact: str) -> SkinMCPError:
    return ConfirmationRequired(
        f"{op} is destructive and needs explicit confirmation",
        remedy=(
            f"{impact} Re-run the same call with confirm=True once you have reviewed the "
            "dry_run output. Nothing has been changed."
        ),
    )
