"""FastMCP application: tool registration, resources, prompts, transports.

NOTE on the SDK: the spec was written against `mcp.server.fastmcp.FastMCP`
(SDK 1.x). In SDK 2.0 that class is `mcp.server.MCPServer` — same decorator API
(`.tool`, `.resource`, `.prompt`, `.run`), new import path. We import
defensively so the server works on either.

Logging goes to stderr only. A single `print()` in a tool corrupts the JSON-RPC
stream on stdio; `tests/test_no_stdout.py` and ruff's T20 rule both guard this.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import CONFIG

logger = logging.getLogger(__name__)

try:  # SDK >= 2.0
    from mcp.server import MCPServer as _Server
except ImportError:  # pragma: no cover - SDK 1.x fallback
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

INSTRUCTIONS = """\
skin-mcp exposes skin / dermatology single-cell RNA-seq analysis as composable tools.

Start every session with skin.memory.open_project, then skin.memory.brief. brief()
returns the whole project state — datasets, annotations, parameters, recent steps —
in under 1500 tokens, so you never have to remember it.

Core rules:
- You never see a matrix. Tools take and return `dataset_id` handles. Large tables
  are resources (skin://...), not tool returns.
- Handles are immutable. Every transform mints a new one with a parent pointer.
  skin.io.lineage shows the tree.
- Pseudobulk (skin.de.pseudobulk) is the default DE method. Cell-wise Wilcoxon is
  available but is labelled exploratory in its own return, because its p-values
  are pseudo-replicated across cells within a sample.
- Destructive tools (skin.qc.apply_filters over 30% loss, skin.sub.drop_clusters,
  skin.annotate.regress_markers) require dry_run first, then confirm=True.
- Every tool takes dry_run and seed. Every return carries next_suggested_tools.
- Call skin.help.workflow when you are unsure what to do next; it reads the
  project state and tells you.

Read the SOP prompts (sop_new_project, sop_qc_and_filter, ...) for step-by-step
procedures. They are short and name the exact tools.
"""


def build_server(profile: str | None = None) -> Any:
    """Construct the MCP app and register everything."""
    from . import registry
    from .memory import recall, store

    # Importing the tool modules is what populates _base.REGISTRY.
    from .tools import (  # noqa: F401
        _base,
        abundance_tools,
        annotate_tools,
        atlas_tools,
        ccc_tools,
        cluster_tools,
        de_tools,
        doublet_tools,
        enrich_tools,
        export_tools,
        help_tools,
        integrate_tools,
        io_tools,
        memory_tools,
        meta_tools,
        plot_tools,
        qc_tools,
        runtime_tools,
        subcluster_tools,
        traj_tools,
    )

    if profile:
        CONFIG.profile = profile  # type: ignore[assignment]

    mcp = _Server(
        "skin-mcp",
        version="0.1.0",
        instructions=INSTRUCTIONS,
        description="Dermatology / skin single-cell RNA-seq analysis with persistent "
                    "project memory and reproducible provenance.",
    )

    n_registered = 0
    for name, spec in sorted(_base.REGISTRY.items()):
        if not CONFIG.namespace_enabled(name):
            continue
        mcp.add_tool(spec.fn, name=name, description=(spec.fn.__doc__ or spec.summary))
        n_registered += 1
    logger.info("registered %d tools (profile=%s)", n_registered, CONFIG.profile)

    # ----------------------------------------------------------------------- #
    # resources
    # ----------------------------------------------------------------------- #

    def _active(project_id: str) -> str:
        return project_id or (_base.get_active_project() or "")

    @mcp.resource("skin://project/{project_id}/summary", mime_type="application/json")
    def project_summary(project_id: str) -> str:
        import json

        return json.dumps(recall.brief(_active(project_id)), indent=2, default=str)

    @mcp.resource("skin://project/{project_id}/provenance", mime_type="text/markdown")
    def project_provenance(project_id: str) -> str:
        return recall.as_markdown(_active(project_id))

    @mcp.resource("skin://dataset/{dataset_id}/obs_schema", mime_type="application/json")
    def dataset_obs_schema(dataset_id: str) -> str:
        import json

        import pandas as pd

        pid = _active("")
        adata = registry.load(pid, dataset_id)
        out: dict[str, Any] = {}
        for c in adata.obs.columns:
            s = adata.obs[c]
            if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
                vc = s.astype(str).value_counts()
                out[c] = {"dtype": "categorical", "n_levels": int(vc.size),
                          "levels": vc.head(200).to_dict()}
            else:
                out[c] = {"dtype": str(s.dtype), "min": float(s.min()),
                          "median": float(s.median()), "max": float(s.max())}
        return json.dumps({"dataset_id": dataset_id, "n_obs": int(adata.n_obs),
                           "n_vars": int(adata.n_vars), "obs": out}, indent=2, default=str)

    @mcp.resource("skin://dataset/{dataset_id}/markers/{cluster_key}",
                  mime_type="text/csv")
    def dataset_markers(dataset_id: str, cluster_key: str) -> str:
        import scanpy as sc

        pid = _active("")
        adata = registry.load(pid, dataset_id)
        key = f"rank_genes_{cluster_key}"
        if key not in adata.uns:
            return f"# no marker table under uns[{key!r}]; run skin.cluster.marker_genes first"
        df = sc.get.rank_genes_groups_df(adata, group=None, key=key)
        return df.to_csv(index=False)[: CONFIG.max_resource_bytes]

    @mcp.resource("skin://knowledge/markers/{organism}", mime_type="application/json")
    def knowledge_markers(organism: str) -> str:
        import json

        from . import knowledge as Kn

        return json.dumps(Kn.markers(organism), indent=2, default=str)

    @mcp.resource("skin://figure/{artifact_id}", mime_type="image/png")
    def figure_bytes(artifact_id: str) -> bytes:
        pid = _active("")
        row = store.get_artifact(pid, artifact_id)
        if not row:
            raise ValueError(f"unknown artifact {artifact_id!r}")
        p = Path(row["path"]).with_suffix(".png")
        if not p.exists():
            raise ValueError(f"{p} not found on disk")
        return p.read_bytes()

    @mcp.resource("skin://artifact/{artifact_id}", mime_type="text/plain")
    def artifact_text(artifact_id: str) -> str:
        pid = _active("")
        row = store.get_artifact(pid, artifact_id)
        if not row:
            raise ValueError(f"unknown artifact {artifact_id!r}")
        p = Path(row["path"])
        if not p.exists():
            raise ValueError(f"{p} not found on disk")
        return p.read_text(encoding="utf-8", errors="replace")[: CONFIG.max_resource_bytes]

    @mcp.resource("skin://project/{project_id}/return/{name}", mime_type="application/json")
    def spilled_return(project_id: str, name: str) -> str:
        pid = _active(project_id)
        p = CONFIG.project_dir(pid) / "returns" / Path(name).name
        if not p.exists():
            raise ValueError(f"{name} not found")
        return p.read_text(encoding="utf-8")

    # ----------------------------------------------------------------------- #
    # prompts (SOPs)
    # ----------------------------------------------------------------------- #
    prompt_dir = Path(__file__).parent / "prompts"
    for md in sorted(prompt_dir.glob("*.md")):
        _register_prompt(mcp, md)

    return mcp


def _register_prompt(mcp: Any, path: Path) -> None:
    """Register one SOP markdown file as an MCP prompt.

    The body is closed over, not passed as a default argument: the SDK rejects
    prompt parameters whose names start with an underscore, and a visible
    `text=` parameter would show up in the prompt's argument schema.
    """
    text = path.read_text(encoding="utf-8")
    first = next((ln for ln in text.splitlines() if ln.startswith("# ")), path.stem)
    desc = first.lstrip("# ").strip()

    def make(body: str) -> Any:
        def sop() -> str:
            return body

        return sop

    fn = make(text)
    fn.__name__ = path.stem
    fn.__doc__ = desc
    mcp.prompt(name=path.stem, description=desc)(fn)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _configure_logging(level: str, log_dir: Path | None = None) -> None:
    """stderr plus a file. Never stdout: it is the JSON-RPC channel on stdio.

    The file matters because the usual host is LM Studio, which swallows a
    server's stderr. Without it, "the session lost connection to the MCP server"
    is all anyone can report, and a crash leaves nothing behind to read.
    """
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_dir / "server.log", maxBytes=2_000_000,
                                     backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
            # A hard kill (OOM, segfault) never reaches a Python handler; the
            # faulthandler dump is the only trace it leaves.
            faulthandler.enable(file=open(log_dir / "crash.log", "a"))  # noqa: SIM115
        except OSError as e:
            logger.warning("could not open the log directory %s: %s", log_dir, e)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # scanpy and matplotlib are chatty and some of it goes to stdout.
    for noisy in ("matplotlib", "numba", "h5py", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="skin-mcp",
                                 description="MCP server for skin scRNA-seq analysis")
    ap.add_argument("--transport", choices=["stdio", "http", "streamable-http", "sse"],
                    default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--project-root", default=None,
                    help="where projects live (default ~/.skinmcp)")
    ap.add_argument("--profile", choices=["core", "full"], default=None,
                    help="'core' (default) hides specialist namespaces to keep the "
                         "tool list short; 'full' exposes all 124. Also SKINMCP_PROFILE.")
    ap.add_argument("--offline", action="store_true",
                    help="disable every network call; use shipped snapshots")
    ap.add_argument("--allow-raw-exec", action="store_true",
                    help="enable skin.runtime.exec_r_raw (arbitrary R). Off by default.")
    ap.add_argument("--no-r", action="store_true",
                    help="switch the R backend off; every R-backed tool returns its "
                         "pure-Python equivalent immediately. Also SKINMCP_DISABLE_R.")
    ap.add_argument("--rscript", default=None,
                    help="Rscript to drive (default: first on PATH). Pin this to the R "
                         "minor version renv.lock targets — e.g. a rig-managed "
                         "~/.local/bin/R-4.4 — so upgrading your default R does not "
                         "invalidate the lockfile. Also SKINMCP_RSCRIPT.")
    ap.add_argument("--data-dir", action="append", default=[],
                    help="directory to search when a bare filename is given. Repeatable.")
    ap.add_argument("--cache-objects", type=int, default=None)
    ap.add_argument("--cache-gb", type=float, default=None,
                    help="in-memory AnnData cache cap (default: a fifth of RAM)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    # project_root first: the log file lives under it, and a crash before this
    # point is the one case we genuinely cannot record.
    if args.project_root:
        CONFIG.project_root = Path(args.project_root).expanduser()
    CONFIG.project_root.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.log_level, CONFIG.project_root / "logs")
    CONFIG.offline = bool(args.offline)
    CONFIG.allow_raw_exec = bool(args.allow_raw_exec)
    if args.no_r:
        CONFIG.disable_r = True
    if args.rscript:
        CONFIG.rscript = args.rscript
    for d in args.data_dir:
        dp = Path(d).expanduser().resolve()
        if not dp.is_dir():
            logger.error("--data-dir %s is not a directory", dp)
            return 2
        CONFIG.data_dirs.append(dp)
    if args.cache_objects is not None:
        CONFIG.cache_max_objects = args.cache_objects
    if args.cache_gb is not None:
        CONFIG.cache_max_gb = args.cache_gb
    if args.profile:  # otherwise keep the env-aware default from Config
        CONFIG.profile = args.profile  # type: ignore[assignment]

    logger.info("project_root=%s profile=%s offline=%s data_dirs=%s",
                CONFIG.project_root, CONFIG.profile, CONFIG.offline,
                [str(d) for d in CONFIG.data_dirs])
    # Recorded because the most likely way this process dies is the OS killing
    # it for memory, which leaves no Python traceback at all. If server.log ends
    # abruptly after a load, compare that load's size against these numbers.
    from .config import available_ram_gb, total_ram_gb

    avail = available_ram_gb()
    logger.info("memory: %.1f GB total, %s free; AnnData cache capped at %.1f GB / %d objects",
                total_ram_gb(), f"{avail:.1f} GB" if avail else "unknown",
                CONFIG.cache_max_gb, CONFIG.cache_max_objects)
    # Worth a line because the failure it prevents leaves no other trace: with
    # pyarrow-backed strings, an allocation failure is a SIGSEGV, and the only
    # symptom anyone can report is "Connection closed". See skinmcp/__init__.py.
    import pandas as pd

    storage = pd.get_option("mode.string_storage")
    logger.info("pandas %s, string storage=%s", pd.__version__, storage)
    if storage != "python":
        logger.warning("pandas string storage is %r, so string and categorical work runs "
                       "through pyarrow, whose allocation failures segfault the process "
                       "instead of raising. Unset SKINMCP_ALLOW_ARROW_STRINGS.", storage)
    if avail and avail < 4.0:
        logger.warning("only %.1f GB of RAM is free — a large .h5ad may not fit. If the "
                       "connection drops mid-analysis, free memory (e.g. unload the local "
                       "model) or lower --cache-gb.", avail)

    mcp = build_server()
    transport = "streamable-http" if args.transport == "http" else args.transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
