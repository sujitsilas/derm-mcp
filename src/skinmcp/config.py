"""Server-wide configuration.

Everything mutable at runtime lives on the single `CONFIG` object, which is
populated once from CLI flags / environment in `server.main()` and then treated
as read-only. Tools read it; they never write it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Profile = Literal["core", "full"]

#: Tool namespaces exposed under ``--profile core``. Everything else is gated,
#: because a 30B model with 70 tool schemas in context stops calling tools well.
#:
#: The bar is "does a normal analysis need this?", not "is this simple?".
#: `skin.abundance` was gated once and a session asking for a proportions plot —
#: which `skin.abundance.proportions` produces directly — could not see the tool,
#: improvised, and failed. Composition, export and runtime management are all part
#: of ordinary work. What stays gated is genuinely specialist: cell-cell
#: communication, trajectories, atlas mapping, benchmarking.
CORE_NAMESPACES = frozenset(
    {"skin.help", "skin.memory", "skin.io", "skin.qc", "skin.meta", "skin.doublet",
     "skin.integrate", "skin.cluster", "skin.annotate", "skin.sub", "skin.de",
     "skin.enrich", "skin.plot", "skin.abundance", "skin.export", "skin.runtime"}
)


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def _env_str(name: str, default: str) -> str:
    """Like os.environ.get, but treats set-but-empty as unset.

    Hosts routinely pass `env: {"SKINMCP_MEM0_BASE_URL": ""}` or export a bare
    `VAR=`, and taking that literally gives an empty base URL or an empty
    backend name — which does not equal "sqlite", so the memory backend silently
    switched itself back on.
    """
    return os.environ.get(name) or default


def total_ram_gb() -> float:
    """Physical RAM in GB, or 0.0 where the platform will not say."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def available_ram_gb() -> float:
    """Free RAM in GB, or 0.0 if it cannot be measured on this platform."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001 - psutil is optional; never fail startup
        return 0.0


def _default_cache_gb() -> float:
    """Half of *free* RAM, clamped to [2, 16]. Overridable with --cache-gb.

    Free rather than total: the local LLM driving this server is typically
    resident already (a 35B model is ~29 GB of a 51 GB machine), so a share of
    total RAM would hand the cache memory that does not exist and get the
    server OOM-killed mid-analysis — which the host reports only as
    "the session lost connection to the MCP server".
    """
    ram = available_ram_gb() or total_ram_gb() * 0.2
    return 4.0 if ram <= 0 else max(2.0, min(16.0, round(ram * 0.5, 1)))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- storage ---
    project_root: Path = field(
        default_factory=lambda: _env_path("SKINMCP_PROJECT_ROOT", Path.home() / ".skinmcp")
    )

    # --- in-memory AnnData cache: whichever limit trips first ---
    # The GB cap is a share of physical RAM rather than a fixed number, because
    # this server usually runs beside the local LLM that is driving it, and that
    # model may already hold 20 GB+. A flat 16 GB cache was most of the machine.
    cache_max_objects: int = 2
    cache_max_gb: float = field(default_factory=lambda: _default_cache_gb())

    # --- registry .h5ad codec ---
    # Every tool that mints a handle writes an .h5ad, and the client waits for it.
    # On a 675 MB object: gzip 4.8 s / 126 MB, lzf 0.6 s / 246 MB, none 0.3 s /
    # 675 MB. lzf is 8x faster to write and 2.7x faster to read back for 120 MB
    # of extra disk — the right trade for interactive files we rewrite constantly.
    # lzf ships inside h5py, so anything reading these via anndata can read them;
    # it is not an HDF5 standard filter, so files leaving Python (the R bridge,
    # skin.io.export_h5ad) use gzip or none instead.
    h5ad_compression: str | None = "lzf"

    # --- network ---
    offline: bool = field(default_factory=lambda: _env_bool("SKINMCP_OFFLINE"))
    network_timeout_s: float = 20.0
    network_retries: int = 1

    # --- runtimes ---
    # Python is managed by uv, R by renv. Both are plain local environments
    # created inside the project directory; there is no container layer.
    allow_raw_exec: bool = False

    #: Turn the R backend off entirely. Every R-backed tool then returns its
    #: typed Python fallback immediately, without probing for an interpreter or
    #: a library.
    #:
    #: Worth having as one switch rather than a half-built library: a partially
    #: restored renv is *worse* than none, because tools discover it is broken
    #: one `library()` call at a time, deep inside an analysis. Off means the
    #: fallback is chosen up front, which is a supported path -- pseudobulk DE
    #: uses PyDESeq2, doublets use scrublet, abundance uses milo_py.
    disable_r: bool = field(default_factory=lambda: _env_bool("SKINMCP_DISABLE_R"))

    #: Which Rscript to drive. Empty means "first on PATH".
    #:
    #: Bioconductor pins each release to one R minor version (3.20 -> R 4.4,
    #: 3.23 -> R 4.6) and publishes binaries for that version only, so an
    #: renv.lock and the R it was cut against are a matched pair. Taking
    #: whatever is on PATH means a routine `brew upgrade r` silently
    #: invalidates the lockfile. Pointing this at a specific interpreter — a
    #: `rig`-managed R, a framework path, a wrapper script — makes the pairing
    #: explicit and survives upgrades of the user's default R.
    rscript: str = field(default_factory=lambda: _env_str("SKINMCP_RSCRIPT", ""))

    #: Directories to search when a caller gives a bare filename. Repeatable,
    #: and grown automatically as files are loaded, so "macs.h5ad" keeps working
    #: without the caller having to know the full path.
    data_dirs: list[Path] = field(
        default_factory=lambda: [Path(p).expanduser()
                                 for p in os.environ.get("SKINMCP_DATA_DIR", "").split(os.pathsep)
                                 if p]
    )

    # --- memory ---
    # Free-text memory (rationales, decisions, notes) can go through mem0 for
    # semantic recall, configured against a local LM Studio so nothing leaves
    # the machine.
    #
    # Default "sqlite" (keyword recall) rather than "auto", because mem0 is
    # not free: it adds ~700ms to every open_project while it builds a client
    # against the local model, and it only pays for itself once a project has
    # accumulated enough free-text that keyword search starts missing things.
    # Measured on four real projects: 131 steps recorded, but one single note —
    # so the vector store held exactly one embedding.
    # Set SKINMCP_MEMORY_BACKEND=auto (and install the `memory` extra) to enable.
    memory_backend: str = field(
        default_factory=lambda: _env_str("SKINMCP_MEMORY_BACKEND", "sqlite"))
    mem0_provider: str = "lmstudio"
    mem0_base_url: str = field(
        default_factory=lambda: _env_str("SKINMCP_MEM0_BASE_URL",
                                    "http://localhost:1234/v1"))
    mem0_llm_model: str = field(
        default_factory=lambda: _env_str("SKINMCP_MEM0_LLM", "local-model"))
    mem0_embed_model: str = field(
        default_factory=lambda: _env_str("SKINMCP_MEM0_EMBED",
                                  "text-embedding-nomic-embed-text-v1.5"))
    mem0_embed_dims: int = 768

    # --- ergonomics ---
    # "core" by default for the reason documented on CORE_NAMESPACES: tool
    # selection in local models degrades sharply as the schema list grows, and
    # `full` is 124 tools. The gated namespaces are specialist (ccc, traj,
    # atlas, runtime); nothing in the standard QC -> cluster -> annotate -> DE
    # path is hidden. Set SKINMCP_PROFILE=full to expose everything.
    profile: Profile = field(
        default_factory=lambda: "full" if os.environ.get("SKINMCP_PROFILE") == "full" else "core"
    )
    #: Hard ceiling on the JSON size of any tool return (§9.5). Anything larger
    #: is spilled to a file and replaced by a resource URI.
    max_return_bytes: int = 4096
    #: Ceiling on a *resource* read; resources are opt-in so this can be generous.
    max_resource_bytes: int = 1_000_000

    # --- figures ---
    fig_dpi_png: int = 300
    fig_dpi_hero: int = 600
    style_profile: str = "standard"

    # --- pinned external versions (recorded in the manifest, not resolved live) ---
    census_version: str = "2025-01-30"

    def effective_cache_gb(self) -> float:
        """The cache cap *right now*, not the one computed at start-up.

        `cache_max_gb` is fixed when the server starts, but the local LLM that
        drives it usually loads afterwards. One real session started with
        41 GB free and sized the cache at 16 GB; six minutes later the model
        was resident, 10 GB was free, and the cache was still working to a
        16 GB budget the machine no longer had. Re-reading free RAM on every
        eviction keeps the cache proportional to what is actually available,
        and an explicit --cache-gb still acts as the ceiling.
        """
        free = available_ram_gb()
        if free <= 0:
            return self.cache_max_gb          # platform will not say; trust the flag
        return max(1.0, min(self.cache_max_gb, free * 0.5))

    def project_dir(self, project_id: str) -> Path:
        return self.project_root / project_id

    def runtime_dir(self) -> Path:
        """Where the managed runtimes live — one Python venv, one renv library.

        Shared across projects, not per-project. Building them costs minutes and
        hundreds of MB (the venv gets skin-mcp and all of scanpy/pydeseq2), and
        on a single-user workstation there is no reason to pay that again for
        every new project. Built once, then left alone.
        """
        d = self.project_root / "runtimes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shared_venv(self) -> Path:
        return self.runtime_dir() / "python" / "venv"

    def shared_python(self) -> Path:
        """The managed interpreter. Windows venvs use Scripts\\python.exe."""
        venv = self.shared_venv()
        win = venv / "Scripts" / "python.exe"
        return win if win.exists() else venv / "bin" / "python"

    def shared_renv(self) -> Path:
        return self.runtime_dir() / "r"

    def exec_dir(self, project_id: str) -> Path:
        """Scratch for `exec_python` / `exec_r_raw` scripts. Per project: the code
        is part of that project's record, even though the runtime is shared."""
        d = self.project_dir(project_id) / "runtimes" / "exec"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_input(self, path: str) -> Path:
        """Resolve a user-supplied data path.

        An absolute path is used as given. A bare or relative one is searched
        for in each data directory, then one level down, so a caller can say
        "macs.h5ad" without knowing the exact layout.
        """
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        for d in self.data_dirs:
            cand = d / p
            if cand.exists():
                return cand
        for d in self.data_dirs:
            hits = sorted(d.glob(f"*/{p.name}")) if p.name else []
            if hits:
                return hits[0]
        return p

    def register_data_dir(self, path: Path) -> Path | None:
        """Remember `path`'s directory so later bare filenames resolve."""
        d = Path(path).expanduser().resolve()
        d = d if d.is_dir() else d.parent
        for known in self.data_dirs:
            k = known.resolve()
            if d == k or k in d.parents:
                return None
        self.data_dirs.append(d)
        return d

    def ensure_project_dirs(self, project_id: str) -> Path:
        root = self.project_dir(project_id)
        for sub in ("objects", "figures", "tables", "notebooks", "runtimes", "returns", "logs"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        return root

    def namespace_enabled(self, tool_name: str) -> bool:
        if self.profile == "full":
            return True
        ns = ".".join(tool_name.split(".")[:2])
        return ns in CORE_NAMESPACES


CONFIG = Config()
