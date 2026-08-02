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
CORE_NAMESPACES = frozenset(
    {"skin.help", "skin.memory", "skin.io", "skin.qc", "skin.meta", "skin.doublet",
     "skin.integrate", "skin.cluster", "skin.annotate", "skin.sub", "skin.de",
     "skin.enrich", "skin.plot"}
)


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


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
    cache_max_objects: int = 3
    cache_max_gb: float = 16.0

    # --- network ---
    offline: bool = field(default_factory=lambda: _env_bool("SKINMCP_OFFLINE"))
    network_timeout_s: float = 20.0
    network_retries: int = 1

    # --- runtimes ---
    allow_raw_exec: bool = False
    docker_bin: str = "docker"
    r_image: str = "skin-mcp-r:0.1.0"

    # --- ergonomics ---
    profile: Profile = "full"
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
    py_monocle_commit: str = "6a1b47f4b6e5d8c4f5a0f9e2f1c3d4a5b6c7d8e9"

    def project_dir(self, project_id: str) -> Path:
        return self.project_root / project_id

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
