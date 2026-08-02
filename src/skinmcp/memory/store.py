"""SQLite-backed project memory.

One database per project at ``{project_root}/{project_id}/memory.db``. Opened in
WAL mode so a long-running HTTP server and an interactive inspection session can
read concurrently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import CONFIG

logger = logging.getLogger(__name__)

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
_LOCK = threading.RLock()
_CONNS: dict[str, sqlite3.Connection] = {}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str, n: int = 8) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:n]}"


def _dumps(obj: Any) -> str:
    from ..returns import jsonable

    return json.dumps(jsonable(obj), ensure_ascii=False, default=str)


def _loads(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def connect(project_id: str) -> sqlite3.Connection:
    """Get (and cache) the connection for a project, creating the schema if new."""
    with _LOCK:
        conn = _CONNS.get(project_id)
        if conn is not None:
            return conn
        root = CONFIG.ensure_project_dirs(project_id)
        conn = sqlite3.connect(root / "memory.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        _CONNS[project_id] = conn
        return conn


def close_all() -> None:
    with _LOCK:
        for c in _CONNS.values():
            try:
                c.close()
            except sqlite3.Error:
                pass
        _CONNS.clear()


def _fts_index(conn: sqlite3.Connection, kind: str, ref_id: Any, project_id: str,
               title: str, body: str) -> None:
    if not body:
        return
    conn.execute(
        "INSERT INTO memory_fts (kind, ref_id, project_id, title, body) VALUES (?,?,?,?,?)",
        (kind, str(ref_id), project_id, title or "", body),
    )


# --------------------------------------------------------------------------- #
# project
# --------------------------------------------------------------------------- #

def create_project(name: str, organism: str, description: str = "",
                   design_notes: str = "", project_id: str | None = None) -> str:
    pid = project_id or new_id("proj", 8)
    conn = connect(pid)
    with _LOCK:
        conn.execute(
            "INSERT OR IGNORE INTO project (project_id,name,organism,created_at,description,design_notes)"
            " VALUES (?,?,?,?,?,?)",
            (pid, name, organism, now(), description, design_notes),
        )
        conn.commit()
    return pid


def get_project(project_id: str) -> dict[str, Any] | None:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM project WHERE project_id=?", (project_id,)).fetchone()
    return dict(row) if row else None


def find_project_by_name(name: str) -> dict[str, Any] | None:
    """Scan project dirs for a matching name so `open_project` can re-attach."""
    root = CONFIG.project_root
    if not root.exists():
        return None
    for d in sorted(root.iterdir()):
        db = d / "memory.db"
        if not db.is_file():
            continue
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM project WHERE name=?", (name,)).fetchone()
            c.close()
            if row:
                return dict(row)
        except sqlite3.Error:
            continue
    return None


def list_projects() -> list[dict[str, Any]]:
    root = CONFIG.project_root
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        db = d / "memory.db"
        if not db.is_file():
            continue
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM project").fetchone()
            n_ds = c.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
            n_step = c.execute("SELECT COUNT(*) FROM step").fetchone()[0]
            c.close()
            if row:
                r = dict(row)
                r["n_datasets"], r["n_steps"] = n_ds, n_step
                out.append(r)
        except sqlite3.Error:
            continue
    return out


# --------------------------------------------------------------------------- #
# dataset / lineage
# --------------------------------------------------------------------------- #

def upsert_dataset(project_id: str, dataset_id: str, *, parent_id: str | None, op: str,
                   params: dict[str, Any], path: str, n_obs: int, n_vars: int,
                   x_state: str, label: str = "") -> None:
    conn = connect(project_id)
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO dataset "
            "(dataset_id,project_id,parent_id,op,params_json,path,n_obs,n_vars,x_state,created_at,label)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT label FROM dataset WHERE dataset_id=?),?))",
            (dataset_id, project_id, parent_id, op, _dumps(params), path, n_obs, n_vars,
             x_state, now(), dataset_id, label),
        )
        conn.commit()


def get_dataset(project_id: str, dataset_id: str) -> dict[str, Any] | None:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM dataset WHERE dataset_id=?", (dataset_id,)).fetchone()
    return dict(row) if row else None


def resolve_dataset_ref(project_id: str, ref: str) -> str | None:
    """Accept either a `ds_…` handle or a human label such as `macs_final`."""
    if not ref:
        return None
    conn = connect(project_id)
    row = conn.execute("SELECT dataset_id FROM dataset WHERE dataset_id=?", (ref,)).fetchone()
    if row:
        return row["dataset_id"]
    row = conn.execute(
        "SELECT dataset_id FROM dataset WHERE label=? ORDER BY created_at DESC LIMIT 1", (ref,)
    ).fetchone()
    return row["dataset_id"] if row else None


def set_label(project_id: str, dataset_id: str, label: str) -> None:
    conn = connect(project_id)
    with _LOCK:
        conn.execute("UPDATE dataset SET label=? WHERE dataset_id=?", (label, dataset_id))
        conn.commit()


def list_datasets(project_id: str) -> list[dict[str, Any]]:
    conn = connect(project_id)
    return [dict(r) for r in conn.execute(
        "SELECT * FROM dataset ORDER BY created_at").fetchall()]


# --------------------------------------------------------------------------- #
# step (append-only provenance)
# --------------------------------------------------------------------------- #

def record_step(project_id: str, *, tool: str, params: dict[str, Any], inputs: Any,
                outputs: Any, code: str, versions: dict[str, str], seed: int,
                started_at: str, duration_s: float, ok: bool, error: str = "") -> int:
    conn = connect(project_id)
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO step (project_id,tool,params_json,inputs_json,outputs_json,code,"
            "versions_json,seed,started_at,duration_s,ok,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (project_id, tool, _dumps(params), _dumps(inputs), _dumps(outputs), code,
             _dumps(versions), int(seed), started_at, float(duration_s), int(ok), error),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_steps(project_id: str, limit: int = 20, only_ok: bool | None = None,
              include_code: bool = False) -> list[dict[str, Any]]:
    conn = connect(project_id)
    cols = "step_id,tool,params_json,outputs_json,seed,started_at,duration_s,ok,error"
    if include_code:
        cols += ",code,versions_json,inputs_json"
    q = f"SELECT {cols} FROM step"
    args: list[Any] = []
    if only_ok is not None:
        q += " WHERE ok=?"
        args.append(int(only_ok))
    q += " ORDER BY step_id DESC LIMIT ?"
    args.append(limit)
    rows = [dict(r) for r in conn.execute(q, args).fetchall()]
    rows.reverse()
    return rows


def all_steps_for_export(project_id: str) -> list[dict[str, Any]]:
    conn = connect(project_id)
    return [dict(r) for r in conn.execute(
        "SELECT * FROM step WHERE ok=1 ORDER BY step_id").fetchall()]


# --------------------------------------------------------------------------- #
# annotation
# --------------------------------------------------------------------------- #

def record_annotation(project_id: str, dataset_id: str, obs_key: str, cluster: str,
                      label: str, evidence: Any, rationale: str, confidence: float,
                      author: str) -> int:
    conn = connect(project_id)
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO annotation (project_id,dataset_id,obs_key,cluster,label,evidence_json,"
            "rationale,confidence,author,superseded_by,created_at) VALUES (?,?,?,?,?,?,?,?,?,NULL,?)",
            (project_id, dataset_id, obs_key, str(cluster), label, _dumps(evidence),
             rationale, float(confidence), author, now()),
        )
        aid = int(cur.lastrowid)
        _fts_index(conn, "annotation", aid, project_id, f"{obs_key}:{cluster} -> {label}", rationale)
        conn.commit()
        return aid


def get_annotations(project_id: str, dataset_id: str = "", obs_key: str = "",
                    include_superseded: bool = False) -> list[dict[str, Any]]:
    conn = connect(project_id)
    q = "SELECT * FROM annotation WHERE 1=1"
    args: list[Any] = []
    if dataset_id:
        q += " AND dataset_id=?"
        args.append(dataset_id)
    if obs_key:
        q += " AND obs_key=?"
        args.append(obs_key)
    if not include_superseded:
        q += " AND superseded_by IS NULL"
    q += " ORDER BY annotation_id"
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def revise_annotation(project_id: str, annotation_id: int, new_label: str,
                      rationale: str, author: str) -> int:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM annotation WHERE annotation_id=?", (annotation_id,)).fetchone()
    if row is None:
        raise KeyError(f"annotation {annotation_id} not found")
    old = dict(row)
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO annotation (project_id,dataset_id,obs_key,cluster,label,evidence_json,"
            "rationale,confidence,author,superseded_by,created_at) VALUES (?,?,?,?,?,?,?,?,?,NULL,?)",
            (project_id, old["dataset_id"], old["obs_key"], old["cluster"], new_label,
             old["evidence_json"], rationale, old["confidence"], author, now()),
        )
        new_id_ = int(cur.lastrowid)
        conn.execute("UPDATE annotation SET superseded_by=? WHERE annotation_id=?",
                     (new_id_, annotation_id))
        _fts_index(conn, "annotation", new_id_, project_id,
                   f"revised {old['obs_key']}:{old['cluster']} -> {new_label}", rationale)
        conn.commit()
    return new_id_


# --------------------------------------------------------------------------- #
# parameter / decision / note / artifact / run
# --------------------------------------------------------------------------- #

def set_param(project_id: str, name: str, value: Any, scope: str, set_by: str,
              rationale: str) -> None:
    conn = connect(project_id)
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO parameter (project_id,name,value_json,scope,set_by,rationale,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (project_id, name, _dumps(value), scope or "global", set_by, rationale, now()),
        )
        _fts_index(conn, "parameter", f"{name}@{scope}", project_id, name, rationale)
        conn.commit()


def get_param(project_id: str, name: str, scope: str = "global") -> dict[str, Any] | None:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM parameter WHERE name=? AND scope=?",
                       (name, scope or "global")).fetchone()
    if row is None and scope not in ("", "global"):
        row = conn.execute("SELECT * FROM parameter WHERE name=? AND scope='global'",
                           (name,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["value"] = _loads(d.pop("value_json"))
    return d


def list_params(project_id: str) -> list[dict[str, Any]]:
    conn = connect(project_id)
    out = []
    for r in conn.execute("SELECT * FROM parameter ORDER BY name, scope").fetchall():
        d = dict(r)
        d["value"] = _loads(d.pop("value_json"))
        out.append(d)
    return out


def record_decision(project_id: str, question: str, choice: str, alternatives: Any,
                    rationale: str, author: str) -> int:
    conn = connect(project_id)
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO decision (project_id,question,choice,alternatives_json,rationale,author,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (project_id, question, choice, _dumps(alternatives), rationale, author, now()),
        )
        did = int(cur.lastrowid)
        _fts_index(conn, "decision", did, project_id, question, f"{choice}. {rationale}")
        conn.commit()
        return did


def get_decisions(project_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = connect(project_id)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM decision ORDER BY decision_id DESC LIMIT ?", (limit,)).fetchall()]
    rows.reverse()
    return rows


def add_note(project_id: str, tag: str, body: str, author: str) -> int:
    conn = connect(project_id)
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO note (project_id,tag,body,author,created_at) VALUES (?,?,?,?,?)",
            (project_id, tag, body, author, now()),
        )
        nid = int(cur.lastrowid)
        _fts_index(conn, "note", nid, project_id, tag, body)
        conn.commit()
        return nid


def get_notes(project_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = connect(project_id)
    return [dict(r) for r in conn.execute(
        "SELECT * FROM note ORDER BY note_id DESC LIMIT ?", (limit,)).fetchall()]


def record_artifact(project_id: str, artifact_id: str, step_id: int | None, kind: str,
                    path: str, caption: str, params: Any) -> None:
    conn = connect(project_id)
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO artifact (artifact_id,project_id,step_id,kind,path,caption,"
            "params_json,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (artifact_id, project_id, step_id, kind, path, caption, _dumps(params), now()),
        )
        conn.commit()


def get_artifact(project_id: str, artifact_id: str) -> dict[str, Any] | None:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM artifact WHERE artifact_id=?", (artifact_id,)).fetchone()
    return dict(row) if row else None


def list_artifacts(project_id: str, kind: str = "", limit: int = 100) -> list[dict[str, Any]]:
    conn = connect(project_id)
    q, args = "SELECT * FROM artifact", []
    if kind:
        q += " WHERE kind=?"
        args.append(kind)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def record_run(project_id: str, run_id: str, kind: str, dataset_id: str,
               params: Any, result: Any) -> None:
    conn = connect(project_id)
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO run (run_id,project_id,kind,dataset_id,params_json,"
            "result_json,created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, project_id, kind, dataset_id, _dumps(params), _dumps(result), now()),
        )
        conn.commit()


def get_run(project_id: str, run_id: str) -> dict[str, Any] | None:
    conn = connect(project_id)
    row = conn.execute("SELECT * FROM run WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = _loads(d.pop("params_json"), {})
    d["result"] = _loads(d.pop("result_json"), {})
    return d


def list_runs(project_id: str, kind: str = "", limit: int = 25) -> list[dict[str, Any]]:
    conn = connect(project_id)
    q, args = "SELECT run_id,kind,dataset_id,created_at FROM run", []
    if kind:
        q += " WHERE kind=?"
        args.append(kind)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(q, args).fetchall()]


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

def search(project_id: str, query: str, limit: int = 15) -> list[dict[str, Any]]:
    conn = connect(project_id)
    try:
        rows = conn.execute(
            "SELECT kind, ref_id, title, snippet(memory_fts, 4, '[', ']', '…', 18) AS snip "
            "FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Bare user input can be an invalid FTS5 expression; fall back to a phrase query.
        safe = '"' + query.replace('"', " ") + '"'
        rows = conn.execute(
            "SELECT kind, ref_id, title, snippet(memory_fts, 4, '[', ']', '…', 18) AS snip "
            "FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?",
            (safe, limit),
        ).fetchall()
    return [dict(r) for r in rows]
