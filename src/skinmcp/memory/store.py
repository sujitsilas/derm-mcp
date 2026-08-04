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
        _migrate(conn)
        conn.commit()
        _CONNS[project_id] = conn
        return conn


#: Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
#: existing database untouched, so new columns need an explicit ALTER.
_ADDED_COLUMNS = (("project", "output_dir", "TEXT"),)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, col, decl in _ADDED_COLUMNS:
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def close_all() -> None:
    with _LOCK:
        for c in _CONNS.values():
            try:
                c.close()
            except sqlite3.Error:
                pass
        _CONNS.clear()


def _remember(project_id: str, kind: str, ref_id: Any, title: str, body: str,
              author: str = "model") -> None:
    """Hand free text to mem0. Replaces the FTS5 index this used to maintain."""
    from . import semantic

    semantic.remember(project_id, kind, f"{title}\n{body}".strip() if title else body,
                      metadata={"ref_id": ref_id, "title": title}, author=author)


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


def set_output_dir(project_id: str, path: str) -> None:
    """Point this project's figures/ and tables/ at `path`."""
    conn = connect(project_id)
    with _LOCK:
        conn.execute("UPDATE project SET output_dir=? WHERE project_id=?", (path, project_id))
        conn.commit()


def get_output_dir(project_id: str) -> str:
    """The configured output directory, or "" to mean the project dir."""
    proj = get_project(project_id) or {}
    return str(proj.get("output_dir") or "")


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
        _remember(project_id, "annotation", aid, f"{obs_key}:{cluster} -> {label}", rationale, author)
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
        _remember(project_id, "annotation", new_id_,
                  f"revised {old['obs_key']}:{old['cluster']} -> {new_label}", rationale, author)
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
        _remember(project_id, "parameter", f"{name}@{scope}", name, rationale, set_by)
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
        _remember(project_id, "decision", did, question, f"{choice}. {rationale}", author)
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
        _remember(project_id, "note", nid, tag, body, author)
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
    """Semantic recall via mem0, with a keyword fallback.

    The fallback scans the same rows this module already stores, so search keeps
    working when LM Studio is not up — degraded to exact matching, never absent.
    """
    from . import semantic

    hits = semantic.recall(project_id, query, limit)
    if hits is not None:
        return [{"kind": h["kind"], "title": (h["metadata"] or {}).get("title", ""),
                 "snip": h["text"][:220], "score": h["score"], "backend": "mem0"}
                for h in hits]

    return _keyword_search(project_id, query, limit)


#: Words carried by almost every question, so matching on them ranks everything
#: equally and tells the caller nothing.
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it of on or "
    "should that the there they this to was we were what when where which who why with "
    "you your".split()
)


def _keyword_search(project_id: str, query: str, limit: int) -> list[dict[str, Any]]:
    """Rank free-text rows by how many query words they contain.

    The previous fallback matched the whole query as one LIKE pattern, so
    "why was a sample excluded?" found nothing even when a note said exactly
    that in other words. Since the default backend is now SQLite rather than
    mem0, this path is what most installs actually use, and it has to work.

    Scoring is deliberately plain — distinct words matched, over words asked —
    because a project holds tens of these rows, not thousands.
    """
    import re

    words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in _STOPWORDS}
    conn = connect(project_id)
    sources = (
        ("annotation", "SELECT obs_key||':'||cluster||' -> '||label AS t, "
                       "COALESCE(rationale,'') AS b FROM annotation"),
        ("decision", "SELECT question AS t, choice||'. '||COALESCE(rationale,'') AS b "
                     "FROM decision"),
        ("note", "SELECT tag AS t, body AS b FROM note"),
        ("parameter", "SELECT name AS t, COALESCE(rationale,'') AS b FROM parameter"),
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for kind, sql in sources:
        for r in conn.execute(sql).fetchall():
            title, body = r["t"] or "", r["b"] or ""
            hay = f"{title} {body}".lower()
            if words:
                hit = {w for w in words if w in hay}
                if not hit:
                    continue
                score = round(len(hit) / len(words), 3)
            elif query.lower().strip() not in hay:
                continue  # no usable words: fall back to a plain containment test
            else:
                score = 1.0
            scored.append((score, {"kind": kind, "title": title, "snip": body[:220],
                                   "score": score, "backend": "keyword"}))
    scored.sort(key=lambda s: -s[0])
    return [h for _, h in scored[:limit]]
