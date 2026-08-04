-- skin-mcp project memory. WAL mode, append-only provenance.
-- Memory is DESCRIPTIVE, never DIRECTIVE: nothing stored here changes server
-- behaviour. Parameters are defaults *offered* to the model via brief(); they
-- are never silently applied to a tool call.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS project (
  project_id   TEXT PRIMARY KEY,
  name         TEXT,
  organism     TEXT,
  created_at   TEXT,
  description  TEXT,
  design_notes TEXT,
  -- Where figures and tables are written. Empty = the project dir under
  -- ~/.skinmcp, which is hidden and often unreadable by the host app; on first
  -- load we point it at the directory the .h5ad came from instead.
  output_dir   TEXT
);

CREATE TABLE IF NOT EXISTS dataset (
  dataset_id TEXT PRIMARY KEY,
  project_id TEXT,
  parent_id  TEXT,
  op         TEXT,
  params_json TEXT,
  path       TEXT,
  n_obs      INT,
  n_vars     INT,
  x_state    TEXT,
  created_at TEXT,
  label      TEXT
);
CREATE INDEX IF NOT EXISTS ix_dataset_project ON dataset(project_id);
CREATE INDEX IF NOT EXISTS ix_dataset_parent  ON dataset(parent_id);

CREATE TABLE IF NOT EXISTS step (
  step_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id    TEXT,
  tool          TEXT,
  params_json   TEXT,
  inputs_json   TEXT,
  outputs_json  TEXT,
  code          TEXT,
  versions_json TEXT,
  seed          INT,
  started_at    TEXT,
  duration_s    REAL,
  ok            INT,
  error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_step_project ON step(project_id, step_id);

CREATE TABLE IF NOT EXISTS annotation (
  annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id    TEXT,
  dataset_id    TEXT,
  obs_key       TEXT,
  cluster       TEXT,
  label         TEXT,
  evidence_json TEXT,
  rationale     TEXT,
  confidence    REAL,
  author        TEXT,
  superseded_by INTEGER,
  created_at    TEXT
);
CREATE INDEX IF NOT EXISTS ix_annotation_ds ON annotation(dataset_id, obs_key);

CREATE TABLE IF NOT EXISTS parameter (
  project_id TEXT,
  name       TEXT,
  value_json TEXT,
  scope      TEXT,
  set_by     TEXT,
  rationale  TEXT,
  created_at TEXT,
  PRIMARY KEY (project_id, name, scope)
);

CREATE TABLE IF NOT EXISTS decision (
  decision_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id        TEXT,
  question          TEXT,
  choice            TEXT,
  alternatives_json TEXT,
  rationale         TEXT,
  author            TEXT,
  created_at        TEXT
);

CREATE TABLE IF NOT EXISTS artifact (
  artifact_id TEXT PRIMARY KEY,
  project_id  TEXT,
  step_id     INT,
  kind        TEXT,
  path        TEXT,
  caption     TEXT,
  params_json TEXT,
  created_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_artifact_project ON artifact(project_id);

CREATE TABLE IF NOT EXISTS note (
  note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id TEXT,
  tag        TEXT,
  body       TEXT,
  author     TEXT,
  created_at TEXT
);

-- Named analysis runs (DE, enrichment, CCC, trajectory, abundance) so later
-- tools can refer to `de_run_id` / `enrich_run_id` without re-passing params.
CREATE TABLE IF NOT EXISTS run (
  run_id      TEXT PRIMARY KEY,
  project_id  TEXT,
  kind        TEXT,
  dataset_id  TEXT,
  params_json TEXT,
  result_json TEXT,
  created_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_run_project ON run(project_id, kind);

-- Free-text reasoning (annotation rationales, decisions, notes) is stored and
-- recalled semantically by mem0 (see memory/semantic.py). The rows above remain
-- the durable record; mem0 owns the index over them. The former FTS5 virtual
-- table has been removed.
