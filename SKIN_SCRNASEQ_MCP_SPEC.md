# Project Spec — `skin-mcp`: A Modular MCP Server for Dermatology / Skin scRNA-seq Analysis

**Audience:** Claude Code (implementation agent)
**Status:** build-ready spec, v1
**Reference materials in repo root:** `reference/macrophages_resident_recruited.ipynb` (style + analysis source of truth), `reference/mcp.md` (MCP SDK docs)

---

## 0. One-paragraph summary

Build a Python MCP server that exposes skin/dermatology single-cell RNA-seq analysis as a set of small, composable, strongly-typed tools. The server must be driveable by a **small local model** (Qwen-class ~30–35B running via LM Studio/Ollama) as the tool-caller, with a frontier model (Opus 5 / Fable 5) optionally doing the biological reasoning. It never streams matrices through the model: all data lives behind opaque handles on disk. It maintains a **persistent per-project memory** (SQLite) recording every parameter, annotation, and decision so a session can be resumed, audited, and replayed. Every analysis step emits executable code, and the whole project can be exported as a runnable `.ipynb` or `.Rmd`. Pinned Python (uv) and R (Docker + renv) runtimes make the tool-version provenance explicit.

Organisms supported in v1: **Mus musculus** and **Homo sapiens**. Nothing else.

---

## 1. Non-negotiable design principles

1. **The model never sees a matrix.** Tools take and return `dataset_id` handles plus small JSON summaries (target < 2 KB per return). Anything large goes to disk and is exposed as an MCP *resource* URI.
2. **Reproducibility is the product.** Every tool call appends a provenance record (tool, resolved params, input handles, output handles, package versions, wall time, RNG seed) and an executable code cell. If a result cannot be regenerated from the exported notebook, that is a bug.
3. **Small-model ergonomics.** Flat schemas, enums over free text, ≤ 6 required arguments per tool, sensible defaults for everything else, and a `next_suggested_tools` field on every return. Assume the caller has a 32k context and forgets things.
4. **Nothing silently overwrites.** Handles are immutable; every transform mints a new `dataset_id` with a parent pointer. The lineage graph is queryable.
5. **Statistics are conservative by default.** Pseudobulk before cell-wise. Report n_samples, not just n_cells. Refuse-with-explanation rather than silently degrading to an underpowered test.
6. **Honest labels.** Any output produced by a fallback path (Wilcoxon instead of DESeq2, marker-based instead of reference-based annotation) is tagged as such in the return value *and* in the figure caption metadata.

---

## 2. Repository layout

```
skin-mcp/
├── pyproject.toml                 # uv-managed, python >=3.11
├── uv.lock                        # committed
├── README.md
├── reference/
│   ├── macrophages_resident_recruited.ipynb
│   └── mcp.md
├── src/skinmcp/
│   ├── server.py                  # FastMCP app, tool registration, transports
│   ├── config.py                  # project root, cache sizes, offline flag
│   ├── registry.py                # AnnData handle registry + lineage graph
│   ├── returns.py                 # ToolResult envelope, summarizers, truncation
│   ├── errors.py                  # typed, model-readable error taxonomy
│   ├── memory/
│   │   ├── store.py               # SQLite project memory
│   │   ├── schema.sql
│   │   └── recall.py              # search/summarize helpers
│   ├── style/
│   │   ├── rcparams.py            # global matplotlib contract (§8)
│   │   ├── palettes.py            # condition/celltype/subtype palettes
│   │   └── panels.py              # volcano grid, enrichment tile, dotplot, proportions
│   ├── knowledge/
│   │   ├── markers_mouse.yaml
│   │   ├── markers_human.yaml
│   │   ├── contamination.yaml     # lineage-exclusive gene patterns
│   │   ├── platforms.yaml         # QC presets per chemistry (§7.2)
│   │   ├── genesets.yaml          # curated signatures (SAM, LAM, IFN, hypoxia…)
│   │   ├── enrich_libraries.yaml  # curated library recommendations
│   │   └── orthologs_mm_hs.tsv    # static MGI/HGNC homology table (shipped)
│   ├── tools/
│   │   ├── io_tools.py            ├── qc_tools.py       ├── meta_tools.py
│   │   ├── doublet_tools.py       ├── integrate_tools.py├── cluster_tools.py
│   │   ├── annotate_tools.py      ├── subcluster_tools.py
│   │   ├── de_tools.py            ├── enrich_tools.py   ├── abundance_tools.py
│   │   ├── traj_tools.py          ├── ccc_tools.py      ├── plot_tools.py
│   │   ├── atlas_tools.py         ├── runtime_tools.py  ├── export_tools.py
│   │   └── memory_tools.py
│   ├── runtimes/
│   │   ├── python_manifest.py     # importlib.metadata version capture
│   │   ├── r/Dockerfile           # rocker/r-ver pinned
│   │   ├── r/renv.lock
│   │   └── bridge.py              # h5ad <-> SCE round-trip + subprocess exec
│   ├── prompts/                   # MCP prompts = SOPs, one .md per workflow
│   └── vendor/py_monocle/         # vendored, commit-pinned (§7.12)
└── tests/
    ├── golden/                    # small public mouse skin dataset
    ├── test_schemas.py            # every tool: schema size, enum coverage
    ├── test_smallmodel.py         # scripted 20-step run against a 7B model
    └── test_reproducibility.py    # export notebook -> execute -> compare
```

---

## 3. MCP surface

Use the **Python MCP SDK ≥ 2.0.0** with `FastMCP`, per `reference/mcp.md`.

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("skin-mcp")
```

**Transports:** support both. `stdio` for desktop/local clients, `streamable-http` for a lab-shared instance.
```bash
skin-mcp --transport stdio
skin-mcp --transport http --host 0.0.0.0 --port 8931 --project-root /data/skinmcp
```

**Logging:** never write to stdout. `logging.getLogger(__name__)` → stderr only. A single `print()` in a tool corrupts the JSON-RPC stream. Add a CI check that greps for bare `print(` under `src/`.

**Tool naming:** dotted namespaces, verb-last, stable forever:
`skin.io.*`, `skin.qc.*`, `skin.meta.*`, `skin.doublet.*`, `skin.integrate.*`, `skin.cluster.*`, `skin.annotate.*`, `skin.sub.*`, `skin.de.*`, `skin.enrich.*`, `skin.abundance.*`, `skin.traj.*`, `skin.ccc.*`, `skin.plot.*`, `skin.atlas.*`, `skin.runtime.*`, `skin.export.*`, `skin.memory.*`, `skin.help.*`

**Resources** (`@mcp.resource`) for anything the model may want to read but shouldn't get pushed into context automatically:
- `skin://project/{project_id}/summary`
- `skin://dataset/{dataset_id}/obs_schema`
- `skin://dataset/{dataset_id}/markers/{cluster_key}`  (full rank_genes_groups table)
- `skin://project/{project_id}/provenance`
- `skin://figure/{artifact_id}`  (PNG bytes, for vision-capable clients)
- `skin://knowledge/markers/{organism}`

**Prompts** (`@mcp.prompt`) — these are the SOPs that let a small model run the pipeline without a giant system prompt. One per workflow, each a short numbered procedure with the exact tool names:
`sop_new_project`, `sop_qc_and_filter`, `sop_first_pass_annotation`, `sop_decontamination_loop`, `sop_subcluster`, `sop_pseudobulk_de`, `sop_trajectory`, `sop_abundance`, `sop_communication`, `sop_finalize_and_export`.

**Every tool returns this envelope** (`returns.ToolResult`):

```python
{
  "ok": true,
  "dataset_id": "ds_7f3a…",          # new handle, if one was minted
  "summary": {…},                     # <= ~40 scalar fields, no arrays > 20 items
  "warnings": ["…"],                  # human/model-readable, actionable
  "artifacts": [{"id": "...", "kind": "figure|table|h5ad", "path": "...", "uri": "skin://..."}],
  "code": "…",                        # the exact python/R that reproduces this step
  "memory_ref": "step_00042",
  "next_suggested_tools": ["skin.qc.apply_filters", "skin.qc.plot_sample_stats"]
}
```

**Every tool accepts** `dry_run: bool = False` (validate + return resolved params and the code, execute nothing) and `seed: int = 0`.

---

## 4. Handle registry (`registry.py`)

- Project root layout: `{project_root}/{project_id}/{objects,figures,tables,notebooks,runtimes}`.
- `dataset_id` = `ds_` + 8 hex chars, deterministic from (parent_id, tool, resolved_params) so identical operations dedupe.
- Objects persisted as `.h5ad` (gzip). Keep **raw counts in `layers["counts"]`** for the entire lifetime of a project — pseudobulk DE and every re-normalization depend on it. Refuse to mint a handle that lacks `layers["counts"]` unless `allow_no_counts=True`.
- In-memory LRU cache of loaded `AnnData` (default 3 objects or 16 GB, whichever first; configurable — assume a 48 GB laptop).
- Lineage graph: `parent_id`, `op`, `params_hash`. Expose `skin.io.lineage(dataset_id)` returning an ASCII tree.
- `skin.io.describe(dataset_id)` → n_obs, n_vars, obs columns w/ dtype and cardinality (categoricals: up to 20 levels + counts), obsm keys, layers, uns keys, organism, whether X is raw/lognorm/scaled (detect via `X.max()` heuristic as in the notebook, but store it explicitly in `uns["skinmcp"]["x_state"]` — never re-detect).

---

## 5. Project memory (the differentiator)

SQLite at `{project_root}/{project_id}/memory.db`, WAL mode. This is what lets a local model pick up a project three weeks later, and what lets a PI audit how a label was assigned.

### 5.1 Schema

```sql
CREATE TABLE project (
  project_id TEXT PRIMARY KEY, name TEXT, organism TEXT,
  created_at TEXT, description TEXT, design_notes TEXT
);

CREATE TABLE dataset (
  dataset_id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT,
  op TEXT, params_json TEXT, path TEXT, n_obs INT, n_vars INT,
  x_state TEXT, created_at TEXT, label TEXT      -- human alias e.g. "macs_final"
);

CREATE TABLE step (                              -- append-only provenance log
  step_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
  tool TEXT, params_json TEXT, inputs_json TEXT, outputs_json TEXT,
  code TEXT, versions_json TEXT, seed INT,
  started_at TEXT, duration_s REAL, ok INT, error TEXT
);

CREATE TABLE annotation (
  annotation_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
  dataset_id TEXT, obs_key TEXT,                 -- e.g. "macrophage_subtypes"
  cluster TEXT, label TEXT,                      -- "5" -> "LAM-I"
  evidence_json TEXT,                            -- top markers, scores, dotplot artifact ids
  rationale TEXT,                                -- free text, this is the gold
  confidence REAL, author TEXT,                  -- "model:qwen3-35b" | "model:opus-5" | "user:sujit"
  superseded_by INTEGER, created_at TEXT
);

CREATE TABLE parameter (                         -- named, reusable thresholds
  project_id TEXT, name TEXT, value_json TEXT, scope TEXT,
  set_by TEXT, rationale TEXT, created_at TEXT,
  PRIMARY KEY (project_id, name, scope)
);

CREATE TABLE decision (                          -- non-parametric choices
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
  question TEXT, choice TEXT, alternatives_json TEXT,
  rationale TEXT, author TEXT, created_at TEXT
);

CREATE TABLE artifact (
  artifact_id TEXT PRIMARY KEY, project_id TEXT, step_id INT,
  kind TEXT, path TEXT, caption TEXT, params_json TEXT, created_at TEXT
);

CREATE TABLE note (
  note_id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT,
  tag TEXT, body TEXT, author TEXT, created_at TEXT
);
```

FTS5 virtual table over `annotation.rationale`, `decision.rationale`, `note.body`.

### 5.2 Memory tools

| Tool | Purpose |
|---|---|
| `skin.memory.open_project(name, organism, description)` | create/attach; returns `project_id` and a **resume briefing** |
| `skin.memory.brief(project_id)` | ← **the most important tool.** Returns a ≤1500-token state summary: datasets + labels, current annotation sets, open flags, parameters in force, last 10 steps, unresolved warnings |
| `skin.memory.record_annotation(dataset_id, obs_key, mapping, evidence, rationale, confidence, author)` | cluster→label with justification |
| `skin.memory.get_annotations(dataset_id, obs_key, include_superseded=False)` | |
| `skin.memory.revise_annotation(annotation_id, new_label, rationale)` | supersedes, never deletes |
| `skin.memory.set_param(name, value, scope, rationale)` / `get_param` / `list_params` | e.g. `qc.min_genes` scoped to `sample:B_D7_1` |
| `skin.memory.record_decision(question, choice, alternatives, rationale)` | |
| `skin.memory.note(tag, body)` / `skin.memory.search(query)` | FTS over rationales and notes |
| `skin.memory.timeline(project_id, limit)` | compact step log |
| `skin.memory.export(project_id, format="md"\|"json")` | a lab-notebook markdown of the whole project |

**Auto-recording:** the tool decorator writes a `step` row on every call, success or failure. Annotations, parameters, and decisions are *explicit* — the model must call the tool. Make this cheap and put it in every SOP prompt.

**Guardrail:** memory is descriptive, not directive. Never store instructions that alter server behaviour. `set_param` values are *defaults offered to the model*, surfaced in `brief`, and always overridable by an explicit tool argument — they are never silently applied.

---

## 6. Runtime & environment management

The user must be able to answer "which version of scDblFinder produced this?" at any point.

### 6.1 Python
- `uv` with a committed `uv.lock`. Server runs as `uv run skin-mcp`.
- On **every** tool call, `python_manifest.capture()` records versions of the packages that step actually touched (`scanpy`, `anndata`, `harmonypy`, `pydeseq2`, `gseapy`, `liana`, `decoupler`, `scikit-misc`, `igraph`, `leidenalg`, `statsmodels`, `numpy`, `scipy`, `matplotlib`) into `step.versions_json`. Cheap: cache the dict, invalidate never (process-lifetime).

### 6.2 R
Some things have no adequate Python equivalent and should run in R: `CellChat`, `miloR`, `scDblFinder`, `SoupX`/`DecontX`, `DESeq2` (as a cross-check against PyDESeq2), `fgsea`, `Seurat` v5 for users who want it, and `monocle3` if the vendored py-monocle proves inadequate.

- `skin.runtime.create(kind="r", backend="docker"|"renv", force=False)` builds/pulls a pinned image from `runtimes/r/Dockerfile` (base: `rocker/r-ver:4.4.x` at a fixed digest) + `renv.lock`. Emit build logs to a file, return the tail.
- `skin.runtime.status()` → `{python: {...}, r: {backend, image_digest, available, packages: {...}}}`
- `skin.runtime.manifest(project_id)` → full version table for both runtimes, suitable for a methods section. This is a deliverable, not a debug tool.
- `skin.runtime.exec_r(script_id, dataset_id, params)` — **not** arbitrary code from the model. Only named, vetted scripts under `runtimes/r/scripts/*.R` with typed params. Arbitrary R execution is a separate, explicitly opt-in tool (`skin.runtime.exec_r_raw`) disabled unless the server is started with `--allow-raw-exec`.
- **Bridge I/O:** primary path is `zellkonverter::readH5AD/writeH5AD` in the container over a shared temp dir. Fallback (and the path the reference notebook uses, cell 3) is an mtx export: `matrix.mtx`, `genes.txt`, `barcodes.txt`, `metadata.csv`, plus one CSV per `reducedDim`. Implement both; use mtx when h5ad round-trip fails on the object.
- If Docker is unavailable and `renv` isn't bootstrapped, R-backed tools must fail with a **typed** error `RUNTIME_UNAVAILABLE` that names the Python fallback (e.g. "use `skin.abundance.milo_py` instead of `skin.abundance.milo_r`") rather than a stack trace.

### 6.3 Offline mode
`--offline` disables every network call (CellTypist model download, Enrichr, cellxgene Census, MSigDB) and forces use of the shipped snapshots under `knowledge/`. Local models often run air-gapped. Tools must degrade with a clear warning, not hang on a socket timeout. Default network timeout 20 s, one retry.

---

## 7. Tool catalog

Notation: `→` indicates the return summary's key fields. All tools also take `project_id`.

### 7.1 Ingest — `skin.io.*`

| Tool | Notes |
|---|---|
| `load_10x(path, sample_name, organism, chemistry)` | CellRanger `filtered_feature_bc_matrix` (h5 or mtx dir). Also accepts `raw_feature_bc_matrix` for ambient estimation. |
| `load_h5ad(path, organism)` | Validate: counts present? var_names unique? Populate `uns["skinmcp"]`. |
| `load_seurat_rds(path)` | via R bridge → h5ad. |
| `load_mtx_export(dir)` | the SCE-export layout in reference cell 3. |
| `build_multisample(inputs: list[{path, sample, condition, timepoint, batch, ...}])` | concat with `batch_key="Sample"`, `index_unique="_"`; this is the normal entry point |
| `describe`, `lineage`, `save_h5ad(dataset_id, path)`, `set_label(dataset_id, label)` | |

Validation on ingest, all returned as warnings:
- `var_names` duplicated → `make_unique` and warn loudly
- gene symbol casing vs declared organism (mouse `Actb` vs human `ACTB`) — mismatch is a hard error, not a warning
- features file contains Antibody Capture / CRISPR rows → split into `.obsm` and tell the user
- `X` looks already log-normalized on a "raw" load

### 7.2 Sample-wise QC statistics — `skin.qc.*`

This is the first thing the user asked for, and it is threshold **discovery**, not filtering.

`skin.qc.sample_stats(dataset_id, sample_key="Sample")` computes per sample:
- n_cells, median/MAD/quantiles(1,5,25,50,75,95,99) of `total_counts` and `n_genes_by_counts`
- `pct_counts_mt`, `pct_counts_ribo`, `pct_counts_hb` (organism-aware prefixes: mouse `mt-`/`Rp[sl]`/`Hb[ab]-`, human `MT-`/`RP[SL]`/`HB[AB]`)
- complexity `log10(n_genes)/log10(total_counts)`
- top-20-gene fraction; `pct_counts_in_top_50_genes`
- **skin-specific ambient probes**: fraction of counts in `Krt*`, `Col1a*`, `Sbsn/Lor/Flg`, `Hbb-*`. High keratin/collagen ambient is the dominant failure mode in dissected skin.
- estimated empty-droplet profile correlation if a `raw_feature_bc_matrix` was supplied
- expected vs observed doublet rate given loaded cells (10x multiplet table)

`→ {per_sample: [...], flags: [{sample, flag, severity, evidence}], recommended_thresholds: {...}}`

**Flags** (each with severity `info|warn|exclude_candidate`):
`low_cell_count`, `low_median_genes`, `high_mito`, `high_ambient_keratin`, `high_ambient_collagen`, `high_hemoglobin`, `low_complexity`, `outlier_vs_cohort` (sample median > 3 MAD from the cohort median on any of the four core metrics), `saturating_doublet_rate`.

`skin.qc.recommend_thresholds(dataset_id, sample_key, method="mad"|"fixed"|"both", n_mads=3.0)`
- **MAD is the default and the recommended answer.** Compute per-sample on log1p scale, scater-style: outlier if `|x - median| > n_mads * MAD`. Return both per-sample and cohort-wide thresholds so the model can choose.
- `fixed` returns the platform preset from `knowledge/platforms.yaml`.
- Return a `rationale` string the model can paste into `memory.set_param`.

`knowledge/platforms.yaml` presets (starting values — tune, and document every change):

```yaml
# thresholds are FLOORS/CEILINGS applied on top of MAD, not replacements for it
mouse:
  10x_3prime_v3:   {min_genes: 200, max_genes: 6000,  min_counts: 500, max_counts: 60000, max_pct_mt: 10}
  10x_5prime:      {min_genes: 200, max_genes: 6000,  min_counts: 500, max_counts: 60000, max_pct_mt: 10}
  10x_flex:        {min_genes: 200, max_genes: 8000,  min_counts: 400, max_counts: 60000, max_pct_mt: null}
  10x_flex_ffpe:   {min_genes: 150, max_genes: 8000,  min_counts: 300, max_counts: 60000, max_pct_mt: null}
  10x_multiome_gex:{min_genes: 200, max_genes: 6000,  min_counts: 500, max_counts: 60000, max_pct_mt: 5}
  snrna:           {min_genes: 200, max_genes: 6000,  min_counts: 400, max_counts: 40000, max_pct_mt: 5}
  parse_evercode:  {min_genes: 150, max_genes: 5000,  min_counts: 300, max_counts: 30000, max_pct_mt: 10}
  bd_rhapsody:     {min_genes: 150, max_genes: 5000,  min_counts: 300, max_counts: 30000, max_pct_mt: 10}
human:
  10x_3prime_v3:   {min_genes: 200, max_genes: 7500,  min_counts: 500, max_counts: 60000, max_pct_mt: 15}
  10x_5prime:      {min_genes: 200, max_genes: 7500,  min_counts: 500, max_counts: 60000, max_pct_mt: 15}
  10x_flex:        {min_genes: 200, max_genes: 9000,  min_counts: 400, max_counts: 60000, max_pct_mt: null}
  10x_flex_ffpe:   {min_genes: 150, max_genes: 9000,  min_counts: 300, max_counts: 60000, max_pct_mt: null}
  snrna:           {min_genes: 200, max_genes: 7000,  min_counts: 400, max_counts: 40000, max_pct_mt: 5}
```

**Platform rules the code must encode explicitly:**
- **Flex / Fixed RNA Profiling is probe-based.** The probe set covers few or no mitochondrial genes, so `pct_counts_mt` is not a viability metric — `max_pct_mt: null` means *skip the filter and say so*, not "use 0". Substitute a probe-set-aware complexity filter and the ambient-keratin flag.
- **FFPE** has higher ambient and shorter effective libraries. Lower the gene floor, raise the weight on ambient decontamination, and warn that doublet callers are less reliable.
- **snRNA-seq**: mito fraction should be near zero; a high value means cytoplasmic carryover, so the filter is *diagnostic of prep quality*, and the interpretation differs from scRNA.
- **Neutrophils are a trap.** In wound/burn skin, neutrophils legitimately carry 200–600 genes and low counts. A cohort-wide `min_genes` of 500 deletes them and the user will not notice. `recommend_thresholds` must emit a `neutrophil_risk` warning whenever the proposed `min_genes` exceeds 250 and `S100a8/S100a9` (or `S100A8/9`) counts are detectable in the discarded fraction. Report "cells that would be lost, by putative lineage" as part of the preview.

`skin.qc.preview_filters(dataset_id, thresholds)` — dry-run: how many cells/sample would be lost, and their marker profile. Always call before applying.
`skin.qc.apply_filters(dataset_id, thresholds, exclude_samples=[])` → new handle.
`skin.qc.plot_sample_stats(dataset_id)` → violin/scatter grid per sample with threshold lines drawn.
`skin.qc.estimate_ambient(dataset_id, raw_path)` → SoupX/DecontX contamination fraction per sample (R bridge). Optional pluggable backend interface `AmbientBackend` so a custom pipeline (e.g. ProbDecon) can be registered without touching tool code.
`skin.qc.cell_cycle_score(dataset_id)` — organism-aware S/G2M lists.

### 7.3 Metadata & colors — `skin.meta.*`

| Tool | Notes |
|---|---|
| `annotate_samples(dataset_id, table)` | table = list of `{sample, condition, timepoint, batch, sex, replicate, ...}`; validates every sample is covered and errors on partial coverage |
| `parse_sample_names(dataset_id, pattern)` | regex → named groups → obs columns, with a preview |
| `order_categorical(dataset_id, key, order)` | natural timepoint ordering (`D7 < D10 < D14 < D19`, not alphabetical — see reference `order_timepoints`) |
| `make_composite(dataset_id, keys, new_key)` | e.g. `Type_Timepoint` = "Burn D7" |
| `assign_palette(dataset_id, key, scheme, overrides)` | writes `uns[f"{key}_colors"]` in category order |
| `get_palette(dataset_id, key)` | so plotting tools stay consistent across the project |

`assign_palette` schemes, all colorblind-checked:
- `condition` — diverging pair, default Burn/treated `#C0392B`, Sham/control `#2471A3` (matches reference)
- `timepoint` — sequential (viridis or `Blues`), ordered
- `celltype` — a fixed 24-color qualitative palette, assigned by *stable hash of the label string* so "Fibroblasts" is the same color in every project
- `subtype` — the reference macrophage palette is the seeded default (§8.3)
- `manual` — user dict

Palettes are recorded in `memory.parameter` under `palette.{key}` so figures across sessions match.

### 7.4 Doublets — `skin.doublet.*`

- `call(dataset_id, method="scdblfinder"|"scrublet"|"doubletdetection", sample_key="Sample", expected_rate=None)`
  - **Always run per sample**, never on the pooled object. Enforce this; ignore any argument that says otherwise.
  - `scdblfinder` via R bridge (preferred, best benchmarked); `scrublet` as the offline/no-Docker fallback.
  - `expected_rate` defaults to the 10x multiplet table given per-sample recovered cells.
- `→ {per_sample: [{sample, n_called, rate, threshold}], total_rate, warnings}`
- `filter(dataset_id, score_key, threshold=None)` → new handle.
- **Do not filter before clustering.** Emit a warning if `skin.doublet.filter` is called before `skin.cluster.leiden` has ever run on this lineage — the correct workflow is call → cluster → check whether doublet calls concentrate in a cluster → then decide. Homotypic doublets are invisible to these methods and only show up as a cluster with two lineage programs; that is the annotation loop's job (§7.8), not the doublet caller's.
- `cluster_enrichment(dataset_id, cluster_key)` → per-cluster doublet fraction with a binomial test.

### 7.5 Normalization / HVG / PCA — `skin.integrate.*` (part 1)

`preprocess(dataset_id, target_sum=1e4, n_hvg=2000, hvg_flavor="seurat"|"seurat_v3"|"pearson_residuals", scale_max=10, regress_out=[], exclude_genes=[])`

Follows the reference exactly:
```python
adata.X = adata.layers["counts"].copy()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack", use_highly_variable=True)
```
`exclude_genes` accepts pattern groups from `knowledge/contamination.yaml` by name (e.g. `["collagen","keratin","muscle","mito","ribo","hb"]`) — this is how the user removes non-specific genes from the **feature space** before HVG selection. Record what was removed in the return and in memory.

`skin.integrate.harmony(dataset_id, batch_key, basis="X_pca", adjusted_basis="X_pca_harmony", max_iter=20, theta=None, lambda_=None)`
- Wraps `scanpy.external.pp.harmony_integrate`.
- **Guard:** if `batch_key` is confounded with the biological variable of interest (perfect or near-perfect nesting, Cramér's V > 0.9), refuse and explain. Integrating over a confounded key destroys the effect being studied. Return the contingency table.
- `skin.integrate.assess(dataset_id, batch_key, label_key)` → kBET-style / LISI-style batch mixing + label preservation, before vs after. Report both; integration is a tradeoff, not a win.
- Alternatives behind the same signature: `scvi`, `scanorama`, `bbknn` — Harmony is the default because it is what the lab uses.

### 7.6 Neighbors / UMAP / clustering — `skin.cluster.*`

- `neighbors(dataset_id, use_rep="X_pca_harmony", n_neighbors=15, n_pcs=30)`
- `umap(dataset_id, min_dist=0.5, spread=1.0)`
- `leiden(dataset_id, resolution=0.8, key_added=None, flavor="igraph", n_iterations=2, directed=False)` — key defaults to `leiden_res{resolution}`, matching the reference (`leiden_res0.8`). Fall back to the legacy flavor with a warning if `igraph` is unavailable.
- `leiden_sweep(dataset_id, resolutions=[0.2,0.4,0.6,0.8,1.0,1.2])` → per-resolution n_clusters, mean silhouette on the integrated embedding, and cluster-stability (bootstrap ARI). Return a recommendation, not just numbers.
- `marker_genes(dataset_id, groupby, method="wilcoxon", n_genes=50, use_raw=False, pts=True)` → writes `uns`, and exposes the **full table as a resource**, returning only the top 10/cluster in the tool result. This keeps the model's context alive.
- `cluster_qc(dataset_id, cluster_key)` → per cluster: n_cells, median genes/counts, %mt, doublet fraction, sample composition entropy (a cluster from one sample is a batch artifact until proven otherwise).

### 7.7 Cell type annotation — `skin.annotate.*`

`score_lineages(dataset_id, cluster_key, organism)` — `sc.tl.score_genes` for each lineage set in `knowledge/markers_{organism}.yaml`, then a per-cluster mean-score matrix + entropy.

Seed marker sets (mouse; human = ortholog-mapped uppercase, verified against the shipped ortholog table, **not** naive `.upper()` — `Adgre1`→`ADGRE1` works but many do not):

```yaml
Keratinocytes:   [Krt5, Krt14, Krt1, Krt10, Krt15, Dsp, Pkp1, Perp, Lgals7, Col17a1, Trp63, Krtdap, Lor, Flg]
Sebaceous:       [Scd1, Elovl4, Elovl6, Dhcr24, Sdr16c6, Mgst1, Far2, Awat1, Krt79, Cidea]
Hair follicle:   [Lef1, Msx2, Dlx3, Krt25, Krt28, Krt71, Krt73, Tchh, Padi3, Foxn1, Hoxc13, Lhx2, Bnc2]
Endothelial:     [Pecam1, Egfl7, Cldn5, Cdh5, Flt1, Emcn, Kdr, Tie1]
Lymphatic endo:  [Prox1, Lyve1, Pdpn, Flt4, Ccl21a]
Fibroblasts:     [Col1a1, Col1a2, Col3a1, Col5a1, Pdgfra, Dpt, Sparc, Dcn, Lum, Crabp1, Mfap5]
Fibro-activated: [Tnn, Lrrc15, Lox, Tnc, Thbs2, Postn]
Smooth muscle:   [Acta2, Myh11, Tagln, Myl9, Des, Rgs5, Pdgfrb, Notch3]
Skeletal muscle: [Ttn, Neb, Ryr1, Actn3, Tnnt3, Tnnc2, Atp2a1]
Adipocytes:      [Adipoq, Plin1, Lep, Cfd, Fabp4]
Melanocytes:     [Mlana, Pmel, Tyrp1, Dct, Mitf]
Schwann/neural:  [Mpz, Plp1, Sox10, Ncmap]
T cells:         [Cd3e, Cd3d, Cd3g, Cd8a, Trac, Lck, Thy1, Nkg7, Ccl5]
gdT / DETC:      [Trdc, Trgv4, Sox13, Il17a, Rorc, Il23r, Cd163l1, Tcrg-V5]
ILC/NK:          [Ncr1, Klrd1, Klrb1c, Gata3, Rora, Il7r]
B cells:         [Cd79a, Ms4a1, Ighm, Jchain]
Mast cells:      [Cpa3, Cma1, Mcpt4, Kit, Ms4a2]
cDC:             [Cd74, H2-Aa, H2-Ab1, H2-Eb1, Flt3, Xcr1, Itgax, Ciita]
Langerhans:      [Cd207, Epcam, Cd74, Itgax]
Monocytes:       [Ly6c2, Ly6c1, Plac8, Ccr2, Vcan, F13a1, Chil3, Gngt2, Hp, Sell]
Macrophages:     [Lyz2, Ctss, Mrc1, Trem2, C1qa, C1qb, C1qc, Adgre1, Csf1r, Tgfbi, Mmp12, Arg1, Nos2, Stab1, Mertk, Apoe]
Neutrophils:     [S100a8, S100a9, Retnlg, Mpo, Mmp9, Csf3r, Cxcr2, Cxcl2, Il1b, Srgn, Trem1, Acod1]
Proliferating:   [Mki67, Top2a, Stmn1, Birc5, Cdk1, Ncapd2]
```

| Tool | Notes |
|---|---|
| `marker_report(dataset_id, cluster_key, top_n=25)` | per cluster: top DE genes + which lineage sets they hit + a proposed label + confidence. **Proposal only** — never writes obs. |
| `apply_labels(dataset_id, cluster_key, mapping, new_key, order=None, palette=None)` | the `cluster_to_celltype` dict pattern from the reference; validates that every cluster is covered, and errors on unmapped clusters rather than producing NaN |
| `dotplot(dataset_id, groupby, markers, ...)` | §8.4 |
| `celltypist(dataset_id, model)` | see §7.14 |
| `transfer_labels(dataset_id, reference)` | see §7.14 |

### 7.8 Contamination audit & the iterative loop — `skin.annotate.audit_*`, `skin.sub.*`

This is the heart of the user's request: *"check if non-specific markers are being expressed (e.g. keratinocytes expressing Col1a1), regress non-specific markers manually, run an iterative clustering and annotation loop."*

**`skin.annotate.contamination_audit(dataset_id, label_key, organism)`** — per label, compute:

1. **Cross-lineage co-expression rate.** For each pair of mutually-exclusive lineage sets, the fraction of cells with ≥1 detected count in each. Keratinocyte∩Fibroblast, Immune(`Ptprc`)∩Structural, Macrophage∩Keratinocyte, etc. Baseline-corrected against the whole-object rate.
2. **Foreign-program score.** `score_genes` of every *other* lineage set within this label; flag if a foreign score exceeds the native score in > 10 % of cells.
3. **Ambient signature.** Correlation of the label's mean profile with the empty-droplet / soup profile (if available) or with the top-20 most globally-abundant genes. Uniform low-level expression of `Krt`/`Col1a1` across every cluster is ambient, not doublets — different fix.
4. **Doublet concentration** within the label.
5. **Sample skew** — a "cell type" from one mouse.

`→ {per_label: [{label, contamination_score, dominant_foreign_lineage, likely_cause: "ambient"|"doublet"|"mixed_cluster"|"true_biology", evidence, recommended_action}], overall}`

The `likely_cause` heuristic must be spelled out in code comments, because the remedies differ:
- **uniform, low-magnitude, present in every cluster** → ambient → fix with SoupX/DecontX/CellBender upstream, or exclude those genes from the feature space
- **bimodal within a cluster, high-magnitude, elevated doublet score** → heterotypic doublets → remove the cluster or the cells
- **one whole cluster carrying two complete programs, low doublet score** → mixed cluster from under-clustering → raise resolution / subcluster
- **a coherent minority of cells with a genuine dual program (e.g. Arg1⁺Nos2⁺ macrophages)** → true biology → do not remove. The tool must never auto-remove; it proposes, the model or user decides, and the decision goes in `memory.record_decision`.

**`skin.annotate.regress_markers(dataset_id, gene_groups, mode="exclude"|"regress")`**
- `mode="exclude"` (default, and what the reference notebook does): drop the genes from `var` for downstream feature selection and DE.
- `mode="regress"`: `sc.pp.regress_out` on a module score. Slow, distorts variance structure, and rarely the right answer — keep it available, but the return must carry a warning saying so.
- Gene groups are named patterns from `knowledge/contamination.yaml`:

```yaml
collagen:  ['^Col\d+[a-z]{1,2}\d+$']
keratin:   ['^Krt(ap)?\d+']
muscle:    ['^(Acta2|Myh11|Tagln|Cnn\d+|Smtn|Des|Myl\d+|Tpm\d+|Cald\d+)']
cornified: ['^(Lor|Flg|Tchh|Sbsn|Krtdap|Cnfn)$']
ecm_misc:  ['^(Sparc|Fbln2|Tnn|Tnc|Dcn|Postn|Igfbp4|Fn1)$']
stress:    ['^(Vim|Fth1|Ftl1|Srgn)$']
mito:      ['^mt-']
ribo:      ['^Rp[sl]']
hb:        ['^Hb[ab]-']
```
(human variants in the same file; the exact mouse set above reproduces `get_contamination_genes` from the reference notebook.)

**Honest caveat to surface in the tool docstring and in the return warnings:** removing genes from the feature space is *not* decontamination. It stops contaminating genes from driving clusters and from topping DE lists, but the contaminating **counts** remain and still distort library sizes and neighbour graphs. Real ambient removal belongs upstream (`skin.qc.estimate_ambient`). Say this every time; users will otherwise treat gene-dropping as a fix.

**`skin.annotate.refine_loop(dataset_id, label_key, max_rounds=3, resolution_step=0.2, auto_apply=False)`**
Orchestrator, and the only tool allowed to chain others. One round:
1. `contamination_audit`
2. For each label with `contamination_score > threshold` and `likely_cause == "mixed_cluster"`: subcluster it (`skin.sub.extract` → `preprocess` → `harmony` → `leiden` at higher resolution)
3. `marker_report` on the sub-clusters; propose labels
4. Sub-clusters failing canonical-marker checks → propose removal
5. Re-run `contamination_audit`; stop when no label exceeds threshold or `max_rounds` is hit

With `auto_apply=False` (default) it returns a **plan** — an ordered list of tool calls with resolved arguments — for the model or user to approve. With `auto_apply=True` it executes and logs every round as a `decision`. Each round's before/after audit table goes into memory. Cap total runtime; return partial progress on timeout.

**`skin.sub.*` — subclustering:**
| Tool | Notes |
|---|---|
| `extract(dataset_id, label_key, labels, new_label)` | subset **and restore raw counts** from `layers["counts"]`; this is the "renormalize counts" the user asked for — subsetting without re-normalizing and re-selecting HVGs is the single most common subclustering error |
| `pipeline(dataset_id, label_key, labels, resolution, batch_key, exclude_gene_groups)` | convenience: extract → preprocess → harmony → neighbors → umap → leiden → marker_genes, one call, one handle chain, all steps logged individually |
| `drop_clusters(dataset_id, cluster_key, clusters, reason)` | removes contaminating sub-clusters; `reason` is **required** and goes to `memory.decision` |
| `recluster(dataset_id, ...)` | after dropping, re-run the pipeline from `preprocess` — never reuse the old embedding |
| `map_back(sub_dataset_id, parent_dataset_id, obs_key)` | write sub-labels onto the parent by barcode (reference cell 32), reporting the match rate and refusing below 95 % without `force=True` |

Reference macrophage subtypes, shipped as `knowledge/genesets.yaml` priors for skin (these are the populations the user says recur across skin datasets): `Inf. Mono.`, `Early MDM`, `MΦ-Inf`, `MΦ-Act`, `MΦ-IFN/AS DCs`, `LAM-I`, `LAM-II`, `LAM`, `MΦ-Res/Rep`, plus the origin-level grouping `Inflammatory Monocytes` / `MΦ-Recruited` / `MΦ-Resident/Repair` with the Davies/Jenkins-derived gene lists from reference cell 5.

### 7.9 Differential expression — `skin.de.*`

**Pseudobulk is the default and the recommendation.** Cell-wise Wilcoxon is available but must be labelled as exploratory in every return.

`skin.de.pseudobulk(dataset_id, label_key, groups=None, sample_key="Sample", condition_key, contrast=("Burn","Sham"), covariates=["Timepoint"], min_cells=10, min_samples_per_arm=3, exclude_gene_groups=[...], method="pydeseq2")`

Behaviour, mirroring the user's existing analysis:
1. Aggregate **raw counts** (`layers["counts"]`) by `sample_key × label_key`. Sum, never mean.
2. Drop pseudobulk units with `< min_cells` contributing cells.
3. Drop covariate levels that lack both contrast arms — e.g. a D19 timepoint with no Sham for Neutrophils — and *report which were dropped*.
4. Design `~ Timepoint + Type` by default (i.e. `~ {covariates} + {condition_key}`), timepoints pooled as a blocking factor.
5. PyDESeq2: `DeseqDataSet` → `deseq2()` → `DeseqStats(contrast=[condition_key, a, b])` → `lfc_shrink` (apeglm/ashr) on by default; report both shrunk and unshrunk LFC.
6. If any label has `< min_samples_per_arm` replicates in either arm: **do not silently fall back.** Return `ok=true` for the labels that passed and an explicit `skipped: [{label, reason, n_burn, n_sham}]` list. The model can then opt into `skin.de.wilcoxon` deliberately.
7. Genes matching `exclude_gene_groups` removed from the feature space *before* size-factor estimation (defaults: `collagen`, `keratin`, `muscle`, `ecm_misc`, `stress` for immune populations — exactly `get_contamination_genes` in the reference).

`→ {per_label: [{label, n_genes_tested, n_up, n_down, n_samples_a, n_samples_b, table_artifact_id}], design, dropped_levels, skipped, method}`
Full result tables written to `tables/de_{slug}_{a}_vs_{b}.csv` and exposed as resources. Slug function must handle `MΦ` → `mphi` (reference `_slug`).

Other DE tools:
- `skin.de.wilcoxon(...)` — `sc.tl.rank_genes_groups` per label, `method="wilcoxon"`, `pts=True`. Return carries `"inference_level": "cell", "caveat": "p-values are pseudo-replicated across cells within a sample and are not valid for population-level inference"`.
- `skin.de.deseq2_r(...)` — R-bridge DESeq2 cross-check on the same pseudobulk matrix. Useful for reviewer requests.
- `skin.de.compare_methods(...)` — concordance of ranked lists between two DE runs (Spearman, top-N Jaccard). Cheap and worth having when switching cell-wise → pseudobulk.
- `skin.de.pseudobulk_matrix(...)` — export the aggregated count matrix + sample metadata for external use.
- `skin.de.timepoint_interaction(...)` — `~ Type * Timepoint` for when the question is "does the burn effect change over time" rather than "is there a burn effect".

### 7.10 Volcano + enrichment panels — `skin.plot.volcano_grid`, `skin.enrich.*`

**`skin.plot.volcano_grid(de_run_id, labels=None, ncols=None, must_label=[], highlight_genes=[], fdr=0.05, lfc=0.5, n_label=9, xlim=(-8,8))`**

Port `panels.volcano_grid` from reference cell 10 verbatim in behaviour:
- Grid: `ncols = min(4, n_labels)`, `nrows = ceil(n/ncols)`, `figsize=(7*ncols, 7*nrows)`. The user's "2×N" is this layout with `ncols=2`; expose `ncols` so both work.
- Three scatter layers: NS `#D5D8DC` s=5 α=0.4, down/`group_b` `#2471A3` s=10 α=0.8, up/`group_a` `#C0392B` s=10 α=0.8, all `rasterized=True`.
- Dashed guides at `-log10(fdr)` and `±lfc`, `#7F8C8D`, lw 0.9, α 0.6.
- Label selection = `pick_labels`: half by smallest padj, half by largest |LFC|, deduped, per side; union with `must_label` genes (drawn whether or not they pass thresholds); `highlight_genes` rendered in red.
- `adjust_text` with the reference's exact tuning: `expand=(1.3,1.5)`, `force_text=(0.6,0.8)`, `force_static=(0.2,0.3)`, `force_pull=(0.05,0.05)`, `max_move=6`, `min_arrow_len=3`, `only_move={'text':'xy','static':'xy','explode':'xy'}`, `ensure_inside_axes=True`, `time_lim=3.0`, arrows `#7F8C8D` lw 0.6, and the two count-text objects passed as `objects=`.
- Per-panel counts in the top-left, colored by arm.
- Save `.pdf` **and** `.png` at dpi 300 (600 for single-panel figures), `bbox_inches="tight"`.

**`skin.enrich.list_libraries(question_type, organism)`** — the tool that lets the model "reason about the right database". Returns curated candidates with one-line guidance, from `knowledge/enrich_libraries.yaml`:

| question_type | recommended | why |
|---|---|---|
| `broad_biology` | `GO_Biological_Process_2025` | default; verbose but interpretable |
| `coherent_programs` | `MSigDB_Hallmark_2020` (human) / `mh.all` (mouse) | 50 non-redundant programs; best for scoring and for state-space plots |
| `signaling` | `Reactome_Pathways_2024`, `KEGG_2019_Mouse`/`KEGG_2021_Human`, `WikiPathways_2024_Mouse` | |
| `immune_specific` | `MSigDB_C7_ImmuneSigDB`, `Mouse_Gene_Atlas` | |
| `tf_activity` | `TRRUST_Transcription_Factors_2019`, `ChEA_2022`, or `decoupler` + CollecTRI | prefer decoupler over ORA for TF inference |
| `metabolism` | `Hallmark` subset, `KEGG` metabolic branch | |
| `cell_identity` | `CellMarker_2024`, `PanglaoDB_Augmented_2021`, `Azimuth_Cell_Types_2021` | |

Include the caveat: Enrichr's mouse handling maps symbols to human orthologs internally; for mouse-native Hallmark use `gseapy.Msigdb().get_gmt(category="mh.all", dbver="2024.1.Mm")` (reference cell 104) rather than `organism="Mouse"` where it matters.

**`skin.enrich.ora(de_run_id, label, direction="both", library, top_n_terms=5, fdr=0.05, lfc=0.5, exclude_terms=[], background_size=15000)`**
- `gseapy.enrichr` on the up and down gene sets separately, mirroring reference cell 10 §3.
- Fold enrichment = `(k/N) / (n/M_BG)`, `M_BG` default 15000 (configurable — it materially changes the number).
- **`exclude_terms` must be a parameter, never hardcoded.** The reference notebook has a 28-term hardcoded exclusion list (neuro/cartilage/embryonic terms). That list is dataset-specific and, left in the code, it silently biases every future analysis. Ship it as a *named, optional* preset `skin_irrelevant_v1` in `knowledge/enrich_libraries.yaml`, off by default, and record in the return exactly which terms were dropped and why. Any figure produced with an exclusion list carries it in the caption metadata.

**`skin.enrich.gsea(de_run_id, label, library, ranking="stat")`** — `gseapy.prerank` on the full ranked list. Prefer this over ORA when the DE has adequate power; ORA throws away the ranking and depends on arbitrary cutoffs.

**`skin.plot.enrichment_tile(enrich_run_ids, output_name, title, tile_size=0.7, limit=40)`** — port `make_enrichment_tile` from reference cell 10 exactly: pathways as rows (reversed), directions as columns in `[control, treated]` order, tile fill = `LinearSegmentedColormap('br', ['#377EB8','#E41A1C'])` normalized on `-log10(padj)`, alpha `0.4 + 0.6 * (Count - min)/(max - min)`, fold-enrichment printed in bold at tile center, term text truncated at 40 chars, equal aspect, external colorbar labeled `-log10(p_adj)`, dpi 600 pdf + 300 png. Font sizes are large by design (the user renders these for print at figure scale) — keep them parameterized but keep the defaults.

**`skin.plot.de_panel(de_run_id, ...)`** — convenience combining volcano grid + one enrichment tile per label into a single call, since that pairing is the standard output.

### 7.11 Single-cell signature scoring — `skin.enrich.score_*`

- `score_signature(dataset_id, name, genes | library_term, method="score_genes"|"aucell"|"ssgsea")` — `sc.tl.score_genes(..., use_raw=False, random_state=0)` by default; `decoupler` AUCell/ssGSEA for the rank-based alternatives.
- `score_hallmark(dataset_id, sets=["HALLMARK_GLYCOLYSIS","HALLMARK_OXIDATIVE_PHOSPHORYLATION","HALLMARK_HYPOXIA",...], organism)` — reference cells 86 / 92 / 104.
- `score_panel(dataset_id, panel="skin_wound_v1")` — shipped panel: glycolysis, OXPHOS, hypoxia, type-I IFN (`Isg15, Ifit1/3, Rsad2, Irf7, Stat1/2`), type-II IFN, cGAS-STING, inflammation (union of the three inflammatory Hallmark sets, per reference cell 99), resolution/repair, LAM/SAM signature, phagocytosis, efferocytosis, ECM remodeling, proliferation.
- `skin.plot.state_space(dataset_id, x_score, y_score, color_score, groupby, style="ellipse"|"ellipsoid3d")` — the 2D confidence-ellipse and 3D ellipsoid Hallmark state-space plots from reference cells 86/92.
- `skin.plot.score_umap_grid(dataset_id, scores, cmap="RdBu_r", vcenter=0)` — reference cell 5's `style_umap` grid.

### 7.12 Trajectory — `skin.traj.*`

`skin.traj.monocle(dataset_id, cluster_key, root_label, basis="X_umap", n_centroids=20, p_threshold=14, split_by=None)`

Port reference cell 127:
- `py_monocle.learn_graph(matrix=xy, clusters=clu, n_centroids=20, prune=True, p_threshold=14)` → `order_cells(..., root_pr_cells=root)`
- **Root selection** (`_pick_inf_root`): candidate centroids where the root label's fraction ≥ 0.35; prefer MST leaves (degree 1); tie-break by `root_purity + 0.25 * frac_earliest_timepoint`; safety fallback to the nearest centroid actually containing root-label cells. Return `root_purity`, `is_leaf`, and the chosen centroid so the choice is auditable.
- **Sanity metric**: Spearman ρ between normalized pseudotime and the real experimental timepoint. Report it prominently. A trajectory with ρ ≈ 0 across a designed timecourse is a red flag and the return should say so in `warnings`.
- `split_by` (e.g. `Type`) fits an independent graph per condition and plots them side by side — that comparison is the point.
- Rendering: MST stitched into chains at leaves/forks (`_chains_from_edges`), Chaikin corner-cutting ×3, white halo underlay (lw+4) then black line (lw 5), boxed subtype labels at cluster medians, red star at the root with a white stroke, grey context cells behind.

**Vendor `py-monocle`.** It is a third-party port (`github.com/bioturing/py-monocle`), not on PyPI in a maintained form, and installing from a git HEAD is not reproducible. Vendor it under `src/skinmcp/vendor/py_monocle/` at a pinned commit, record that commit in the runtime manifest, and note the upstream license in `NOTICE`.

Alternatives behind the same signature, because no single method should be trusted alone: `skin.traj.paga`, `skin.traj.dpt`, `skin.traj.scfates`, `skin.traj.cellrank` (`RealTimeKernel` + `GPCCA`, as imported in reference cell 0 — real timepoints available means CellRank's real-time kernel is often the better-grounded choice than a UMAP-space principal graph).
`skin.traj.pseudotime_genes(dataset_id, pseudotime_key, groupby)` — genes varying along pseudotime (GAM or binned-correlation), with a heatmap.

### 7.13 Differential abundance — `skin.abundance.*`

`skin.abundance.milo_py(dataset_id, label_key, sample_key, condition_key, contrast, covariates=["Timepoint"], prop=0.10, k=30, mix_thresh=0.70, alpha_fdr=0.10, use_rep="X_pca_harmony")`

Port reference cell 46 (pure-Python Milo):
- kNN graph → union-symmetrize → binarize; sample `prop` of cells as neighbourhood indices, snap each to its local-centroid nearest neighbour, dedupe.
- Counts matrix = neighbourhoods × samples; offset = `log(colSums)`.
- Design `~ C(Timepoint) + Burn`; **rank-check and fall back to `~ Type`** with an explicit warning when timepoint is collinear with condition.
- Per-neighbourhood Poisson GLM, shared dispersion `φ = max(1, median(pearson_chi2/df_resid))`, z = `coef/(se*sqrt(φ))`, BH FDR.
- Annotate neighbourhoods by majority label, "Mixed" below `mix_thresh`.
- Figures: beeswarm (groups ordered by median LFC, colorbar labeled with the two arm names rather than numbers, `RdBu_r` on a `TwoSlopeNorm` clipped at the 98th percentile) and the neighbourhood-graph-on-UMAP with node size ∝ neighbourhood size (reference cell 47).

Also provide, because DA methods disagree and the user will be asked about it:
- `skin.abundance.milo_r` — the real `miloR` with spatial FDR, via bridge
- `skin.abundance.sccoda` — compositional, handles the sum-to-one constraint properly
- `skin.abundance.propeller` — simple, well-powered for designs with real replicates
- `skin.abundance.proportions(dataset_id, label_key, group_keys, sample_key)` → per-sample proportion table + stacked bars + line plots with SEM and significance stars (reference cells 20/21/22, `render_prop_table`, arcsine or logit transform for the test — the reference's `transform`/`msem`/`stars` helpers)

### 7.14 Reference atlas queries — `skin.atlas.*`

The user asked whether a skin atlas can be queried for annotations. Yes, with real caveats.

| Tool | Backend | Notes |
|---|---|---|
| `list_models()` | CellTypist `models.json` | live fetch, cached |
| `celltypist(dataset_id, model, majority_voting=True, over_clustering=None)` | CellTypist | **human only for skin.** `Adult_Human_Skin.pkl` (34 cell types, Reynolds et al., *Science* 2021, Human Skin Cell Atlas) and `Fetal_Human_Skin.pkl` (14 types) exist. `Immune_All_Low/High` for the immune compartment. |
| `census_celltypes(organism, tissue="skin of body", disease=None)` | CELLxGENE Census | inventory of annotated cell types + cell counts + contributing datasets |
| `census_expression(organism, genes, tissue, group_by="cell_type")` | Census | mean/pct expression of a gene set across skin cell types — a fast, external sanity check on a marker before committing to a label |
| `census_reference(organism, tissue, cell_types, max_cells=50000, seed=0)` | Census | download a downsampled reference AnnData for label transfer |
| `transfer_labels(dataset_id, reference_id, method="knn_harmony"\|"scanvi"\|"scarches"\|"ingest")` | | returns per-cell label + confidence + a query↔reference confusion table |
| `marker_lookup(cell_type, organism, source="cellmarker"\|"panglaodb"\|"local")` | shipped snapshots | offline-safe |
| `ortholog_map(genes, from_organism, to_organism)` | shipped MGI/HGNC table | used everywhere human↔mouse conversion is needed |
| `search_datasets(query, organism, tissue)` | CELLxGENE Discover / GEO | discovery only, returns metadata + links, no bulk download |

**Caveats the tools must return, not bury:**
- **There is no mouse skin CellTypist model.** For mouse, either (a) map mouse genes to human orthologs and run `Adult_Human_Skin` with the cross-species mapping flagged as low-confidence, (b) build a custom CellTypist model from a public mouse skin/wound atlas via `skin.atlas.train_model(reference_id, label_key)`, or (c) stay marker-based. Ship (c) as the default and make (b) a first-class, documented path — training a mouse skin model from the lab's own annotated atlases is the highest-value thing this server can accumulate over time.
- `Adult_Human_Skin` is **healthy adult skin**. It has no burn, wound, LAM, or MDM states. Applied to wound data it will confidently map everything onto homeostatic labels. Return a `domain_shift_warning` whenever the query dataset's condition metadata indicates injury/disease and the model is a healthy-tissue model.
- Census `tissue_general` for skin is `"skin of body"`; verify the exact ontology string at build time against the current Census release and pin `census_version` rather than using `"latest"` — Census releases change cell counts and the analysis must not silently shift.
- All of these endpoints need verification at build time. Wrap each in a `@requires_network` decorator that degrades to the shipped snapshot with a warning under `--offline`.

`skin.atlas.train_model(reference_id, label_key, name)` — `celltypist.train` on an annotated in-house object, saved to `{project_root}/models/`, registered in memory, and immediately available to `skin.atlas.celltypist`. This is how the lab's accumulated annotation work becomes reusable.

### 7.15 Cell–cell communication — `skin.ccc.*`

| Tool | Notes |
|---|---|
| `liana(dataset_id, label_key, groupby_context=None, method="rank_aggregate", resource="mouseconsensus"\|"consensus", expr_prop=0.1)` | `liana.mt.rank_aggregate`; run **per context group** (e.g. per `Type × Timepoint`) so conditions are comparable |
| `liana_differential(run_a, run_b, top_n=10, specificity_cutoff=0.05)` | the `diff_table` pattern from reference cell 156: join on LR pair, `delta = lr_expr_a - lr_expr_b`, keep pairs specific in the arm they're up in, top N each direction |
| `plot_differential_bars(diff_ids, by="timepoint")` | horizontal Δ bars per timepoint, arm-colored (reference cell 156) |
| `cellchat_r(dataset_id, label_key, split_by, organism)` | R bridge, CellChatDB mouse/human; runs `computeCommunProb` → `netAnalysis_computeCentrality` per split |
| `cellchat_compare(run_ids)` | `mergeCellChat` → **information-flow scatter** (rankNet) and **pathway × timepoint log2(A/B) heatmap** — the two comparative figures the user's burn/sham work already relies on |
| `plot_chord(run_id, sources, targets, pathways)` | |
| `plot_lr_dotplot(run_id, sources, targets, top_n)` | |

Guard: LR inference on fewer than ~30 cells in a sending or receiving population is noise. Enforce a `min_cells=30` floor per population, drop populations below it, and list the dropped ones.

### 7.16 General plotting — `skin.plot.*`

`umap`, `umap_split(dataset_id, color_key, split_key, ncols)` (the Type × Timepoint grid), `umap_highlight(dataset_id, key, labels)` (highlight subset, everything else `#E6E6E6` — reference cell 32), `dotplot`, `dotplot_clustered` (gene order from hierarchical clustering of the group×gene z-scored profile — reference cell 35 `cluster_gene_order`), `stacked_proportions`, `proportion_lines`, `heatmap`, `violin_qc`, `legend_only(key, orientation)` (standalone legend figure, since the user assembles panels in Illustrator — reference cells 23/34).

Every plot tool: `save_prefix`, writes both `.pdf` (vector, fonttype 42) and `.png` (dpi 300; 600 for hero figures), registers an `artifact`, returns `artifact_id` + path + the plotting code. Never returns image bytes inline unless `return_image=True` (for vision-capable clients).

### 7.17 Export — `skin.export.*`

- `notebook(project_id, format="ipynb"|"rmd"|"both", include_steps=None, path=None)` — assemble every logged `step.code` into a linear, executable document:
  - header cell: project description, organism, session info, the full runtime manifest
  - a params cell with every resolved parameter as literals (no MCP calls, no handles — resolve `dataset_id` to real file paths)
  - one cell per step, with a markdown cell above carrying the tool name, timestamp, and any recorded rationale
  - figures re-generated by the code, not embedded from cache
  - `.Rmd` path emits knitr chunks for R-bridge steps and `reticulate` chunks for Python steps, or a pure-R translation where one exists
- `report(project_id, format="md"|"html")` — a lab-notebook narrative: annotations with rationales, decisions with alternatives, figures inline, warnings summary. This is the PI-facing artifact.
- `bundle(project_id)` — zip of notebook + figures + tables + `uv.lock` + `renv.lock` + `manifest.json` + `memory.db`.
- `methods_paragraph(project_id)` — drafts a methods section from the provenance log with versions and parameters filled in. Draft only; label it as such.

**Acceptance test:** `tests/test_reproducibility.py` runs a 15-step project, exports the notebook, executes it in a clean container, and asserts the resulting `.h5ad` obs columns and the DE result tables match to within floating-point tolerance. If this test doesn't pass, reproducibility is a claim, not a feature.

---

## 8. Figure style contract

All of this lives in `style/` and is applied by a context manager, never by scattered `rcParams.update` calls. Extracted from the reference notebook — match it.

### 8.1 Global rcParams

```python
PUBLICATION = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "Arial",          # fall back to DejaVu Sans with a warning if absent
    "pdf.fonttype": 42, "ps.fonttype": 42,     # editable text in Illustrator — non-negotiable
    "axes.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}
```
Two size profiles: `PUB_LARGE` (the DE/enrichment panels: base font 30, axis labels 28 bold, titles 30 bold, annotations 19) and `PUB_STANDARD` (UMAPs, dotplots: axis labels 24 bold, ticks 17–20). Expose `skin.plot.set_style(profile)` and record it in memory so a project's figures stay internally consistent.

Conventions that apply everywhere: bold axis labels and titles; no top/right spines; no grid; scatter layers `rasterized=True` with vector text on top (keeps PDFs openable); always write `.pdf` **and** `.png`; axis labels `UMAP 1` / `UMAP 2` with ticks removed on UMAPs.

### 8.2 Condition palette
`Burn/treated #C0392B`, `Sham/control #2471A3` (volcanoes) or `#2980B9` (trajectory/DA titles), NS grey `#D5D8DC`, missing/context grey `#E6E6E6`.

### 8.3 Seeded subtype palettes

```python
MAC_COLORS = {
    "MΦ-Inf": "#FA8072", "MΦ-Act": "#E31A1C", "MΦ-IFN/AS DCs": "#1F78B4",
    "Early MDM": "#6A3D9A", "MΦ-Res/Rep": "#33A02C", "LAM": "#FB9A99",
    "LAM-I": "#FDBF6F", "LAM-II": "#B15928", "Inf. Mono.": "#FF7F00",
}
MAC_IDENTITY_COLORS = {   # origin level
    "Inflammatory Monocytes": "#D62728", "MΦ-Recruited": "#F39C12",
    "MΦ-Resident/Repair": "#2CA02C",
}
```
Label matching must be tolerant to `Φ`/`φ`/`M`, spaces, dots, and slashes — reuse the reference's `key = lambda s: re.sub(r'[^a-z0-9]', '', s.lower().replace('Φ','').replace('φ',''))`. Unknown labels get `#999999`, never a silent reassignment.

### 8.4 Dotplot defaults
`standard_scale="var"`, `cmap="plasma_r"` for major cell types (with `dendrogram=True`) and `cmap="Reds"` with `dot_max=1.0` for curated marker panels, `figsize=(0.34*n_genes + 2, 0.5*n_groups + 2)`, legend column width 2.8 with titles `"Scaled\nexpression"` / `"% expressing"`, gene labels 15 bold, group labels 18 bold, lineage group headers 17 bold rotated 30° anchored left.

### 8.5 Filenames
`{figdir}/{plot_type}_{key}_{contrast}.{pdf,png}`, slugged via the reference `_slug` (φ→phi, non-alphanumerics→`_`, lowercased). Figures go to `{project}/figures/{group}/`. Every figure writes a sidecar `.json` with the tool, params, dataset_id, and step_id — so any figure found on a shared drive months later can be traced back.

---

## 9. Making this work for a 30B local model

The failure mode is not intelligence, it's schema overload and context loss. Design for that:

1. **Tool count discipline.** ~70 tools is already a lot. Group aggressively; expose `skin.help.list_tools(category)` and `skin.help.workflow(project_id)` which returns *state-aware* next steps ("you have clusters but no labels → call `skin.annotate.marker_report`"). Consider gating advanced namespaces behind a `--profile full|core` flag; `core` exposes ~25 tools.
2. **Enums, not prose.** `method`, `organism`, `chemistry`, `library`, `direction` are all enums. Never accept a free-text gene-set name where a controlled vocabulary exists.
3. **Idempotency.** Deterministic `dataset_id`s mean a retried call returns the existing handle instead of duplicating work — small models retry constantly.
4. **Typed errors.** `errors.py` defines a closed set: `INVALID_HANDLE`, `MISSING_OBS_KEY`, `MISSING_COUNTS`, `INSUFFICIENT_REPLICATES`, `CONFOUNDED_BATCH`, `RUNTIME_UNAVAILABLE`, `NETWORK_UNAVAILABLE`, `ORGANISM_MISMATCH`, `AMBIGUOUS_LABELS`. Each carries `{code, message, remedy, suggested_tool}`. A small model recovers from `remedy`; it does not recover from a traceback.
5. **Return-size budget.** Hard-truncate every return at 4 KB; anything longer becomes a resource URI and a one-line pointer. Add a test that asserts this for every tool.
6. **`skin.memory.brief` at the top of every SOP prompt.** That single call is what makes a 32k-context model usable across a long project.
7. **Confirmation for destructive ops.** `skin.sub.drop_clusters`, `skin.qc.apply_filters` with >30 % cell loss, and `skin.annotate.regress_markers` require `confirm=True` after a `dry_run`. Two-step, always.
8. **Prompts do the reasoning scaffolding.** Put the biology in `prompts/*.md`, not in tool docstrings — docstrings are always in context and burn tokens; prompts are pulled on demand.

---

## 10. Testing & acceptance

- **Golden dataset:** a small public mouse skin/wound scRNA-seq dataset (2 conditions × 2 timepoints, subsampled to ~15k cells) committed to `tests/golden/` as `.h5ad`. Pick one at build time and document the accession.
- `test_schemas.py` — every tool: JSON schema is < 2 KB, all string params with a fixed domain are enums, ≤ 6 required args, `dry_run` and `seed` present, docstring has an Args block.
- `test_returns.py` — every tool return validates against the `ToolResult` model and is < 4 KB.
- `test_pipeline.py` — end-to-end: load → sample_stats → filters → doublets → preprocess → harmony → leiden → marker_report → apply_labels → contamination_audit → subcluster → pseudobulk DE → volcano+enrichment → export. Assert on the *data behind* figures (DE tables, proportion tables), not on pixels.
- `test_figures.py` — structural assertions only (n_axes, n_collections, colors present in the palette, text objects for forced labels). Pixel-hash comparison is too brittle across matplotlib versions.
- `test_smallmodel.py` — a scripted run against a local 7B model through the MCP client; asserts it completes the core workflow without a schema-validation error. This is the ergonomics test; it will catch schema bloat before users do.
- `test_reproducibility.py` — see §7.17.
- `test_no_stdout.py` — grep `src/` for bare `print(` and `sys.stdout`.

---

## 11. Build order

| Milestone | Contents | Done when |
|---|---|---|
| **M0** | FastMCP skeleton, both transports, `registry.py`, `memory/`, `returns.py`, `errors.py`, `style/`, `skin.io.*`, `skin.memory.*`, `skin.help.*` | a model can load an h5ad, describe it, take notes, and get a brief |
| **M1** | `skin.qc.*`, `skin.meta.*`, `skin.doublet.*`, `platforms.yaml` | sample stats → thresholds → filtered handle, with the neutrophil warning firing correctly on the golden set |
| **M2** | `skin.integrate.*`, `skin.cluster.*`, `skin.annotate.*` (score/report/apply), `skin.plot.umap*`, `dotplot` | first-pass major cell types on the golden set, matching the reference's labels |
| **M3** | `contamination_audit`, `regress_markers`, `skin.sub.*`, `refine_loop` | the iterative decontamination loop runs and logs each round |
| **M4** | `skin.de.*`, `skin.enrich.*`, `volcano_grid`, `enrichment_tile` | reproduces the reference notebook's volcano + GO tile figures from the same input |
| **M5** | `skin.traj.*`, `skin.abundance.*`, `skin.ccc.*`, `score_panel`, `state_space` | reproduces the monocle, Milo beeswarm, and LIANA differential figures |
| **M6** | `skin.runtime.*` (Docker R + renv), `skin.atlas.*`, offline snapshots | R tools work, manifest is complete, `--offline` degrades cleanly |
| **M7** | `skin.export.*`, prompts, small-model harness, docs | reproducibility test passes; a 7B model completes the core workflow |

M0–M4 is a genuinely useful product on its own. Ship it before starting M5.

---

## 12. Things worth pushing back on / decide before building

These are real risks in the request as written. Resolve them explicitly rather than discovering them at M4.

1. **Gene exclusion is not decontamination.** The reference notebook's `get_contamination_genes` drops collagen/keratin/muscle genes before DE. That is a reasonable *presentation* guard, but the contaminating counts still inflate library sizes and shape the neighbour graph. If ambient is the real problem, it must be fixed at the count level (SoupX / CellBender / DecontX) before normalization. Build the exclusion path because it reproduces existing figures, but make `skin.qc.estimate_ambient` a prominent, recommended step and say the quiet part in the warnings.

2. **The hardcoded `EXCLUDE_TERMS` list.** 28 GO terms are filtered out of every enrichment result in the reference notebook. Some are defensible (neuronal terms in a skin dataset), some are not (`Epithelial To Mesenchymal Transition`, `Mesenchymal Cell Differentiation` — plausibly real in wound fibroblasts). Baking this into a shared tool would propagate one project's judgment call to every user of the server. It must be an opt-in named preset, recorded in figure metadata whenever used.

3. **Cell-wise Wilcoxon vs pseudobulk.** The reference's volcano figures come from `rank_genes_groups` on cells. With n=3–4 mice per arm, cell-wise p-values are pseudo-replicated and the false-positive rate is high; the counts printed on the volcano panels ("Burn↑ 412") are not comparable to pseudobulk counts and will drop substantially. Pseudobulk is the right default and matches where the burn atlas work is already heading — but be prepared for the figures to look different, and keep `compare_methods` available to quantify the shift for reviewers.

4. **py-monocle is unmaintained.** A pinned vendored copy is the minimum. Consider whether CellRank's `RealTimeKernel` — which uses the actual experimental timepoints rather than inferring order from UMAP geometry — is the better primary method for designed timecourses, with the monocle graph kept for figure continuity.

5. **Trajectories fit in UMAP space.** `learn_graph(matrix=X_umap)` fits the principal graph on a 2D non-linear embedding. It produces the figures the lab wants, but the geometry is UMAP's, not the data's. Offer `basis="X_pca_harmony"` and report the pseudotime↔timepoint correlation for both so the difference is visible.

6. **No mouse skin reference model.** Everything in the atlas layer is human-first. Budget the work for `skin.atlas.train_model` early — a CellTypist model trained on the lab's own annotated mouse skin atlases is more valuable to this server than any external database, and it's the piece nobody else can provide.

7. **Scope of "local memory."** Provenance + annotations + parameters + decisions is a well-defined, high-value store. Resist letting it become a general-purpose agent scratchpad; if the model can write arbitrary directives into memory that later change server behaviour, the reproducibility guarantee is gone. Memory describes what happened; it never instructs.

8. **Server-side orchestration vs model-side.** `refine_loop` is the one tool that calls other tools. Every additional orchestrator makes the server easier for weak models and harder to audit. Keep the count at one, and make its plan-mode output the default.

---

## 13. Quick reference — reference notebook cell map

| What | Cell(s) | Use for |
|---|---|---|
| Style + UMAP score grid, origin gene sets | 0, 5, 6 | `style/`, `score_panel`, `score_umap_grid` |
| SCE mtx export round-trip | 3 | `runtimes/bridge.py` fallback |
| DE + volcano + GO tile (full pattern) | 10, 11, 13 | `de_tools`, `panels.volcano_grid`, `panels.enrichment_tile` |
| `get_contamination_genes`, `prepare_celltype_for_DE`, `_slug`, `pick_labels` | 10 | `knowledge/contamination.yaml`, `de_tools` |
| Proportion tables, transforms, significance stars | 14, 20, 21, 22 | `abundance.proportions` |
| Dotplot styling, gene-order clustering | 16, 34, 35, 36 | `plot.dotplot`, `dotplot_clustered` |
| `mac_colors`, cluster→subtype mapping | 29, 31 | `style/palettes.py`, `annotate.apply_labels` |
| Highlight-subset UMAP, map-back by barcode | 32 | `plot.umap_highlight`, `sub.map_back` |
| Milo-style DA + beeswarm + nhood graph | 46, 47 | `abundance.milo_py` |
| Hallmark scoring, state-space ellipse / 3D | 86, 92, 104 | `enrich.score_hallmark`, `plot.state_space` |
| Major cell type cluster→label maps | 87, 88 | seed for `knowledge/markers_*.yaml` |
| py-monocle trajectory, root picking, smoothing | 127 | `traj.monocle` |
| Harmony re-subclustering pipeline | 129, 152, 160 | `sub.pipeline` |
| T cell / fibroblast subclustering + markers | 152, 153, 160 | `knowledge/markers_*.yaml` subtype sets |
| LIANA runs + differential LR table + bars | 150, 156 | `ccc.liana`, `ccc.liana_differential` |

---

## 14. Deliverables checklist

- [ ] `skin-mcp` installable via `uv tool install`, runs on stdio and HTTP
- [ ] `claude_desktop_config.json` / LM Studio / Ollama connection snippets in the README
- [ ] All ~70 tools documented with Args blocks and at least one example call
- [ ] 10 SOP prompts
- [ ] Knowledge YAMLs populated for mouse and human
- [ ] R runtime image builds reproducibly from a pinned base digest
- [ ] Golden-dataset end-to-end test green in CI
- [ ] Exported `.ipynb` from the golden run executes clean in a fresh container
- [ ] `--offline` mode verified with the network disabled
