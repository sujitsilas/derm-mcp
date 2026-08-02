# skin-mcp

An MCP server that exposes dermatology / skin single-cell RNA-seq analysis as
small, composable, strongly-typed tools — with persistent per-project memory and
provenance good enough that the exported notebook actually reproduces the result.

Built to be driven by a **small local model** (Qwen-class, 30–35B via LM Studio or
Ollama) as the tool-caller, with a frontier model optionally doing the biological
reasoning. Organisms supported: **Mus musculus** and **Homo sapiens**.

---

## What makes it different

**The model never sees a matrix.** Tools take and return `ds_xxxxxxxx` handles plus
small JSON summaries. Every return is hard-capped at 4 KB; anything larger spills to
a file and comes back as a `skin://` resource URI.

**Persistent project memory.** A SQLite database per project records every
parameter, annotation, decision and step. `skin.memory.brief()` returns the entire
project state in under 1500 tokens, so a 32k-context model can pick up a project
three weeks later — and a PI can audit how any label was assigned. Memory is
*descriptive, never directive*: nothing stored in it changes server behaviour.

**Reproducibility is the product.** Every tool call appends a provenance row with
resolved parameters, package versions, seed and wall time, plus the executable code
for that step. `skin.export.notebook` assembles those into a runnable `.ipynb` with
handles resolved to real paths. The acceptance test executes that notebook in a clean
namespace and asserts the DE table it produces matches the server's to floating-point
tolerance — see `tests/test_reproducibility.py`.

**Conservative statistics, honest labels.** Pseudobulk is the default DE method and
reports n_samples, not n_cells. A cell type with too few replicates is *skipped with
its counts*, never silently downgraded to a cell-wise test. Anything produced by a
fallback path (Wilcoxon instead of DESeq2, cross-species label transfer, a
healthy-tissue reference on injured data) is tagged as such in the return and in the
figure's caption metadata.

---

## Install

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd derm-mcp
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

# optional heavier backends, each with a documented fallback:
uv pip install --python .venv/bin/python -e ".[atlas]"   # CellTypist, CELLxGENE Census
uv pip install --python .venv/bin/python -e ".[ccc]"     # LIANA
uv pip install --python .venv/bin/python -e ".[traj]"    # CellRank
```

Verify:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/skin-mcp --help
```

## Run

```bash
# desktop / local clients
skin-mcp --transport stdio

# lab-shared instance
skin-mcp --transport http --host 0.0.0.0 --port 8931 --project-root /data/skinmcp

# a 30B local model does better with fewer schemas in context
skin-mcp --transport stdio --profile core

# air-gapped: no CellTypist download, no Enrichr, no Census
skin-mcp --transport stdio --offline
```

| flag | default | what it does |
|---|---|---|
| `--project-root` | `~/.skinmcp` | where projects, objects, figures and memory live |
| `--profile` | `full` | `core` gates the advanced namespaces |
| `--offline` | off | disables every network call; shipped snapshots only |
| `--allow-raw-exec` | off | enables `skin.runtime.exec_r_raw` (arbitrary R) |
| `--cache-objects` / `--cache-gb` | 3 / 16 | LRU cache of loaded AnnData |

### Client configuration

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "skin-mcp": {
      "command": "/absolute/path/to/derm-mcp/.venv/bin/skin-mcp",
      "args": ["--transport", "stdio", "--project-root", "/data/skinmcp"]
    }
  }
}
```

**LM Studio** — `~/.lmstudio/mcp.json`:

```json
{
  "mcpServers": {
    "skin-mcp": {
      "command": "/absolute/path/to/derm-mcp/.venv/bin/skin-mcp",
      "args": ["--transport", "stdio", "--profile", "core"]
    }
  }
}
```

**Ollama / any HTTP client** — start with `--transport http` and point the client at
`http://127.0.0.1:8931/mcp`.

**Claude Code** — `claude mcp add skin-mcp -- /path/to/.venv/bin/skin-mcp --transport stdio`

> `--profile core` is recommended for models under ~70B. Tool schemas are context you
> spend before the model has seen your data.

---

## A first session

```
skin.memory.open_project(name="burn_sham_2026", organism="mouse",
                         design_notes="Burn vs Sham, D7/D14, n=3 mice per arm")

skin.io.build_multisample(inputs=[
  {"path": "/data/B_D7_1/filtered_feature_bc_matrix.h5", "sample": "B_D7_1",
   "condition": "Burn", "timepoint": "D7"},
  ...
])

skin.qc.sample_stats(dataset_id=...)            # discovery, not filtering
skin.qc.recommend_thresholds(dataset_id=...)    # MAD-based, with a rationale string
skin.qc.preview_filters(dataset_id=..., thresholds={...})   # what would be lost, by lineage
skin.qc.apply_filters(dataset_id=..., thresholds={...})

skin.doublet.call(dataset_id=...)               # per sample; does NOT filter
skin.integrate.preprocess(...) → harmony(...) → cluster.neighbors/umap/leiden(...)
skin.cluster.marker_genes(groupby="leiden_res0.8")
skin.annotate.marker_report(...)                # proposes; never writes obs
skin.annotate.apply_labels(...)                 # you decide
skin.memory.record_annotation(...)              # why — this is the part that lasts

skin.annotate.contamination_audit(label_key="cell_types")
skin.de.pseudobulk(label_key="cell_types", condition_key="Type",
                   contrast=["Burn","Sham"], covariates=["Timepoint"])
skin.plot.de_panel(de_run_id=...)               # volcano grid + GO tile per label

skin.export.notebook(fmt="both")
skin.export.report(fmt="md")
```

Lost? `skin.help.workflow()` reads the project state and tells you what to do next.
`skin.help.list_tools(category="qc")` browses the catalogue without loading it all.

Ten SOP prompts (`sop_new_project`, `sop_qc_and_filter`, `sop_decontamination_loop`,
…) are registered as MCP prompts — short numbered procedures naming the exact tools.

---

## Tool namespaces

| namespace | what it covers |
|---|---|
| `skin.memory.*` | open/resume, **brief**, annotations, parameters, decisions, notes, FTS search, export |
| `skin.io.*` | 10x / h5ad / mtx / Seurat ingest, describe, lineage, save, cache |
| `skin.qc.*` | per-sample stats, MAD thresholds, preview, filter, ambient RNA, cell cycle |
| `skin.meta.*` | sample tables, regex parsing, categorical order, composites, palettes |
| `skin.doublet.*` | per-sample calling, cluster enrichment, filtering |
| `skin.integrate.*` | preprocess, Harmony (+ scVI/Scanorama/BBKNN), integration assessment |
| `skin.cluster.*` | neighbours, UMAP, Leiden, resolution sweep, markers, cluster QC |
| `skin.annotate.*` | lineage scoring, marker report, apply labels, **contamination audit**, refine loop |
| `skin.sub.*` | extract, subcluster pipeline, drop clusters, recluster, map back |
| `skin.de.*` | **pseudobulk**, Wilcoxon, interaction, matrix export, method comparison, DESeq2-R |
| `skin.enrich.*` | library guidance, ORA, GSEA, TF activity, signature/Hallmark/panel scoring |
| `skin.abundance.*` | Milo (Python + R), scCODA, proportions with stats |
| `skin.traj.*` | principal graph, PAGA, DPT, CellRank, pseudotime genes |
| `skin.ccc.*` | LIANA, differential LR, CellChat, chord/dotplot |
| `skin.plot.*` | UMAPs, dotplots, volcano grids, enrichment tiles, state space, legends |
| `skin.atlas.*` | CellTypist, Census, label transfer, **train your own model**, orthologs |
| `skin.runtime.*` | status, build R container, version manifest, vetted R scripts |
| `skin.export.*` | notebook, report, methods draft, bundle |
| `skin.help.*` | list tools, workflow, explain a tool |

Every tool takes `dry_run` and `seed`. Every return carries `next_suggested_tools`.
Destructive tools (`apply_filters` over 30 % loss, `drop_clusters`, `regress_markers`)
require `dry_run` first and then `confirm=True`.

---

## Things this server will tell you that you may not want to hear

These are deliberate, and each one is surfaced in a tool's return rather than buried.

**Gene exclusion is not decontamination.** Dropping collagen/keratin genes before DE
stops them topping the list, but the counts remain and still inflate library sizes,
size factors and the neighbour graph. `skin.qc.estimate_ambient` (SoupX/DecontX) is
the real fix, and every gene-exclusion return says so.

**Cell-wise Wilcoxon p-values are pseudo-replicated.** With n=3–4 mice per arm the
false-positive rate is high, and the counts printed on a volcano from a cell-wise run
are several-fold higher than pseudobulk. `skin.de.compare_methods` quantifies the
shift — reviewers ask about exactly this.

**Sample-within-condition nesting is fine; batch-is-condition is not.** The
confounding guard checks whether each biological level contains ≥2 batches, not just
Cramér's V. A naive V>0.9 rule would refuse every normal experiment.

**Neutrophils are a trap.** In wound and burn skin they legitimately carry 200–600
genes. A cohort-wide `min_genes` of 500 deletes them and you will not notice;
`recommend_thresholds` fires a `neutrophil_risk` warning with the count of cells you
would lose.

**Flex is probe-based.** The probe set covers few or no mitochondrial genes, so
`pct_counts_mt` is not a viability metric. `max_pct_mt: null` means *skip the filter*,
not "use 0".

**There is no mouse skin CellTypist model.** For mouse the default is marker-based.
Cross-species transfer works but is tagged low-confidence, and
`skin.atlas.train_model` — building a model from your own annotated data — is the
path that actually compounds in value.

**`Adult_Human_Skin` is healthy skin.** It has no burn, wound, LAM or MDM states, and
applied to wound data it will confidently map everything onto homeostatic labels. Any
query whose metadata indicates injury gets a `domain_shift_warning`.

**Trajectories fit in UMAP space have UMAP's geometry, not the data's.**
`skin.traj.monocle` reports Spearman ρ between pseudotime and the real experimental
timepoint; ρ ≈ 0 across a designed timecourse is a red flag, and the tool says so.

**The 28-term GO exclusion list is opt-in.** Some of those terms are defensible in
skin; `Epithelial To Mesenchymal Transition` is plausibly real in wound fibroblasts.
It ships as the named preset `skin_irrelevant_v1`, off by default, and every dropped
term is recorded in the result and the figure's sidecar JSON.

---

## The R runtime

Some things have no adequate Python equivalent: `scDblFinder`, `SoupX`/`DecontX`,
`miloR` with spatial FDR, `CellChat`, `DESeq2` as a cross-check, `Seurat` v5 import.

```bash
skin.runtime.create(kind="r", backend="docker")   # builds from a pinned base digest
skin.runtime.status()                             # what is available right now
skin.runtime.manifest()                           # the methods-section version table
```

Only **named, vetted scripts** under `src/skinmcp/runtimes/r/scripts/` can run; the
model supplies a script id and typed parameters, never R source. Arbitrary R lives
behind `skin.runtime.exec_r_raw` and is disabled unless the server was started with
`--allow-raw-exec`.

Without a working R backend, R-backed tools return a typed `RUNTIME_UNAVAILABLE`
naming their Python fallback (`milo_r` → `milo_py`, `deseq2_r` → `pseudobulk`,
`cellchat_r` → `liana`) rather than a stack trace.

---

## Layout

```
src/skinmcp/
├── server.py        MCP app, tools, resources, prompts, transports
├── config.py        project root, cache, offline flag, profile
├── registry.py      handle registry, lineage graph, LRU cache
├── returns.py       ToolResult envelope, JSON coercion, 4 KB budget
├── errors.py        closed error taxonomy with remedies
├── memory/          SQLite store, schema, brief()/recall
├── style/           rcParams contract, palettes, panels (volcano, tile, dotplot)
├── knowledge/       markers (mouse + generated human), contamination patterns,
│                    platform presets, gene sets, enrichment libraries, MGI orthologs
├── tools/           one module per namespace
├── runtimes/        Python manifest, R Dockerfile + renv.lock + vetted scripts, bridge
├── prompts/         10 SOP markdown files
└── vendor/py_monocle/  principal-graph implementation (see NOTICE)
tests/               schemas, returns, no-stdout, pipeline, reproducibility, golden data
scripts/             build_markers_human.py (regenerates the human marker YAML)
```

The human marker sets are **generated**, not hand-written: `scripts/build_markers_human.py`
maps the mouse sets through the shipped MGI ortholog table plus a curated override
list. Naive `.upper()` is wrong often enough to matter — `Trp63`→`TP63`,
`Lyz2`→`LYZ`, `H2-Aa`→`HLA-DRA` — and `Ly6c1/2`, `Retnlg`, `Chil3` and `Ly6a` have no
human ortholog at all. Edit the mouse YAML or the override table, then re-run.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

- `test_schemas.py` — every tool: schema <2 KB, ≤6 required args, `dry_run`/`seed`
  present, documented Args block
- `test_returns.py` — envelope validity, the 4 KB budget, typed errors with remedies
- `test_no_stdout.py` — AST scan for `print(`/`sys.stdout` plus a live capture check
  (a single `print` corrupts JSON-RPC on stdio)
- `test_pipeline.py` — end-to-end on the golden dataset, asserting on the data behind
  the figures rather than pixels
- `test_reproducibility.py` — **the acceptance test**: export the notebook, execute
  it, compare the DE result

The golden dataset is synthetic (`tests/golden/make_golden.py`): 12 samples ×
2 conditions × 2 timepoints, 7 cell types, with a planted burn effect, a
low-complexity neutrophil population, keratin/collagen ambient in every cell, and a
deliberately mixed keratinocyte/fibroblast cluster — so the neutrophil warning,
the ambient classification and the mixed-cluster branch all have something real to
fire on. See `tests/golden/README.md`.

The `.h5ad` itself is **not committed** (12.5 MB, and `*.h5ad` is gitignored). It is
deterministic at `seed=0`, and `conftest.py` regenerates it on first use, so a fresh
clone and CI both work with no extra step:

```bash
python tests/golden/make_golden.py   # optional; the fixture does it for you
```

## License

MIT. See `NOTICE` for third-party attributions and the py-monocle situation.
