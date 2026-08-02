# SOP: start or resume a project

1. `skin.memory.open_project(name=..., organism="mouse"|"human", description=..., design_notes=...)`
   Re-attaches by name if it already exists. The return includes the full briefing.
2. If resuming, read the `briefing.workflow.next_suggested_tools` field — it is computed
   from what has actually run, not from a script.
3. Load data. The normal entry point is
   `skin.io.build_multisample(inputs=[{path, sample, condition, timepoint, ...}, ...])`.
   For one file use `skin.io.load_10x` or `skin.io.load_h5ad`.
4. `skin.io.describe(dataset_id)` — confirm n_obs/n_vars, obs columns, and that
   `layers` contains `counts`. Without raw counts, pseudobulk DE and subclustering
   re-normalization are unavailable.
5. `skin.meta.order_categorical(key="Timepoint")` so D7 < D10 < D14, not alphabetical.
6. `skin.meta.assign_palette(key="Type", scheme="condition")` and
   `assign_palette(key="Timepoint", scheme="timepoint")` — recorded in memory so
   every figure in the project matches.
7. `skin.memory.note(tag="setup", body=...)` with anything about the experiment a
   reader would need and the metadata does not carry.

Organism is fixed for the life of the project and a gene-casing mismatch on load is a
hard error, not a warning.
