# SOP: differential abundance

Composition questions need biological replicates. Report n_samples.

1. `skin.abundance.proportions(dataset_id, label_key=..., sample_key="Sample",
   group_keys=["Type","Timepoint"])`
   Per-sample table, stacked bars, timepoint lines with SEM and significance stars.
   Descriptive first — always look before testing.
2. `skin.abundance.milo_py(dataset_id, label_key=..., condition_key="Type",
   contrast=["Burn","Sham"], covariates=["Timepoint"])`
   Tests kNN neighbourhoods rather than discrete labels, so it sees shifts *within* a
   cell type. Design falls back from `~ C(Timepoint) + Type` to `~ Type` with a warning
   when the two are collinear. Produces the beeswarm and the neighbourhood-graph UMAP.
3. Proportions are compositional: one population expanding mechanically shrinks the
   others. Before writing "X decreased", confirm with
   `skin.abundance.sccoda(...)`, which models the sum-to-one constraint.
4. DA methods disagree. If the answer matters, run two and report both:
   `skin.abundance.milo_r` (real spatial FDR) is the R cross-check.
5. `skin.memory.record_decision(...)` naming which method you are reporting and why.
