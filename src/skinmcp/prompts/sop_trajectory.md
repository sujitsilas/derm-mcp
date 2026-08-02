# SOP: trajectory

1. `skin.traj.monocle(dataset_id, cluster_key="macrophage_subtypes",
   root_label="Inf. Mono.", basis="X_umap", split_by="Type")`
   - `split_by` fits an independent graph per condition and plots them side by side.
     That comparison is usually the point.
   - Root selection is auditable: the return carries `root_purity`, `is_leaf`, and
     whether the safety fallback was used.
2. **Read `rho_pseudotime_vs_timepoint`.** Across a designed timecourse, rho near 0
   means the trajectory is not recovering the experimental order. That is a red flag,
   not a detail.
3. Re-run with `basis="X_pca_harmony"` and compare rho. `X_umap` reproduces the lab's
   existing figures, but the geometry is UMAP's, not the data's — distances along that
   graph are not distances in expression space.
4. Because you have real timepoints, `skin.traj.cellrank(time_key="Timepoint")` is often
   better grounded than any geometry-based method. Run it as a cross-check.
5. `skin.traj.pseudotime_genes(pseudotime_key=..., top_n=60)` for genes varying along
   the ordering, with the binned heatmap.
6. `skin.memory.record_decision(question="which trajectory method to report", ...)`.

Note: `skin.traj.monocle` uses the upstream `py_monocle` package when installed and
otherwise a shipped implementation of the same published algorithm. The return field
`implementation` says which one ran.
