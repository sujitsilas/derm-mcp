# SOP: first-pass major cell types

1. `skin.integrate.preprocess(dataset_id, n_hvg=2000, hvg_flavor="seurat")`
   Consider `exclude_gene_groups=["mito","ribo","hb"]` on whole-skin data.
2. `skin.integrate.harmony(dataset_id, batch_key="Sample", biological_key="Type")`
   Refuses if the batch key is confounded with the biology (Cramér's V > 0.9). If it
   refuses, do not force it — integrate on a key that varies *within* each condition,
   or model the batch in the DE design instead.
3. `skin.integrate.assess(...)` — batch mixing should rise, label LISI should not.
   Both rising means biology was removed.
4. `skin.cluster.neighbors(use_rep="X_pca_harmony")` -> `skin.cluster.umap` ->
   `skin.cluster.leiden(resolution=0.8)`.
   Unsure about resolution? `skin.cluster.leiden_sweep` returns stability and a
   recommendation.
5. `skin.cluster.marker_genes(groupby="leiden_res0.8", method="wilcoxon")`
   Only the top 10 per cluster come back; the full table is a resource.
6. `skin.cluster.cluster_qc(cluster_key=...)` — a cluster from one sample is a batch
   artifact until proven otherwise.
7. `skin.annotate.score_lineages(cluster_key=...)` then
   `skin.annotate.marker_report(cluster_key=...)`.
   marker_report **proposes**. It never writes obs.
8. Decide. Then:
   - `skin.annotate.apply_labels(cluster_key=..., mapping={...}, new_key="cell_types")`
     Every cluster must be mapped; unmapped clusters are an error, not NaN.
   - `skin.memory.record_annotation(dataset_id, obs_key="cell_types", mapping=...,
      rationale="...", confidence=..., author="model:...")`
     The rationale is the part that survives the session. Name the markers you used.
9. `skin.plot.dotplot(groupby="cell_types")` and `skin.plot.umap(color=["cell_types"])`.
