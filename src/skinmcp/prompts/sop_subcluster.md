# SOP: subclustering a compartment

The most common error is subsetting and reusing the parent's HVGs and embedding. Those
features were chosen to separate keratinocytes from fibroblasts; they cannot resolve
LAM-I from LAM-II.

1. One call: `skin.sub.pipeline(dataset_id, label_key="cell_types",
   labels=["Macrophages","Monocytes"], resolution=1.2, batch_key="Sample",
   exclude_gene_groups=["collagen","keratin","muscle"])`
   That is extract -> preprocess -> harmony -> neighbors -> umap -> leiden ->
   marker_genes, each logged separately.
   Step by step instead: `skin.sub.extract` (restores raw counts, drops the parent's
   feature selection) then the individual tools.
2. `skin.annotate.marker_report(cluster_key="leiden_res1.2", family="macrophage")`
   The shipped subtype priors are in `knowledge/genesets.yaml`.
3. `skin.annotate.apply_labels(..., new_key="macrophage_subtypes",
   order=[...], scheme="subtype")`
   The seeded macrophage palette matches labels tolerantly across Φ/φ/M and
   punctuation. Unknown labels get grey, never a silent reassignment.
4. `skin.memory.record_annotation(...)` with the markers you used.
5. Contaminating sub-clusters:
   `skin.sub.drop_clusters(clusters=[...], reason="...")` — reason is required and goes
   to the decision log — then `skin.sub.recluster(...)`. Never reuse the old embedding
   after dropping cells.
6. Push labels back to the parent:
   `skin.sub.map_back(sub_dataset_id, parent_dataset_id, obs_key="macrophage_subtypes")`
   Refuses below a 95% barcode match rate.
