# SOP: QC and filtering

Threshold **discovery** first. Nothing is removed until you say so.

1. `skin.qc.sample_stats(dataset_id, sample_key="Sample")`
   Per-sample medians/MADs/quantiles, mito/ribo/haemoglobin, complexity, and the
   skin-specific keratin / collagen / cornified ambient probes. Read `flags`.
2. `skin.qc.plot_sample_stats(dataset_id)` — look at the violins before trusting numbers.
3. `skin.qc.recommend_thresholds(dataset_id, method="mad")`
   MAD is the default and the recommended answer. `method="fixed"` returns the
   platform preset; `"both"` intersects them.
   - **If `max_pct_mt` comes back null**, the chemistry is probe-based (Flex) or
     nuclear (snRNA/Multiome). That means SKIP the mito filter, not use 0.
   - **If `neutrophil_risk` fires**, the proposed `min_genes` is deleting real
     neutrophils. Wound and burn neutrophils legitimately carry 200-600 genes.
     Lower the floor to <=200, or filter per cell type after clustering.
4. `skin.qc.preview_filters(dataset_id, thresholds=...)`
   Always. Read `lost_by_lineage` — if a lineage is >2x enriched in the discarded
   fraction, you are deleting a population, not debris.
5. `skin.memory.set_param(name="qc.min_genes", value=..., rationale=...)` — paste the
   `rationale` string from step 3 and edit it.
6. `skin.qc.apply_filters(dataset_id, thresholds=..., exclude_samples=[...])`
   Needs `confirm=True` above 30% cell loss.
7. If `high_ambient_keratin` or `high_ambient_collagen` fired:
   `skin.qc.estimate_ambient(dataset_id, raw_path=..., method="decontx")`.
   That is the real fix. Excluding genes later stops them driving clusters but leaves
   the counts distorting library sizes.
8. `skin.doublet.call(dataset_id, method="scrublet", sample_key="Sample")` — per sample,
   always. Do NOT filter yet: call -> cluster -> check where the calls concentrate -> decide.
