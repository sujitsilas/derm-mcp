# SOP: differential expression

Pseudobulk is the default. n is the number of SAMPLES, not cells.

1. `skin.de.pseudobulk(dataset_id, label_key="cell_types", condition_key="Type",
   contrast=["Burn","Sham"], covariates=["Timepoint"], min_samples_per_arm=3)`
   - Raw counts are summed per sample x label (sum, never mean).
   - Design is `~ Timepoint + Type` by default: timepoints pooled as a blocking factor.
   - Covariate levels lacking both arms are dropped and reported in `dropped_levels`.
   - Labels with too few replicates land in `skipped` **with their counts**. They are
     not silently downgraded to a cell-wise test.
   - `exclude_gene_groups` defaults to `["immune_de"]` (collagen, keratin, muscle,
     ecm_misc, stress), removed before size-factor estimation. Pass `[]` to keep them.
2. Read `skipped`. If a label you care about is there, the design does not support
   population-level inference for it. Options, in order of preference: merge labels;
   report it as underpowered; or run `skin.de.wilcoxon` and label it exploratory.
3. `skin.plot.volcano_grid(de_run_id, ncols=2, must_label=["Arg1","Nos2"],
   highlight_genes=["Arg1","Nos2"])`
4. `skin.enrich.list_libraries(question_type="broad_biology", organism="mouse")` if you
   are unsure which library to use.
5. `skin.enrich.ora(de_run_id, label=..., library="GO_Biological_Process_2025")`
   `exclude_preset` is OFF by default and should stay that way unless you can defend
   each dropped term. Whatever you exclude is recorded in the figure metadata.
   With adequate power prefer `skin.enrich.gsea` — ORA throws away the ranking.
6. `skin.plot.enrichment_tile(enrich_run_ids=[...])`, or
   `skin.plot.de_panel(de_run_id)` for the volcano + one tile per label in one call.
7. Switching from a previous cell-wise analysis? `skin.de.compare_methods(run_a, run_b)`
   quantifies the shift. The significant-gene counts will drop a lot; the LFC rankings
   usually agree. Reviewers ask about exactly this.
