# SOP: contamination audit and the iterative refinement loop

The question this answers: are keratinocytes expressing Col1a1, and if so, why?

1. `skin.annotate.contamination_audit(dataset_id, label_key="cell_types")`
   Per label: cross-lineage co-detection (baseline-corrected), foreign-program scores,
   ambient correlation, doublet fraction, sample skew. Read `likely_cause`.

2. Act on the cause, because the remedies differ:

   - **ambient** — uniform, low magnitude, present in every label.
     -> `skin.qc.estimate_ambient(method="decontx", apply_correction=True)` and re-run
        preprocessing from the corrected counts. Excluding genes is a presentation
        guard, not a fix.
   - **doublet** — bimodal within the label, high magnitude, elevated doublet score.
     -> `skin.doublet.cluster_enrichment`, then `skin.sub.drop_clusters(reason=...)`.
   - **mixed_cluster** — one label carrying two complete programs, low doublet score.
     -> `skin.sub.pipeline(labels=[...], resolution=<higher>)` and re-label.
   - **true_biology** — a coherent minority with a genuine dual program
     (Arg1+Nos2+ macrophages are the canonical example).
     -> Do NOT remove. `skin.memory.note(tag="biology", body=...)`.

3. `skin.annotate.refine_loop(dataset_id, label_key=..., auto_apply=False)`
   Returns an ordered PLAN of tool calls with resolved arguments. Review it, then either
   run the steps yourself or re-call with `auto_apply=True`. Plan mode is the default
   because this is the only tool that calls other tools.

4. Whatever you decide: `skin.memory.record_decision(question=..., choice=...,
   alternatives=[...], rationale=...)`.

5. If you exclude genes from the feature space
   (`skin.annotate.regress_markers(gene_groups=["collagen","keratin"], mode="exclude")`),
   note in the record that the counts are still there. `mode="regress"` distorts the
   variance structure of every gene and is rarely the right answer.
