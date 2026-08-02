# SOP: cell-cell communication

1. `skin.ccc.liana(dataset_id, label_key=..., groupby_context="Type_Timepoint",
   method="rank_aggregate")`
   Run **per context group** so the conditions are comparable — scoring the pooled
   object and splitting afterwards is not the same thing, because expression
   proportions are computed across all cells.
   Populations under 30 cells are dropped and listed. LR scores from fewer than that
   are sampling noise.
2. `skin.ccc.liana_differential(run_id, context_a="Burn D7", context_b="Sham D7")`
   Joins on the LR pair, takes the delta, and keeps only pairs specific in the arm they
   are up in. Produces the horizontal delta bars.
3. `skin.ccc.plot_lr_dotplot(run_id, sources=[...], targets=[...])`
4. For the comparative figures (information-flow scatter, pathway x timepoint heatmap):
   `skin.ccc.cellchat_r(label_key=..., split_by="Type_Timepoint")` then
   `skin.ccc.cellchat_compare(run_ids=[...])`. Needs the R runtime —
   `skin.runtime.create(kind="r")` first.
5. LR inference is a hypothesis generator. Expression of a ligand and a receptor in two
   populations is not evidence they interact. Record that framing in the note you leave.
