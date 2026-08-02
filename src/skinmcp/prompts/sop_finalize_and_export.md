# SOP: finalize and export

1. `skin.memory.brief()` — check `open_flags` for failed steps and unresolved warnings.
2. Confirm every label carries a rationale:
   `skin.memory.get_annotations(include_superseded=True)`.
   Any label without one is unauditable. Fix it now, while you remember.
3. `skin.io.set_label(dataset_id, label="final")` on the object the figures came from,
   and `skin.io.lineage()` to check the tree is what you think it is.
4. `skin.runtime.manifest()` — the version table for the methods section. It is a
   deliverable, not a debug tool.
5. `skin.export.notebook(fmt="both")` — the executable .ipynb and .Rmd. Every step's
   code, with handles resolved to real paths. Run it in a clean environment: if it does
   not reproduce the result, that is a bug, not a caveat.
6. `skin.export.report(fmt="md")` — the PI-facing narrative: annotations with
   rationales, decisions with the alternatives you rejected, figures inline, failed
   steps listed.
7. `skin.export.methods_paragraph()` — DRAFT. It reports what the log says happened.
   It does not know what you intended. Read every sentence.
8. `skin.export.bundle(include_objects=True)` — zip of notebook, figures, tables,
   uv.lock, renv.lock, manifest.json and memory.db.
9. `skin.memory.export(fmt="md")` — the lab-notebook markdown, for the shared drive.
