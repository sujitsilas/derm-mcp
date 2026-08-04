"""End-to-end: load -> QC -> filter -> doublets -> preprocess -> harmony -> leiden
-> markers -> annotate -> audit -> subcluster -> pseudobulk DE -> figures -> export.

Assertions are on the DATA BEHIND the figures (DE tables, proportion tables,
audit classifications), never on pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skinmcp.tools import (
    abundance_tools,
    annotate_tools,
    cluster_tools,
    de_tools,
    export_tools,
    integrate_tools,
    io_tools,
    memory_tools,
    plot_tools,
    qc_tools,
    subcluster_tools,
    traj_tools,
)


def ok(r, what=""):
    assert r["ok"], f"{what} failed: {json.dumps(r.get('error'), indent=2)}"
    return r


class TestIngest:
    def test_load_and_describe(self, loaded):
        pid, ds = loaded
        r = ok(io_tools.describe(dataset_id=ds, project_id=pid), "describe")
        s = r["summary"]
        assert s["n_obs"] == 7200
        assert s["organism"] == "mouse"
        assert "counts" in s["layers"]
        assert s["x_state"] == "counts"

    def test_organism_mismatch_is_fatal(self, project, golden_path):
        r = io_tools.load_h5ad(path=str(golden_path), organism="human", project_id=project)
        assert not r["ok"]
        assert r["error"]["code"] == "ORGANISM_MISMATCH"
        assert "remedy" in r["error"]

    def test_handles_are_deterministic(self, project, golden_path):
        a = ok(io_tools.load_h5ad(path=str(golden_path), organism="mouse", project_id=project))
        b = ok(io_tools.load_h5ad(path=str(golden_path), organism="mouse", project_id=project))
        assert a["dataset_id"] == b["dataset_id"], "identical calls must dedupe"

    def test_invalid_handle_is_typed(self, project):
        r = io_tools.describe(dataset_id="ds_deadbeef", project_id=project)
        assert not r["ok"]
        assert r["error"]["code"] == "INVALID_HANDLE"


class TestQC:
    def test_sample_stats_and_ambient_flags(self, loaded):
        pid, ds = loaded
        r = ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid), "sample_stats")
        s = r["summary"]
        assert s["n_samples"] == 12
        # above 8 samples only flagged ones are inlined; the cohort range carries
        # the rest, and the full table is an artifact
        assert s["per_sample_shown"] == "flagged only"
        cr = s["cohort_range"]
        for k in ("n_cells", "median_genes", "median_counts", "median_pct_mt",
                  "frac_keratin", "frac_collagen"):
            assert {"min", "median", "max"} <= set(cr[k]), f"missing range for {k}"
        # the golden data bakes in keratin/collagen ambient into every cell
        assert cr["frac_keratin"]["median"] > 0
        assert cr["frac_collagen"]["median"] > 0
        assert s["genes_matched"]["mt"] >= 3
        # the quantiles live in the artifact, not the return
        import pandas as pd
        full = pd.read_csv(s["full_table"])
        assert len(full) == 12
        assert any(c.startswith("genes_quantiles") for c in full.columns)
        # and the whole return still fits the budget
        assert len(json.dumps(r).encode()) <= 4096

    def test_neutrophil_risk_fires(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        ok(qc_tools.recommend_thresholds(dataset_id=ds, method="fixed", project_id=pid))
        # the "fixed" preset uses min_genes=200; force a floor that eats neutrophils
        r2 = ok(qc_tools.preview_filters(dataset_id=ds, project_id=pid,
                                         thresholds={"min_genes": 500}))
        nr = r2["summary"]["neutrophil_risk"]
        assert nr["at_risk"] is True, "a min_genes floor of 500 must flag neutrophil loss"
        assert nr["n_neutrophil_like"] > 0
        assert any("neutrophil_risk" in w for w in r2["warnings"])

    def test_preview_reports_lost_lineages(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        r = ok(qc_tools.preview_filters(dataset_id=ds, project_id=pid,
                                        thresholds={"min_genes": 500}))
        lost = r["summary"]["lost_by_lineage"]
        assert lost, "must report which lineages the filter would remove"
        assert any(d["lineage"] == "Neutrophils" for d in lost)

    def test_apply_filters_needs_confirm_above_30pct(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        r = qc_tools.apply_filters(dataset_id=ds, project_id=pid,
                                   thresholds={"min_genes": 1200})
        assert not r["ok"]
        assert r["error"]["code"] == "CONFIRMATION_REQUIRED"

    def test_flex_chemistry_skips_mito(self, project, golden_path):
        import anndata as ad

        from skinmcp import registry

        a = ad.read_h5ad(golden_path)
        registry.skinmcp_uns(a)["chemistry"] = "10x_flex"
        p = Path(golden_path).parent / "_tmp_flex.h5ad"
        a.write_h5ad(p)
        try:
            ds = ok(io_tools.load_h5ad(path=str(p), organism="mouse",
                                       project_id=project))["dataset_id"]
            # chemistry must survive the round trip via uns
            ok(qc_tools.sample_stats(dataset_id=ds, project_id=project))
            r = ok(qc_tools.recommend_thresholds(dataset_id=ds, method="mad",
                                                 project_id=project))
            assert r["summary"]["cohort"]["max_pct_mt"] is None
            assert any("null" in w or "skip" in w.lower() for w in r["warnings"])
        finally:
            p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def filtered(request):
    """Filtered + doublet-called handle, shared across the heavier tests."""
    return None


class TestClusteringAnnotation:
    @pytest.fixture()
    def clustered(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        f = ok(qc_tools.apply_filters(dataset_id=ds, project_id=pid,
                                      thresholds={"min_genes": 150, "max_pct_mt": 25}),
               "apply_filters")["dataset_id"]
        p = ok(integrate_tools.preprocess(dataset_id=f, project_id=pid, n_hvg=400,
                                          n_comps=20), "preprocess")["dataset_id"]
        h = ok(integrate_tools.harmony(dataset_id=p, project_id=pid, batch_key="Sample",
                                       biological_key="Type"), "harmony")["dataset_id"]
        n = ok(cluster_tools.neighbors(dataset_id=h, project_id=pid, n_pcs=20),
               "neighbors")["dataset_id"]
        u = ok(cluster_tools.umap(dataset_id=n, project_id=pid), "umap")["dataset_id"]
        c = ok(cluster_tools.leiden(dataset_id=u, project_id=pid, resolution=0.6),
               "leiden")["dataset_id"]
        m = ok(cluster_tools.marker_genes(dataset_id=c, project_id=pid,
                                          groupby="leiden_res0.6"), "markers")["dataset_id"]
        return pid, m, "leiden_res0.6"

    def test_preprocess_keeps_counts_and_lognorm(self, loaded):
        pid, ds = loaded
        r = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))
        from skinmcp import registry

        a = registry.load(pid, r["dataset_id"])
        assert "counts" in a.layers, "raw counts must survive for the project's lifetime"
        assert "lognorm" in a.layers
        assert registry.get_x_state(a) == "scaled"

    def test_harmony_allows_normal_nesting(self, loaded):
        """Sample-within-condition nesting is universal and must NOT be refused."""
        pid, ds = loaded
        p = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        r = ok(integrate_tools.harmony(dataset_id=p, project_id=pid, batch_key="Sample",
                                       biological_key="Type"), "harmony")
        c = r["summary"]["confounding"]
        assert c["cramers_v"] == 1.0, "Sample is fully nested in Type by construction"
        assert c["confounded"] is False, "6 samples per arm is not confounding"
        assert c["nested_but_safe"] is True
        assert any("normal" in w for w in r["warnings"])
        from skinmcp import registry
        a = registry.load(pid, r["dataset_id"])
        assert a.obsm["X_pca_harmony"].shape[0] == a.n_obs

    def test_harmony_refuses_true_confounding(self, loaded):
        """batch_key == the biological variable: nothing to align within a level."""

        pid, ds = loaded
        p = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        r = integrate_tools.harmony(dataset_id=p, project_id=pid, batch_key="Type",
                                    biological_key="Type", force=False)
        assert not r["ok"]
        assert r["error"]["code"] == "CONFOUNDED_BATCH"
        d = r["error"]["details"]
        assert d["min_batches_per_bio_level"] == 1
        assert "contingency" in d
        # ...and proceeds, tagged, when explicitly forced
        r2 = ok(integrate_tools.harmony(dataset_id=p, project_id=pid, batch_key="Type",
                                        biological_key="Type", force=True))
        assert r2["summary"]["forced"] is True

    def test_clusters_recover_cell_types(self, clustered):
        pid, ds, ck = clustered
        from skinmcp import registry

        a = registry.load(pid, ds)
        import pandas as pd

        tab = pd.crosstab(a.obs[ck].astype(str), a.obs["true_celltype"].astype(str))
        purity = (tab.max(axis=1) / tab.sum(axis=1))
        assert purity.median() > 0.75, f"clusters are not clean: {purity.to_dict()}"
        assert a.obs[ck].nunique() >= 5

    def test_marker_report_proposes_without_writing(self, clustered):
        pid, ds, ck = clustered
        from skinmcp import registry

        before = set(registry.load(pid, ds).obs.columns)
        r = ok(annotate_tools.marker_report(dataset_id=ds, cluster_key=ck, project_id=pid))
        after = set(registry.load(pid, ds).obs.columns)
        assert before == after, "marker_report must never write obs"
        prop = r["summary"]["proposed_mapping"]
        assert len(prop) >= 5
        assert "PROPOSAL ONLY" in r["summary"]["note"]

    def test_apply_labels_rejects_unmapped_clusters(self, clustered):
        pid, ds, ck = clustered
        r = annotate_tools.apply_labels(dataset_id=ds, cluster_key=ck, project_id=pid,
                                        mapping={"0": "Macrophages"}, new_key="cell_types")
        assert not r["ok"]
        assert r["error"]["code"] == "AMBIGUOUS_LABELS"
        assert r["error"]["details"]["unmapped"]

    def test_contamination_audit_classifies_causes(self, clustered):
        pid, ds, ck = clustered
        r = ok(annotate_tools.marker_report(dataset_id=ds, cluster_key=ck, project_id=pid))
        mapping = r["summary"]["proposed_mapping"]
        lab = ok(annotate_tools.apply_labels(dataset_id=ds, cluster_key=ck, project_id=pid,
                                             mapping=mapping, new_key="cell_types"),
                 "apply_labels")["dataset_id"]
        au = ok(annotate_tools.contamination_audit(dataset_id=lab, label_key="cell_types",
                                                   project_id=pid), "audit")
        s = au["summary"]
        assert s["n_labels_audited"] >= 3
        causes = {r_["likely_cause"] for r_ in s["per_label"]}
        assert causes <= {"ambient", "doublet", "mixed_cluster", "true_biology", "clean"}
        for row in s["per_label"]:
            assert "recommended_action" in row and row["recommended_action"]
        assert "Nothing was removed" in s["note"]


class TestDE:
    @pytest.fixture()
    def labelled(self, loaded):
        pid, ds = loaded

        from skinmcp import registry

        # Use the ground-truth labels so DE is testable independent of clustering.
        a = registry.load(pid, ds, copy=True)
        a.obs["cell_types"] = a.obs["true_celltype"].astype(str).astype("category")
        new = registry.mint(pid, a, parent_id=ds, op="test.label",
                            params={"src": "true_celltype"}, label="labelled")
        return pid, new

    def test_pseudobulk_finds_the_planted_burn_effect(self, labelled):
        import pandas as pd

        pid, ds = labelled
        r = ok(de_tools.pseudobulk(dataset_id=ds, label_key="cell_types",
                                   condition_key="Type", contrast=["Burn", "Sham"],
                                   covariates=["Timepoint"], groups=["Macrophages"],
                                   exclude_gene_groups=[], project_id=pid), "pseudobulk")
        s = r["summary"]
        assert s["inference_level"] == "sample"
        assert s["design"] == "~ Timepoint + Type"
        assert len(s["per_label"]) == 1
        row = s["per_label"][0]
        assert row["n_samples_Burn"] == 6 and row["n_samples_Sham"] == 6
        df = pd.read_csv(s["tables"][row["label"]]).set_index("gene")
        for g in ("Arg1", "Nos2", "Spp1"):
            assert g in df.index, f"{g} missing from the DE table"
            assert df.loc[g, "lfc"] > 0.5, f"{g} should be up in Burn, got {df.loc[g, 'lfc']}"
            assert df.loc[g, "padj"] < 0.05, f"{g} not significant: {df.loc[g, 'padj']}"

    def test_skips_underpowered_labels_instead_of_falling_back(self, labelled):
        pid, ds = labelled
        r = ok(de_tools.pseudobulk(dataset_id=ds, label_key="cell_types",
                                   condition_key="Type", contrast=["Burn", "Sham"],
                                   min_samples_per_arm=99, groups=None,
                                   project_id=pid)) if False else de_tools.pseudobulk(
            dataset_id=ds, label_key="cell_types", condition_key="Type",
            contrast=["Burn", "Sham"], min_samples_per_arm=99, project_id=pid)
        assert not r["ok"]
        assert r["error"]["code"] == "INSUFFICIENT_REPLICATES"
        assert "skipped" in r["error"]["details"]

    def test_wilcoxon_is_labelled_exploratory(self, labelled):
        pid, ds = labelled
        r = ok(de_tools.wilcoxon(dataset_id=ds, label_key="cell_types",
                                 condition_key="Type", contrast=["Burn", "Sham"],
                                 groups=["Macrophages"], exclude_gene_groups=[],
                                 project_id=pid), "wilcoxon")
        assert r["summary"]["inference_level"] == "cell"
        assert "pseudo-replicated" in r["summary"]["caveat"]
        assert any("EXPLORATORY" in w for w in r["warnings"])

    def test_compare_methods(self, labelled):
        pid, ds = labelled
        a = ok(de_tools.pseudobulk(dataset_id=ds, label_key="cell_types",
                                   condition_key="Type", contrast=["Burn", "Sham"],
                                   groups=["Macrophages"], exclude_gene_groups=[],
                                   project_id=pid))["summary"]["run_id"]
        b = ok(de_tools.wilcoxon(dataset_id=ds, label_key="cell_types",
                                 condition_key="Type", contrast=["Burn", "Sham"],
                                 groups=["Macrophages"], exclude_gene_groups=[],
                                 project_id=pid))["summary"]["run_id"]
        r = ok(de_tools.compare_methods(run_a=a, run_b=b, project_id=pid), "compare")
        row = r["summary"]["per_label"][0]
        assert row["spearman_lfc"] > 0.3, "LFC rankings should broadly agree"
        assert row["n_sig_b"] > row["n_sig_a"], "cell-wise should call far more genes"

    def test_volcano_grid_writes_pdf_and_png(self, labelled, tmp_path):
        pid, ds = labelled
        run = ok(de_tools.pseudobulk(dataset_id=ds, label_key="cell_types",
                                     condition_key="Type", contrast=["Burn", "Sham"],
                                     groups=["Macrophages", "Fibroblasts"],
                                     exclude_gene_groups=[], project_id=pid))
        rid = run["summary"]["run_id"]
        r = ok(plot_tools.volcano_grid(de_run_id=rid, ncols=2, must_label=["Arg1", "Nos2"],
                                       highlight_genes=["Arg1"], project_id=pid), "volcano")
        s = r["summary"]
        assert Path(s["paths"]["pdf"]).exists() and Path(s["paths"]["png"]).exists()
        assert Path(s["paths"]["pdf"]).with_suffix(".json").exists(), "sidecar missing"
        assert s["counts"]["Macrophages"]["Burn_up"] >= 3
        assert s["inference_level"] == "sample"


class TestAbundanceAndTrajectory:
    @pytest.fixture()
    def labelled(self, loaded):
        from skinmcp import registry

        pid, ds = loaded
        a = registry.load(pid, ds, copy=True)
        a.obs["cell_types"] = a.obs["true_celltype"].astype(str).astype("category")
        return pid, registry.mint(pid, a, parent_id=ds, op="test.label",
                                  params={"src": "true_celltype"})

    def test_proportions(self, labelled):
        pid, ds = labelled
        r = ok(abundance_tools.proportions(dataset_id=ds, label_key="cell_types",
                                           group_keys=["Type", "Timepoint"],
                                           project_id=pid), "proportions")
        s = r["summary"]
        assert s["n_samples"] == 12
        assert s["n_labels"] == 7
        assert Path(s["table_path"]).exists()
        assert any("compositional" in w for w in r["warnings"])

    def test_milo_py(self, labelled):
        pid, ds = labelled
        p = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        h = ok(integrate_tools.harmony(dataset_id=p, project_id=pid, batch_key="Batch",
                                       biological_key="Type"))["dataset_id"]
        n = ok(cluster_tools.neighbors(dataset_id=h, project_id=pid, n_pcs=20))["dataset_id"]
        u = ok(cluster_tools.umap(dataset_id=n, project_id=pid))["dataset_id"]
        r = ok(abundance_tools.milo_py(dataset_id=u, label_key="cell_types",
                                       condition_key="Type", contrast=["Burn", "Sham"],
                                       prop=0.05, project_id=pid), "milo")
        s = r["summary"]
        assert s["n_neighbourhoods"] > 20
        assert s["dispersion_phi"] >= 1.0
        assert s["n_samples"]["Burn"] == 6
        assert Path(s["table_path"]).exists()

    def test_trajectory_reports_time_correlation(self, labelled):
        pid, ds = labelled
        p = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        n = ok(cluster_tools.neighbors(dataset_id=p, project_id=pid, use_rep="X_pca",
                                       n_pcs=20))["dataset_id"]
        u = ok(cluster_tools.umap(dataset_id=n, project_id=pid))["dataset_id"]
        r = ok(traj_tools.monocle(dataset_id=u, cluster_key="cell_types",
                                  root_label="Macrophages", n_centroids=12,
                                  p_threshold=0.5, project_id=pid), "monocle")
        s = r["summary"]
        assert "implementation" in s
        res = s["per_split"]["all"]
        assert "rho_pseudotime_vs_timepoint" in res
        assert "root_purity" in res and "is_leaf" in res
        assert any("UMAP space" in w for w in r["warnings"])


class TestSubclusterAndMemory:
    def test_extract_restores_counts(self, loaded):
        from skinmcp import registry

        pid, ds = loaded
        a = registry.load(pid, ds, copy=True)
        a.obs["cell_types"] = a.obs["true_celltype"].astype(str).astype("category")
        lab = registry.mint(pid, a, parent_id=ds, op="test.label", params={})
        # give it a stale embedding, which extract must discard
        p = ok(integrate_tools.preprocess(dataset_id=lab, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        r = ok(subcluster_tools.extract(dataset_id=p, label_key="cell_types",
                                        labels=["Macrophages"], project_id=pid), "extract")
        sub = registry.load(pid, r["dataset_id"])
        assert registry.get_x_state(sub) == "counts"
        assert "X_pca" not in sub.obsm, "parent embedding must be dropped"
        assert "highly_variable" not in sub.var.columns
        assert any("raw counts" in w for w in r["warnings"])

    def test_drop_clusters_requires_reason_and_confirm(self, loaded):
        from skinmcp import registry

        pid, ds = loaded
        a = registry.load(pid, ds, copy=True)
        a.obs["cell_types"] = a.obs["true_celltype"].astype(str).astype("category")
        lab = registry.mint(pid, a, parent_id=ds, op="test.label", params={})
        r = subcluster_tools.drop_clusters(dataset_id=lab, cluster_key="cell_types",
                                           clusters=["Neutrophils"], reason="",
                                           project_id=pid)
        assert not r["ok"] and r["error"]["code"] == "BAD_PARAM"
        r = subcluster_tools.drop_clusters(dataset_id=lab, cluster_key="cell_types",
                                           clusters=["Neutrophils"], reason="test",
                                           project_id=pid)
        assert not r["ok"] and r["error"]["code"] == "CONFIRMATION_REQUIRED"
        r = ok(subcluster_tools.drop_clusters(dataset_id=lab, cluster_key="cell_types",
                                              clusters=["Neutrophils"],
                                              reason="low complexity, confirmed debris",
                                              confirm=True, project_id=pid))
        assert r["summary"]["decision_id"]
        from skinmcp.memory import store

        decs = store.get_decisions(pid)
        assert any("Neutrophils" in d["question"] for d in decs)

    def test_memory_records_and_searches(self, loaded):
        pid, ds = loaded
        ok(memory_tools.record_annotation(
            dataset_id=ds, obs_key="cell_types", mapping={"0": "Macrophages"},
            rationale="C1qa/C1qb/Adgre1 high, Ptprc positive, Krt5 negative",
            confidence=0.9, author="user:test", project_id=pid))
        r = ok(memory_tools.search(query="Adgre1", project_id=pid))
        assert r["summary"]["n_hits"] >= 1
        b = ok(memory_tools.brief(project_id=pid))
        assert b["summary"]["annotation_sets"]
        assert "parameter_note" in b["summary"]

    def test_annotation_without_rationale_is_rejected(self, loaded):
        pid, ds = loaded
        r = memory_tools.record_annotation(dataset_id=ds, obs_key="x",
                                           mapping={"0": "Mac"}, rationale="  ",
                                           project_id=pid)
        assert not r["ok"] and r["error"]["code"] == "BAD_PARAM"

    def test_lineage_tree(self, loaded):
        pid, ds = loaded
        p = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300,
                                          n_comps=20))["dataset_id"]
        r = ok(io_tools.lineage(project_id=pid))
        assert ds in r["summary"]["tree"] and p in r["summary"]["tree"]


class TestExport:
    def test_notebook_and_report(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300, n_comps=20))
        r = ok(export_tools.notebook(fmt="both", project_id=pid), "notebook")
        assert len(r["summary"]["files"]) == 2
        nb = Path(r["summary"]["files"][0])
        assert nb.exists()
        import nbformat

        n = nbformat.read(str(nb), as_version=4)
        assert len(n.cells) >= 4
        src = "\n".join(c["source"] for c in n.cells)
        assert "HANDLES = {" in src, "params cell must resolve handles to paths"
        assert "sc.pp.normalize_total" in src

        rep = ok(export_tools.report(fmt="md", project_id=pid), "report")
        assert Path(rep["summary"]["path"]).exists()

    def test_methods_draft_is_labelled_draft(self, loaded):
        pid, ds = loaded
        ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid))
        r = ok(export_tools.methods_paragraph(project_id=pid))
        assert any("DRAFT" in w for w in r["warnings"])
        assert "scanpy" in r["summary"]["text"]

    def test_bundle(self, loaded):
        pid, ds = loaded
        ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=300, n_comps=20))
        ok(export_tools.notebook(fmt="ipynb", project_id=pid))
        r = ok(export_tools.bundle(project_id=pid), "bundle")
        z = Path(r["summary"]["path"])
        assert z.exists()
        import zipfile

        with zipfile.ZipFile(z) as f:
            names = f.namelist()
        assert "manifest.json" in names
        assert any(n.startswith("notebooks/") for n in names)
        assert "memory.db" in names
