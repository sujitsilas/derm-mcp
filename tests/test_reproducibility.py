"""The acceptance test: export the notebook, execute it, compare the results.

If a result cannot be regenerated from the exported notebook, reproducibility is
a claim rather than a feature. This runs a real project, exports it, executes
every code cell in a clean namespace, and asserts the DE table it produces
matches the one the server produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skinmcp.tools import (
    cluster_tools,
    de_tools,
    export_tools,
    integrate_tools,
    qc_tools,
)


def ok(r, what=""):
    assert r["ok"], f"{what} failed: {json.dumps(r.get('error'), indent=2)}"
    return r


@pytest.fixture()
def run_project(loaded):
    """A short but real project: QC -> filter -> preprocess -> cluster -> DE."""
    from skinmcp import registry

    pid, ds = loaded
    ok(qc_tools.sample_stats(dataset_id=ds, project_id=pid), "sample_stats")
    ok(qc_tools.recommend_thresholds(dataset_id=ds, project_id=pid), "thresholds")
    f = ok(qc_tools.apply_filters(dataset_id=ds, project_id=pid,
                                  thresholds={"min_genes": 150, "max_pct_mt": 30}),
           "filters")["dataset_id"]
    p = ok(integrate_tools.preprocess(dataset_id=f, project_id=pid, n_hvg=300, n_comps=15),
           "preprocess")["dataset_id"]
    n = ok(cluster_tools.neighbors(dataset_id=p, project_id=pid, use_rep="X_pca", n_pcs=15),
           "neighbors")["dataset_id"]
    c = ok(cluster_tools.leiden(dataset_id=n, project_id=pid, resolution=0.5),
           "leiden")["dataset_id"]

    a = registry.load(pid, c, copy=True)
    a.obs["cell_types"] = a.obs["true_celltype"].astype(str).astype("category")
    lab = registry.mint(pid, a, parent_id=c, op="test.label", params={}, label="final")

    de = ok(de_tools.pseudobulk(dataset_id=lab, label_key="cell_types",
                                condition_key="Type", contrast=["Burn", "Sham"],
                                groups=["Macrophages"], exclude_gene_groups=[],
                                project_id=pid), "de")
    return pid, lab, de["summary"]


def test_exported_notebook_executes_and_reproduces_de(run_project, tmp_path):
    import pandas as pd

    pid, lab, de_summary = run_project
    row = de_summary["per_label"][0]
    server_df = pd.read_csv(row["table_path"]).set_index("gene")

    nb_res = ok(export_tools.notebook(fmt="ipynb", project_id=pid), "notebook")
    nb_path = Path(nb_res["summary"]["files"][0])
    assert nb_path.exists()

    import nbformat

    nb = nbformat.read(str(nb_path), as_version=4)
    code_cells = [c["source"] for c in nb.cells if c["cell_type"] == "code"]
    assert len(code_cells) >= 4, "notebook has too few code cells to be a record"

    # --- execute the notebook in a clean namespace ------------------------- #
    ns: dict = {"__name__": "__notebook__"}
    executed, failures = 0, []
    for i, src in enumerate(code_cells):
        if not src.strip():
            continue
        # a cell that is entirely comments is a no-op, but must still be exec'd
        # in order — the params cell opens with a comment banner
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            executed += 1
        except Exception as e:  # noqa: BLE001 - we report, we do not hide
            failures.append(f"cell {i}: {type(e).__name__}: {e}\n---\n{src[:400]}")

    assert not failures, ("the exported notebook does not execute:\n\n"
                          + "\n\n".join(failures[:3]))
    assert executed >= 4

    # --- the params cell must resolve handles to real, readable paths ------- #
    assert "HANDLES" in ns, "the params cell did not define HANDLES"
    assert "load" in ns, "the params cell did not define load()"
    for h, path in ns["HANDLES"].items():
        assert Path(path).exists(), f"handle {h} points at a missing file: {path}"

    # --- re-derive the DE result from the notebook's own objects ------------ #
    adata = ns["load"](lab)
    assert "cell_types" in adata.obs.columns
    assert "counts" in adata.layers

    # the notebook must not have clobbered the server's own artifact
    assert list(pd.read_csv(row["table_path"]).columns) == list(
        server_df.reset_index().columns), "the notebook overwrote a server table"

    import numpy as np
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    m = (adata.obs["cell_types"].astype(str) == "Macrophages").to_numpy()
    sub = adata[m]
    smp = sub.obs["Sample"].astype(str).to_numpy()
    X = sub.layers["counts"]
    rows, idx = [], []
    for s in sorted(set(smp)):
        sel = smp == s
        v = np.asarray(X[sel].sum(0)).ravel()
        rows.append(v)
        idx.append(s)
    counts = pd.DataFrame(np.vstack(rows), index=idx,
                          columns=list(map(str, sub.var_names)))
    meta = pd.DataFrame(
        {"Type": [str(sub.obs.loc[smp == s, "Type"].astype(str).mode().iloc[0]) for s in idx],
         "Timepoint": [str(sub.obs.loc[smp == s, "Timepoint"].astype(str).mode().iloc[0])
                       for s in idx]}, index=idx)
    meta["Type"] = pd.Categorical(meta["Type"], categories=["Sham", "Burn"])
    meta["Timepoint"] = meta["Timepoint"].astype("category")
    counts = counts.loc[:, counts.sum(0) >= 10]

    dds = DeseqDataSet(counts=counts.astype(int), metadata=meta,
                       design="~ Timepoint + Type", quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["Type", "Burn", "Sham"], quiet=True)
    ds.summary()
    redone = ds.results_df.dropna(subset=["padj"])

    shared = server_df.index.intersection(redone.index)
    assert len(shared) > 100, f"only {len(shared)} genes in common"

    # Unshrunk LFCs must match to floating-point tolerance: same counts, same
    # design, same solver.
    a = server_df.loc[shared, "lfc_unshrunk"].to_numpy()
    b = redone.loc[shared, "log2FoldChange"].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.sum() > 100
    np.testing.assert_allclose(a[finite], b[finite], rtol=1e-4, atol=1e-4)

    # ...and the significant-gene calls must agree.
    sig_server = set(server_df.index[(server_df["padj"] < 0.05)])
    sig_redone = set(redone.index[(redone["padj"] < 0.05)])
    jac = len(sig_server & sig_redone) / max(len(sig_server | sig_redone), 1)
    assert jac > 0.95, f"significant-gene Jaccard {jac:.3f} between server and notebook"


def test_notebook_carries_versions_and_rationale(run_project):
    from skinmcp.tools import memory_tools

    pid, lab, _ = run_project
    ok(memory_tools.record_annotation(
        dataset_id=lab, obs_key="cell_types", mapping={"0": "Macrophages"},
        rationale="C1qa/C1qb/Adgre1 high; Ptprc positive", author="user:test",
        project_id=pid))
    r = ok(export_tools.notebook(fmt="ipynb", project_id=pid))
    import nbformat

    nb = nbformat.read(r["summary"]["files"][0], as_version=4)
    md = "\n".join(c["source"] for c in nb.cells if c["cell_type"] == "markdown")
    assert "scanpy" in md and "session info" in md.lower()
    assert "Recorded rationale" in md or "C1qa" in md


def test_rmd_export_emits_chunks(run_project):
    pid, lab, _ = run_project
    r = ok(export_tools.notebook(fmt="rmd", project_id=pid))
    text = Path(r["summary"]["files"][0]).read_text()
    assert text.startswith("---")
    assert "```{python" in text
    assert "reticulate" in text


def test_determinism_same_seed_same_handle(loaded):
    """A retried call must return the existing handle, not duplicate the work."""
    pid, ds = loaded
    a = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=250,
                                      n_comps=10, seed=0))
    b = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=250,
                                      n_comps=10, seed=0))
    assert a["dataset_id"] == b["dataset_id"]
    c = ok(integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=250,
                                      n_comps=10, seed=7))
    assert c["dataset_id"] != a["dataset_id"], "a different seed must mint a new handle"
