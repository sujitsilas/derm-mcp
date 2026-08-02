"""Build the synthetic golden dataset used by the test suite.

WHY SYNTHETIC: a real public mouse wound dataset is the right long-term choice
(see the accession note in tests/golden/README.md), but it is hundreds of MB and
cannot be committed. This generator produces a small object with the structure
the tests actually need to exercise:

  * 2 conditions x 2 timepoints x 3 replicates = 12 samples (so pseudobulk DE has
    n=6 per arm and is genuinely testable)
  * 6 skin cell types with real marker genes at real-ish expression levels
  * a deliberate burn effect in macrophages (Arg1, Nos2, Spp1, Trem2 up)
  * a low-complexity neutrophil population (~250 genes/cell) so the
    neutrophil_risk warning has something to fire on
  * ambient keratin/collagen bleed into every cell, so contamination_audit has a
    real ambient signal to classify
  * a deliberately mixed cluster (keratinocyte + fibroblast programs) so the
    mixed_cluster branch is exercised

Regenerate with:  python tests/golden/make_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

CELL_TYPES = {
    "Keratinocytes": ["Krt5", "Krt14", "Krt1", "Krt10", "Col17a1", "Dsp", "Perp", "Lgals7"],
    "Fibroblasts": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum", "Pdgfra", "Sparc", "Mfap5"],
    "Macrophages": ["Lyz2", "C1qa", "C1qb", "C1qc", "Adgre1", "Csf1r", "Mrc1", "Apoe",
                    "Trem2", "Arg1", "Nos2", "Spp1", "Mmp12", "Ctss", "Mertk"],
    "Neutrophils": ["S100a8", "S100a9", "Retnlg", "Mmp9", "Csf3r", "Cxcr2", "Il1b", "Srgn"],
    "T cells": ["Cd3e", "Cd3d", "Cd3g", "Cd8a", "Trac", "Lck", "Nkg7", "Ccl5", "Thy1"],
    "Endothelial": ["Pecam1", "Cldn5", "Cdh5", "Egfl7", "Flt1", "Emcn", "Kdr", "Tie1"],
}
# The mixed cluster: a real under-clustered population carrying two programs.
MIXED = CELL_TYPES["Keratinocytes"][:5] + CELL_TYPES["Fibroblasts"][:5]

AMBIENT = ["Krt5", "Krt14", "Col1a1", "Col1a2", "Krt10"]
BURN_UP = {"Macrophages": ["Arg1", "Nos2", "Spp1", "Trem2", "Mmp12"],
           "Fibroblasts": ["Postn", "Tnc", "Acta2"],
           "Neutrophils": ["Il1b", "Cxcl2"]}
EXTRA = ["Postn", "Tnc", "Acta2", "Cxcl2", "Mki67", "Top2a", "Actb", "Gapdh",
         "mt-Co1", "mt-Nd1", "mt-Cytb", "Rps6", "Rpl13", "Rps19", "Hbb-bs", "Hba-a1",
         "Ptprc", "Epcam", "Vim", "Fn1", "Sbsn", "Lor", "Flg"]

PROPORTIONS = {"Keratinocytes": 0.26, "Fibroblasts": 0.24, "Macrophages": 0.20,
               "T cells": 0.11, "Endothelial": 0.09, "Neutrophils": 0.07,
               "Mixed_KC_Fibro": 0.03}


def build(n_cells: int = 7200, n_bg_genes: int = 550, seed: int = 0):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)

    marker_genes = sorted({g for v in CELL_TYPES.values() for g in v} | set(EXTRA))
    bg = [f"Gene{i:04d}" for i in range(n_bg_genes)]
    genes = marker_genes + bg
    gidx = {g: i for i, g in enumerate(genes)}

    samples = [f"{t}_{d}_{r}" for t in ("Burn", "Sham") for d in ("D7", "D14")
               for r in (1, 2, 3)]
    per_sample = n_cells // len(samples)

    types = list(PROPORTIONS)
    probs = np.array([PROPORTIONS[t] for t in types], dtype=float)
    probs /= probs.sum()

    rows, obs_rows = [], []
    for s in samples:
        cond, tp, rep = s.split("_")
        for _ in range(per_sample):
            ct = types[rng.choice(len(types), p=probs)]
            markers = MIXED if ct == "Mixed_KC_Fibro" else CELL_TYPES[ct]

            # Neutrophils are the trap: genuinely low complexity and low depth.
            if ct == "Neutrophils":
                depth = rng.uniform(700, 1800)
                n_expressed = int(rng.uniform(180, 340))
            else:
                depth = rng.uniform(2500, 9000)
                n_expressed = int(rng.uniform(900, 2200))

            mu = np.zeros(len(genes))
            # background: a random subset of genes at low level
            picked = rng.choice(len(genes), size=min(n_expressed, len(genes)), replace=False)
            mu[picked] = rng.gamma(0.6, 1.2, size=picked.size)
            # identity markers, high
            for g in markers:
                mu[gidx[g]] = rng.gamma(9.0, 2.2)
            # housekeeping / mito / ribo
            for g, k in (("Actb", 14), ("Gapdh", 12), ("Rps6", 9), ("Rpl13", 9),
                         ("Rps19", 8), ("mt-Co1", 6), ("mt-Nd1", 4), ("mt-Cytb", 4)):
                mu[gidx[g]] = rng.gamma(k, 1.4)
            # lineage gates
            if ct in ("Macrophages", "Neutrophils", "T cells"):
                mu[gidx["Ptprc"]] = rng.gamma(7, 1.5)
            if ct in ("Keratinocytes", "Mixed_KC_Fibro"):
                mu[gidx["Epcam"]] = rng.gamma(5, 1.4)
            # ambient keratin/collagen soup in EVERY cell — the skin failure mode
            for g in AMBIENT:
                mu[gidx[g]] += rng.gamma(1.6, 1.0)
            # the burn effect
            if cond == "Burn":
                for g in BURN_UP.get(ct, []):
                    mu[gidx[g]] *= rng.uniform(2.5, 4.5)
                if tp == "D7" and ct == "Macrophages":
                    mu[gidx["Arg1"]] *= 1.8

            mu = mu / mu.sum() * depth
            rows.append(rng.poisson(mu))
            obs_rows.append({"Sample": s, "Type": cond, "Timepoint": tp,
                             "Replicate": rep, "true_celltype": ct,
                             "Batch": "b1" if rep in ("1", "2") else "b2"})

    X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{obs_rows[i]['Sample']}_{i:05d}" for i in range(len(obs_rows))]
    for c in ("Sample", "Type", "Timepoint", "Replicate", "true_celltype", "Batch"):
        obs[c] = obs[c].astype("category")

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
    adata.layers["counts"] = adata.X.copy()
    adata.uns["skinmcp"] = {"organism": "mouse", "x_state": "counts",
                            "chemistry": "10x_3prime_v3", "source": "synthetic golden"}
    return adata


if __name__ == "__main__":  # pragma: no cover
    import sys

    a = build()
    out = HERE / "golden_mouse_skin.h5ad"
    a.write_h5ad(out, compression="gzip")
    sys.stderr.write(
        f"wrote {out} — {a.n_obs} cells x {a.n_vars} genes, "
        f"{a.obs['Sample'].nunique()} samples, "
        f"{a.obs['true_celltype'].nunique()} cell types\n")
