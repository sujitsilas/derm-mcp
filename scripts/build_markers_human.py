"""Generate knowledge/markers_human.yaml from the mouse file via the shipped
MGI ortholog table, then apply hand-curated overrides.

Naive .upper() is wrong often enough to matter (Trp63->TP63, H2-Aa->HLA-DRA,
Lyz2->LYZ, Wisp2->CCN5), and several mouse genes have no human ortholog at all
(Ly6c1/2, Retnlg, Chil3, Ly6a). Run this, don't hand-edit the output.

    python scripts/build_markers_human.py
"""
import sys
from pathlib import Path

import yaml

K = Path(__file__).resolve().parents[1] / "src/skinmcp/knowledge"

# mouse -> human overrides that MGI gets wrong, is missing, or that need a
# biologically-equivalent substitute rather than a sequence ortholog.
OVERRIDE = {
    "H2-Aa": ["HLA-DRA"], "H2-Ab1": ["HLA-DRB1"], "H2-Eb1": ["HLA-DPA1"],
    "Lyz2": ["LYZ"], "Trp63": ["TP63"], "Wisp2": ["CCN5"], "Dsg1a": ["DSG1"],
    "Dsc1": ["DSC1"], "Lor": ["LOR"], "Flg": ["FLG"], "Ivl": ["IVL"],
    "Sdr16c6": ["SDR16C5"], "Ccl21a": ["CCL21"], "Klrb1c": ["KLRB1"],
    "Mcpt4": ["CMA1", "TPSAB1"], "Cma1": ["CMA1"], "Cpa3": ["CPA3"],
    "Hba-a1": ["HBA1"], "Hba-a2": ["HBA2"], "Hbb-bs": ["HBB"], "Hbb-bt": ["HBB"],
    "Acod1": ["ACOD1"], "Ncapd2": ["NCAPD2"], "Cenpu": ["CENPU"],
    "Trgv4": ["TRGV9"], "Trgv5": [], "Cd163l1": ["CD163L1"],
    "Selenop": ["SELENOP"], "Gngt2": ["GNGT2"], "Ms4a7": ["MS4A7"],
    "Atp6v0d2": ["ATP6V0D2"], "Il1rl1": ["IL1RL1"], "Cd8b1": ["CD8B"],
    "Cd40lg": ["CD40LG"], "Krt6a": ["KRT6A"], "Krt6b": ["KRT6B"],
    "Sprr1a": ["SPRR1A"], "Pi16": ["PI16"], "Dpp4": ["DPP4"], "Cd34": ["CD34"],
    "Has1": ["HAS1"], "Ptgs2": ["PTGS2"], "Ccl3": ["CCL3"], "Ccl4": ["CCL4"],
    # No human ortholog exists. Substituting the accepted functional equivalent
    # and recording the substitution is better than dropping the set silently.
    "Ly6c1": [], "Ly6c2": [],            # classical monocyte -> FCN1/S100A12/VCAN
    "Ly6a": [],                          # Sca-1, rodent-specific
    "Retnlg": [],                        # -> FCGR3B / CXCR2
    "Chil3": [],                         # Ym1, rodent-specific
    "Mmrn1": ["MMRN1"],
    # MGI maps these into many:many classes where the first hit is the wrong paralog.
    "Trac": ["TRAC"], "Trdc": ["TRDC"], "Cxcl2": ["CXCL2"], "Cxcl9": ["CXCL9"],
    "Ccl2": ["CCL2"], "Gzmk": ["GZMK"], "Il17a": ["IL17A"], "Mcm4": ["MCM4"],
    "Tubb4b": ["TUBB4B"], "Cks1b": ["CKS1B"], "Hbb-b1": ["HBB"],
}

# Sets that need extra human-native genes because the mouse set lost members.
ADD = {
    ("lineages", "Monocytes"): ["FCN1", "S100A12", "CD14", "FCGR3A", "LYZ"],
    ("lineages", "Neutrophils"): ["FCGR3B", "CSF3R", "CXCR2", "FCGR2A"],
    ("lineages", "Macrophages"): ["CD68", "CD163", "MSR1"],
    ("lineages", "gdT / DETC"): ["TRDC", "TRGC1", "TRGC2"],
    ("subtypes", "macrophage", "Inf. Mono."): ["FCN1", "S100A12", "CD14"],
    ("subtypes", "macrophage_origin", "Inflammatory Monocytes"): ["FCN1", "S100A12", "CD14"],
    ("subtypes", "macrophage_origin", "MΦ-Recruited"): ["CD14", "FCN1"],
}


def load_orthologs():
    m2h = {}
    for line in (K / "orthologs_mm_hs.tsv").read_text().splitlines():
        if line.startswith("#") or line.startswith("mouse_symbol"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        m, h, rel = parts[0], parts[1], parts[2]
        if rel == "1:1" or m not in m2h:
            m2h.setdefault(m, []).append(h)
    return m2h


def main() -> int:
    m2h = load_orthologs()
    mouse = yaml.safe_load((K / "markers_mouse.yaml").read_text())
    unmapped: list[str] = []

    def conv(genes):
        out = []
        for g in genes:
            if g in OVERRIDE:
                out.extend(OVERRIDE[g])
                continue
            hits = m2h.get(g)
            if hits:
                out.extend(hits[:1])
            else:
                unmapped.append(g)
        seen, uniq = set(), []
        for g in out:
            if g and g not in seen:
                seen.add(g)
                uniq.append(g)
        return uniq

    human = {"organism": "human"}
    human["lineages"] = {}
    for k, v in mouse["lineages"].items():
        human["lineages"][k] = conv(v) + ADD.get(("lineages", k), [])
        human["lineages"][k] = list(dict.fromkeys(human["lineages"][k]))
    human["compartments"] = mouse["compartments"]
    human["gates"] = {k: conv(v) for k, v in mouse["gates"].items()}
    human["exclusive_pairs"] = mouse["exclusive_pairs"]
    human["subtypes"] = {}
    for fam, sets_ in mouse["subtypes"].items():
        human["subtypes"][fam] = {}
        for name, genes in sets_.items():
            g = conv(genes) + ADD.get(("subtypes", fam, name), [])
            human["subtypes"][fam][name] = list(dict.fromkeys(g))
    human["cell_cycle"] = {k: conv(v) for k, v in mouse["cell_cycle"].items()}
    human["qc_patterns"] = {
        "mito": "^MT-", "ribo": "^RP[SL]", "hb": "^HB[AB]",
        "ambient_keratin": r"^KRT\d", "ambient_collagen": r"^COL\d+[A-Z]{1,2}\d+$",
        "ambient_cornified": "^(SBSN|LOR|FLG|KRTDAP|CNFN|TCHH)$",
        "neutrophil_probe": "^(S100A8|S100A9|FCGR3B|MMP9)$",
    }
    human["notes"] = {
        "no_human_ortholog": sorted(set(unmapped)),
        "adgre1": ("ADGRE1 (EMR1) is eosinophil-restricted in human and is NOT a "
                   "pan-macrophage marker as Adgre1/F4-80 is in mouse. Use CD68, "
                   "CD163, MSR1, MRC1 instead."),
        "ly6c": ("Ly6c1/Ly6c2 have no human ortholog. Classical monocytes are "
                 "identified by CD14, FCN1, S100A12, VCAN, SELL."),
        "detc": "Dendritic epidermal T cells (DETC) are a mouse-specific population.",
        "generated_by": "scripts/build_markers_human.py — do not hand-edit",
    }

    out = K / "markers_human.yaml"
    with out.open("w", encoding="utf-8") as f:
        f.write("# GENERATED by scripts/build_markers_human.py from markers_mouse.yaml\n")
        f.write("# via the shipped MGI ortholog table. Edit the mouse file or the\n")
        f.write("# OVERRIDE/ADD tables in that script, then re-run. Do not hand-edit.\n")
        yaml.safe_dump(human, f, sort_keys=False, allow_unicode=True, width=100)
    sys.stderr.write(f"wrote {out}\nunmapped mouse symbols: {sorted(set(unmapped))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
