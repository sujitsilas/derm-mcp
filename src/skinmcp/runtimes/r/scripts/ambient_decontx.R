# DecontX ambient RNA estimation, per sample. skin.qc.estimate_ambient(method="decontx").
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_common.R"))
`%||%` <- function(a, b) if (is.null(a)) b else a
suppressPackageStartupMessages({ library(celda); library(SingleCellExperiment) })

skin_main(function(work, p) {
  set.seed(p$seed %||% 0)
  sce <- skin_read_sce(work)
  samp_key <- p$sample_key %||% "Sample"
  batch <- if (samp_key %in% colnames(colData(sce))) as.character(colData(sce)[[samp_key]]) else NULL

  sce <- decontX(sce, batch = batch)
  cont <- as.numeric(sce$decontX_contamination)

  per <- if (is.null(batch)) {
    list(list(sample = "all", contamination = round(mean(cont), 4), n_cells = ncol(sce)))
  } else {
    lapply(unique(batch), function(s) {
      m <- batch == s
      list(sample = s, n_cells = sum(m),
           contamination = round(mean(cont[m]), 4),
           median_contamination = round(median(cont[m]), 4))
    })
  }

  out <- list(per_sample = per, mean_contamination = round(mean(cont), 4),
              version = as.character(packageVersion("celda")))
  if (isTRUE(p$apply_correction)) {
    suppressPackageStartupMessages(library(zellkonverter))
    dec <- SingleCellExperiment(assays = list(X = round(decontXcounts(sce))))
    colnames(dec) <- colnames(sce); rownames(dec) <- rownames(sce)
    outp <- file.path(work, "corrected.h5ad")
    writeH5AD(dec, outp, X_name = "X")
    out$corrected_path <- outp
  }
  out
})
