# DecontX ambient RNA estimation, per sample. skin.qc.estimate_ambient(method="decontx").
# Locating this script's own directory has to work when Rscript runs the file
# directly, which is how bridge.py invokes it: `Rscript <script>.R <work_dir>`.
# `sys.frame(1)$ofile` only exists under source(), and at top level it aborts
# with "not that many frames on the stack" -- so every vetted script failed on
# line 2, before loading a package or reading an argument.
.skin_script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) return(dirname(normalizePath(f[[1]])))
  of <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(of)) return(dirname(normalizePath(of)))
  "."
}
source(file.path(.skin_script_dir(), "_common.R"))
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
    # Written as mtx + names, the same layout the Python side already reads.
    # No h5ad writer here means no zellkonverter and no embedded Python.
    d <- file.path(work, "corrected_mtx"); dir.create(d, showWarnings = FALSE)
    m <- as(as(round(decontXcounts(sce)), "CsparseMatrix"), "dgCMatrix")
    writeMM(m, file.path(d, "matrix.mtx"))
    writeLines(rownames(sce), file.path(d, "genes.txt"))
    writeLines(colnames(sce), file.path(d, "barcodes.txt"))
    out$corrected_mtx <- d
  }
  out
})
