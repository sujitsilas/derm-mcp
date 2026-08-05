# scDblFinder, ALWAYS per sample. Called by skin.doublet.call(method="scdblfinder").
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
suppressPackageStartupMessages({ library(scDblFinder); library(SingleCellExperiment) })

skin_main(function(work, p) {
  set.seed(p$seed %||% 0)
  sce <- skin_read_sce(work)
  samp_key <- p$sample_key %||% "Sample"
  if (!samp_key %in% colnames(colData(sce)))
    stop(sprintf("sample_key '%s' not in colData", samp_key))
  samples <- as.character(colData(sce)[[samp_key]])

  # dbr per sample from the 10x multiplet table the Python side computed.
  rates <- p$expected_rates
  dbr <- if (!is.null(rates)) unname(unlist(rates))[match(samples, names(rates))] else NULL

  sce <- scDblFinder(sce, samples = samples, dbr = dbr, BPPARAM = BiocParallel::SerialParam())
  scores <- as.numeric(sce$scDblFinder.score)
  pred   <- as.character(sce$scDblFinder.class) == "doublet"

  per <- lapply(unique(samples), function(s) {
    m <- samples == s
    list(sample = s, n_cells = sum(m), n_called = sum(pred[m]),
         rate = round(mean(pred[m]), 4),
         expected_rate = if (!is.null(rates) && !is.null(rates[[s]])) rates[[s]] else NA)
  })
  list(scores = scores, predicted = pred, per_sample = per,
       version = as.character(packageVersion("scDblFinder")))
})
