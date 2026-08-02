# scDblFinder, ALWAYS per sample. Called by skin.doublet.call(method="scdblfinder").
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_common.R"))
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
