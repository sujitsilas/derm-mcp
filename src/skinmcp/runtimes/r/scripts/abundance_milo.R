# miloR with spatial FDR. skin.abundance.milo_r().
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_common.R"))
`%||%` <- function(a, b) if (is.null(a)) b else a
suppressPackageStartupMessages({ library(miloR); library(SingleCellExperiment) })

skin_main(function(work, p) {
  set.seed(p$seed %||% 0)
  sce <- skin_read_sce(work)
  rep_key <- p$use_rep %||% "X_pca_harmony"
  if (!rep_key %in% reducedDimNames(sce)) rep_key <- reducedDimNames(sce)[1]
  ck <- p$condition_key; sk <- p$sample_key %||% "Sample"
  a <- p$contrast[[1]]; b <- p$contrast[[2]]

  mi <- Milo(sce)
  mi <- buildGraph(mi, k = p$k %||% 30, d = 30, reduced.dim = rep_key)
  mi <- makeNhoods(mi, prop = p$prop %||% 0.1, k = p$k %||% 30, d = 30,
                   refined = TRUE, reduced_dims = rep_key)
  mi <- countCells(mi, meta.data = as.data.frame(colData(mi)), sample = sk)

  design <- unique(as.data.frame(colData(mi))[, c(sk, ck, p$covariates), drop = FALSE])
  rownames(design) <- design[[sk]]
  design <- design[colnames(nhoodCounts(mi)), , drop = FALSE]
  design[[ck]] <- factor(as.character(design[[ck]]), levels = c(b, a))

  covs <- intersect(p$covariates %||% character(0), colnames(design))
  covs <- covs[vapply(covs, function(cv) length(unique(design[[cv]])) > 1, logical(1))]
  form <- as.formula(paste("~", paste(c(covs, ck), collapse = " + ")))

  mi <- calcNhoodDistance(mi, d = 30, reduced.dim = rep_key)
  res <- testNhoods(mi, design = form, design.df = design, reduced.dim = rep_key)

  lk <- p$label_key
  if (!is.null(lk) && lk %in% colnames(colData(mi))) {
    mi  <- buildNhoodGraph(mi)
    res <- annotateNhoods(mi, res, coldata_col = lk)
    res[[lk]][res$nhood_annotation_frac < 0.7] <- "Mixed"
  }
  outp <- file.path(work, "milo_results.csv")
  write.csv(res, outp, row.names = FALSE)

  per <- list()
  if (!is.null(lk) && lk %in% colnames(res)) {
    for (g in unique(res[[lk]])) {
      m <- res[[lk]] == g
      per[[length(per) + 1]] <- list(
        label = g, n_nhoods = sum(m),
        n_sig = sum(res$SpatialFDR[m] < 0.1, na.rm = TRUE),
        median_lfc = round(median(res$logFC[m], na.rm = TRUE), 3))
    }
  }
  list(per_label = per, n_nhoods = nrow(res), table_path = outp,
       design = deparse(form), version = as.character(packageVersion("miloR")))
})
