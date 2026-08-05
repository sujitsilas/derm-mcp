# Reference DESeq2 on the pseudobulk matrix. skin.de.deseq2_r().
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
suppressPackageStartupMessages({ library(DESeq2) })

skin_main(function(work, p) {
  set.seed(p$seed %||% 0)
  d <- p$input_dir
  counts <- as.matrix(read.csv(file.path(d, "counts.csv"), row.names = 1, check.names = FALSE))
  meta   <- read.csv(file.path(d, "metadata.csv"), row.names = 1, check.names = FALSE)
  ck <- p$condition_key; a <- p$contrast[[1]]; b <- p$contrast[[2]]
  covs <- setdiff(p$covariates %||% character(0), "")

  per <- list()
  for (lb in unique(meta$label)) {
    keep <- meta$label == lb
    m <- meta[keep, , drop = FALSE]; cts <- counts[, rownames(m), drop = FALSE]
    m[[ck]] <- factor(as.character(m[[ck]]), levels = c(b, a))   # reference first
    use <- covs[vapply(covs, function(cv) length(unique(m[[cv]])) > 1, logical(1))]
    for (cv in use) m[[cv]] <- factor(as.character(m[[cv]]))
    if (min(table(m[[ck]])) < 2) next
    form <- as.formula(paste("~", paste(c(use, ck), collapse = " + ")))
    cts <- cts[rowSums(cts) >= 10, , drop = FALSE]
    dds <- DESeqDataSetFromMatrix(round(cts), colData = m, design = form)
    dds <- DESeq(dds, quiet = TRUE)
    res <- as.data.frame(results(dds, contrast = c(ck, a, b)))
    res$gene <- rownames(res)
    names(res)[names(res) == "log2FoldChange"] <- "lfc"
    outp <- file.path(work, sprintf("de_%s.csv", gsub("[^A-Za-z0-9]+", "_", lb)))
    write.csv(res[order(res$padj), ], outp, row.names = FALSE)
    per[[length(per) + 1]] <- list(
      label = lb, design = deparse(form), table_path = outp,
      n_up = sum(res$padj < 0.05 & res$lfc > 0.5, na.rm = TRUE),
      n_down = sum(res$padj < 0.05 & res$lfc < -0.5, na.rm = TRUE))
  }
  list(per_label = per, version = as.character(packageVersion("DESeq2")))
})
