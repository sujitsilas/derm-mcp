# Seurat .rds -> plain files, for assembly into AnnData. skin.io.load_seurat_rds().
#
# The name is historical: nothing here writes .h5ad any more. It used to try
# zellkonverter::writeH5AD with an mtx fallback, and lost most of the object
# either way -- `as.SingleCellExperiment` keeps one assay's counts and
# logcounts, so an object that had been through SCTransform arrived missing the
# SCT assay, its scale.data, every reduction's loadings, and the PCA stdev that
# variance-explained calculations need. zellkonverter also reaches Python
# through basilisk, which is a second runtime to go wrong before a single cell
# is read.
#
# seurat_export() writes all of it; runtimes/seurat_import.py assembles it.
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
source(file.path(.skin_script_dir(), "_seurat_export.R"))
suppressPackageStartupMessages({ library(Seurat) })

skin_main(function(work, p) {
  obj <- readRDS(p$input_rds)
  assay <- p$assay %||% ""
  if (nzchar(assay)) {
    if (!assay %in% Assays(obj))
      stop(sprintf("assay '%s' not in object (%s)", assay,
                   paste(Assays(obj), collapse = ", ")))
    DefaultAssay(obj) <- assay
  }
  d <- file.path(work, "seurat_files")
  man <- seurat_export(obj, d)
  list(output_files = d,
       default_assay = man$default_assay,
       assays = names(man$assays),
       reductions = names(man$reductions),
       n_cells = man$n_cells,
       version = as.character(packageVersion("Seurat")))
})
