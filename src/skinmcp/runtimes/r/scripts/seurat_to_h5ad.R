# Seurat .rds -> h5ad (mtx export fallback). skin.io.load_seurat_rds().
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_common.R"))
`%||%` <- function(a, b) if (is.null(a)) b else a
suppressPackageStartupMessages({ library(Seurat); library(SingleCellExperiment) })

skin_main(function(work, p) {
  obj <- readRDS(p$input_rds)
  assay <- p$assay %||% "RNA"
  if (!assay %in% Assays(obj)) stop(sprintf("assay '%s' not in object (%s)", assay,
                                            paste(Assays(obj), collapse = ", ")))
  DefaultAssay(obj) <- assay
  sce <- as.SingleCellExperiment(obj, assay = assay)

  out <- tryCatch({
    suppressPackageStartupMessages(library(zellkonverter))
    f <- file.path(work, "converted.h5ad"); writeH5AD(sce, f, X_name = "counts")
    list(output_h5ad = f)
  }, error = function(e) {
    # Documented fallback: the mtx layout the reference notebook uses.
    d <- file.path(work, "converted_mtx"); dir.create(d, showWarnings = FALSE)
    writeMM(as(counts(sce), "dgCMatrix"), file.path(d, "matrix.mtx"))
    writeLines(rownames(sce), file.path(d, "genes.txt"))
    writeLines(colnames(sce), file.path(d, "barcodes.txt"))
    write.csv(as.data.frame(colData(sce)), file.path(d, "metadata.csv"))
    for (rd in reducedDimNames(sce))
      write.csv(reducedDim(sce, rd), file.path(d, sprintf("reducedDim_%s.csv", rd)))
    list(output_mtx = d, h5ad_error = conditionMessage(e))
  })
  c(out, list(n_cells = ncol(sce), n_genes = nrow(sce),
              version = as.character(packageVersion("Seurat"))))
})
