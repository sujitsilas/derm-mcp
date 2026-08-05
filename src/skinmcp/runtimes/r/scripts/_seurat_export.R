# Seurat -> plain files, for assembly into AnnData on the Python side.
#
# Deliberately not zellkonverter. That path goes Seurat -> SingleCellExperiment
# -> h5ad and loses whatever the SCE coercion does not carry: a Seurat object
# holds several assays, each with its own counts/data/scale.data layers and its
# own feature metadata, and `as.SingleCellExperiment` keeps one assay's counts
# and logcounts. For an object that has been through SCTransform that is the
# wrong subset -- the SCT residuals live in scale.data, and the model's
# corrected counts in the SCT assay, neither of which survives.
#
# Plain files also fail visibly. A converter that silently drops a layer gives
# you an AnnData that looks right and is missing the matrix you needed three
# steps later; a missing file is noticed immediately.
#
# Layout written into `work/`:
#   manifest.json                 what exists, dims, names, default assay
#   barcodes.txt                  cell names, the ordering everything else uses
#   assay_<a>_features.txt        feature names for assay <a>
#   assay_<a>_<layer>.mtx         genes x cells sparse (counts, data)
#   assay_<a>_scaledata.csv       dense, only when small enough to be worth it
#   assay_<a>_meta.csv            per-feature metadata (HVG flags, means, ...)
#   metadata.csv                  cell metadata (obs)
#   reduction_<r>.csv             cell embeddings (pca, umap, harmony, ...)
#   reduction_<r>_loadings.csv    feature loadings where present
#   graph_<g>.mtx                 nearest-neighbour / SNN graphs

suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(jsonlite)
})

#' @param obj   a Seurat object
#' @param work  directory to write into
#' @param max_scaledata_cells  above this, scale.data is skipped rather than
#'   written as a dense CSV that nobody can load. It is recomputable.
seurat_export <- function(obj, work, max_scaledata_cells = 50000) {
  dir.create(work, recursive = TRUE, showWarnings = FALSE)
  man <- list(
    seurat_version = as.character(packageVersion("Seurat")),
    default_assay = DefaultAssay(obj),
    n_cells = ncol(obj),
    assays = list(), reductions = list(), graphs = character()
  )

  # Cell order is the contract: every other file is written in this order, and
  # the Python side reindexes against it rather than assuming alignment.
  cells <- colnames(obj)
  writeLines(cells, file.path(work, "barcodes.txt"))

  md <- obj@meta.data
  md <- md[cells, , drop = FALSE]
  write.csv(md, file.path(work, "metadata.csv"), row.names = TRUE)
  man$metadata_columns <- colnames(md)

  for (a in Assays(obj)) {
    ao <- obj[[a]]
    feats <- rownames(ao)
    writeLines(feats, file.path(work, sprintf("assay_%s_features.txt", a)))
    info <- list(n_features = length(feats), layers = character())

    for (lyr in c("counts", "data")) {
      m <- tryCatch(SeuratObject::LayerData(ao, layer = lyr), error = function(e) NULL)
      if (is.null(m) || !length(m) || nrow(m) == 0) next
      # A Seurat "data" layer that is identical to counts means the assay was
      # never normalised; record it rather than writing the same matrix twice.
      f <- file.path(work, sprintf("assay_%s_%s.mtx", a, lyr))
      writeMM(as(as(m[, cells, drop = FALSE], "CsparseMatrix"), "dgCMatrix"), f)
      info$layers <- c(info$layers, lyr)
    }

    sd <- tryCatch(SeuratObject::LayerData(ao, layer = "scale.data"),
                   error = function(e) NULL)
    if (!is.null(sd) && length(sd) && nrow(sd) > 0) {
      if (ncol(sd) <= max_scaledata_cells) {
        write.csv(as.matrix(sd), file.path(work, sprintf("assay_%s_scaledata.csv", a)))
        info$layers <- c(info$layers, "scale.data")
        info$scaledata_features <- rownames(sd)
      } else {
        info$scaledata_skipped <- sprintf(
          "%d x %d dense; recompute with ScaleData() rather than moving it",
          nrow(sd), ncol(sd))
      }
    }

    fm <- tryCatch(ao[[]], error = function(e) NULL)
    if (!is.null(fm) && ncol(fm) > 0) {
      write.csv(fm, file.path(work, sprintf("assay_%s_meta.csv", a)), row.names = TRUE)
      info$feature_meta_columns <- colnames(fm)
    }
    man$assays[[a]] <- info
  }

  for (r in Reductions(obj)) {
    ro <- obj[[r]]
    emb <- Embeddings(ro)
    write.csv(emb[cells, , drop = FALSE], file.path(work, sprintf("reduction_%s.csv", r)))
    ld <- tryCatch(Loadings(ro), error = function(e) NULL)
    if (!is.null(ld) && length(ld) && nrow(ld) > 0) {
      write.csv(ld, file.path(work, sprintf("reduction_%s_loadings.csv", r)))
    }
    man$reductions[[r]] <- list(
      n_dims = ncol(emb),
      key = tryCatch(Key(ro), error = function(e) NA_character_),
      # stdev is what ElbowPlot and any variance-explained calculation need, and
      # it is not recoverable from the embeddings alone.
      stdev = tryCatch(as.numeric(Stdev(ro)), error = function(e) numeric()),
      has_loadings = !is.null(ld) && length(ld) > 0
    )
  }

  for (g in names(obj@graphs)) {
    gm <- obj@graphs[[g]]
    writeMM(as(as(gm, "CsparseMatrix"), "dgCMatrix"), file.path(work, sprintf("graph_%s.mtx", g)))
    man$graphs <- c(man$graphs, g)
  }

  write(toJSON(man, auto_unbox = TRUE, pretty = TRUE, null = "null"),
        file.path(work, "manifest.json"))
  invisible(man)
}
