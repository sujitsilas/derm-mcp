# Plain files -> Seurat. The mirror of seurat_export.R.
#
# Reads the layout written by either side -- `seurat_export()` here, or
# `runtimes/seurat_import.py:write_interchange()` in Python -- so an object can
# make the round trip without a converter in the middle. See seurat_export.R for
# why that is a file layout and not zellkonverter.
#
# What is reconstructed: every assay's counts/data layers and feature metadata,
# cell metadata, scale.data where it was small enough to be written, every
# reduction (with loadings and stdev, so ElbowPlot and variance-explained still
# work), and every cell-by-cell graph.

suppressPackageStartupMessages({
  library(Seurat); library(SeuratObject); library(Matrix); library(jsonlite)
})

#' @param work directory holding manifest.json and friends
#' @param assay which assay to make the default; "" means the manifest's
seurat_from_files <- function(work, assay = "") {
  man <- fromJSON(file.path(work, "manifest.json"), simplifyVector = FALSE)
  cells <- readLines(file.path(work, "barcodes.txt"), warn = FALSE)
  assay_names <- names(man$assays)
  if (!length(assay_names)) stop("manifest lists no assays in ", work)
  primary <- if (nzchar(assay)) assay else (man$default_assay %||% assay_names[[1]])
  if (!primary %in% assay_names)
    stop(sprintf("assay '%s' not in export (%s)", primary,
                 paste(assay_names, collapse = ", ")))

  read_layer <- function(a, lyr) {
    f <- file.path(work, sprintf("assay_%s_%s.mtx", a, lyr))
    if (!file.exists(f)) return(NULL)
    m <- readMM(f)                                    # genes x cells
    feats <- readLines(file.path(work, sprintf("assay_%s_features.txt", a)), warn = FALSE)
    rownames(m) <- feats; colnames(m) <- cells
    as(as(m, "CsparseMatrix"), "dgCMatrix")
  }

  counts <- read_layer(primary, "counts")
  data_m <- read_layer(primary, "data")
  if (is.null(counts) && is.null(data_m))
    stop("neither counts nor data present for assay ", primary)

  md <- NULL
  mdf <- file.path(work, "metadata.csv")
  if (file.exists(mdf)) {
    md <- read.csv(mdf, row.names = 1, check.names = FALSE)
    md <- md[cells, , drop = FALSE]
  }

  # CreateSeuratObject insists on counts; when only a normalised matrix crossed
  # we seed with it and overwrite the data layer, rather than inventing counts.
  seed <- if (!is.null(counts)) counts else data_m
  obj <- CreateSeuratObject(counts = seed, assay = primary, meta.data = md)
  if (!is.null(data_m)) {
    obj <- SetAssayData(obj, assay = primary, layer = "data", new.data = data_m)
  } else if (!is.null(counts)) {
    obj <- NormalizeData(obj, assay = primary, verbose = FALSE)
  }

  sdf <- file.path(work, sprintf("assay_%s_scaledata.csv", primary))
  if (file.exists(sdf)) {
    sd <- as.matrix(read.csv(sdf, row.names = 1, check.names = FALSE))
    obj <- SetAssayData(obj, assay = primary, layer = "scale.data",
                        new.data = sd[, cells, drop = FALSE])
  }

  fmf <- file.path(work, sprintf("assay_%s_meta.csv", primary))
  if (file.exists(fmf)) {
    fm <- read.csv(fmf, row.names = 1, check.names = FALSE)
    keep <- intersect(rownames(fm), rownames(obj[[primary]]))
    if (length(keep)) obj[[primary]] <- AddMetaData(obj[[primary]], fm[keep, , drop = FALSE])
  }

  # Secondary assays come across as their own Assay objects.
  for (a in setdiff(assay_names, primary)) {
    ac <- read_layer(a, "counts"); ad_ <- read_layer(a, "data")
    if (is.null(ac) && is.null(ad_)) next
    obj[[a]] <- CreateAssayObject(counts = if (!is.null(ac)) ac else ad_)
    if (!is.null(ad_)) obj <- SetAssayData(obj, assay = a, layer = "data", new.data = ad_)
  }

  for (r in names(man$reductions %||% list())) {
    f <- file.path(work, sprintf("reduction_%s.csv", r))
    if (!file.exists(f)) next
    emb <- as.matrix(read.csv(f, row.names = 1, check.names = FALSE))
    emb <- emb[cells, , drop = FALSE]
    rinfo <- man$reductions[[r]]
    ld <- NULL
    lf <- file.path(work, sprintf("reduction_%s_loadings.csv", r))
    if (file.exists(lf)) ld <- as.matrix(read.csv(lf, row.names = 1, check.names = FALSE))
    sdv <- as.numeric(unlist(rinfo$stdev %||% numeric()))
    key <- rinfo$key %||% paste0(r, "_")
    colnames(emb) <- paste0(key, seq_len(ncol(emb)))
    obj[[r]] <- CreateDimReducObject(
      embeddings = emb,
      loadings = if (!is.null(ld)) ld else new("matrix"),
      stdev = if (length(sdv)) sdv else numeric(),
      key = key, assay = primary)
  }

  for (g in unlist(man$graphs %||% character())) {
    f <- file.path(work, sprintf("graph_%s.mtx", g))
    if (!file.exists(f)) next
    gm <- readMM(f)
    rownames(gm) <- cells; colnames(gm) <- cells
    obj[[g]] <- as.Graph(as(as(gm, "CsparseMatrix"), "dgCMatrix"))
  }

  DefaultAssay(obj) <- primary
  obj
}
