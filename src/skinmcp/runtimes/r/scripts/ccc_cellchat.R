# CellChat per split group. skin.ccc.cellchat_r().
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
suppressPackageStartupMessages({ library(CellChat); library(SingleCellExperiment) })

skin_main(function(work, p) {
  set.seed(p$seed %||% 0)
  sce <- skin_read_sce(work)
  lk <- p$label_key; sb <- p$split_by %||% ""
  db <- if (identical(p$organism, "human")) CellChatDB.human else CellChatDB.mouse

  X <- if ("logcounts" %in% assayNames(sce)) assay(sce, "logcounts") else {
    cts <- assay(sce, "counts"); t(t(cts) / pmax(colSums(cts), 1) * 1e4) |> log1p()
  }
  meta <- as.data.frame(colData(sce))
  groups <- if (nzchar(sb) && sb %in% colnames(meta)) unique(as.character(meta[[sb]])) else "all"

  per <- list()
  for (g in groups) {
    m <- if (identical(g, "all")) rep(TRUE, ncol(sce)) else as.character(meta[[sb]]) == g
    md <- meta[m, , drop = FALSE]
    md[[lk]] <- droplevels(factor(as.character(md[[lk]])))
    keep <- names(table(md[[lk]]))[table(md[[lk]]) >= (p$min_cells %||% 30)]
    sel <- md[[lk]] %in% keep
    if (length(keep) < 2) next
    cc <- createCellChat(object = X[, m][, sel, drop = FALSE],
                         meta = md[sel, , drop = FALSE], group.by = lk)
    cc@DB <- db
    cc <- subsetData(cc); cc <- identifyOverExpressedGenes(cc)
    cc <- identifyOverExpressedInteractions(cc)
    cc <- computeCommunProb(cc, type = "triMean")
    cc <- filterCommunication(cc, min.cells = p$min_cells %||% 30)
    cc <- computeCommunProbPathway(cc); cc <- aggregateNet(cc)
    cc <- netAnalysis_computeCentrality(cc, slot.name = "netP")
    f <- file.path(work, sprintf("cellchat_%s.rds", gsub("[^A-Za-z0-9]+", "_", g)))
    saveRDS(cc, f)
    per[[length(per) + 1]] <- list(
      split = g, n_cells = sum(sel), n_populations = length(keep),
      dropped_populations = setdiff(names(table(md[[lk]])), keep),
      n_pathways = length(cc@netP$pathways), rds = f,
      top_pathways = head(cc@netP$pathways, 15))
  }
  list(per_split = per, work_dir = work,
       version = as.character(packageVersion("CellChat")))
})
