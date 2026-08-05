# mergeCellChat: information-flow scatter + pathway heatmap. skin.ccc.cellchat_compare().
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
suppressPackageStartupMessages({ library(CellChat); library(ggplot2) })

skin_main(function(work, p) {
  objs <- list(); nms <- character(0)
  for (i in seq_along(p$work_dirs)) {
    d <- p$work_dirs[[i]]
    if (is.null(d) || !dir.exists(d)) next
    for (f in list.files(d, pattern = "^cellchat_.*\\.rds$", full.names = TRUE)) {
      objs[[length(objs) + 1]] <- readRDS(f)
      nms <- c(nms, sub("^cellchat_", "", tools::file_path_sans_ext(basename(f))))
    }
  }
  if (length(objs) < 2) stop("need at least two CellChat objects to compare")
  names(objs) <- make.unique(nms)
  merged <- mergeCellChat(objs, add.names = names(objs))

  figs <- character(0)
  g1 <- rankNet(merged, mode = "comparison", stacked = FALSE, do.stat = TRUE)
  f1 <- file.path(work, "information_flow.pdf"); ggsave(f1, g1, width = 7, height = 9)
  figs <- c(figs, f1)

  flow <- rankNet(merged, mode = "comparison", stacked = FALSE, return.data = TRUE)
  df <- if (is.list(flow) && !is.null(flow$signaling.contribution)) flow$signaling.contribution else flow
  write.csv(df, file.path(work, "information_flow.csv"), row.names = FALSE)

  list(figures = figs, information_flow = head(as.data.frame(df), 40),
       datasets = names(objs), version = as.character(packageVersion("CellChat")))
})
