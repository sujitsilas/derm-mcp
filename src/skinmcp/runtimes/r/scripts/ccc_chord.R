# CellChat chord diagram. skin.ccc.plot_chord().
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_common.R"))
`%||%` <- function(a, b) if (is.null(a)) b else a
suppressPackageStartupMessages({ library(CellChat) })

skin_main(function(work, p) {
  d <- p$work_dir
  fs <- list.files(d, pattern = "^cellchat_.*\\.rds$", full.names = TRUE)
  if (!length(fs)) stop("no CellChat objects in ", d)
  figs <- character(0)
  for (f in fs) {
    cc <- readRDS(f)
    tag <- sub("^cellchat_", "", tools::file_path_sans_ext(basename(f)))
    paths <- if (length(p$pathways)) intersect(p$pathways, cc@netP$pathways)
             else head(cc@netP$pathways, 1)
    for (pw in paths) {
      out <- file.path(work, sprintf("chord_%s_%s.pdf", tag, gsub("[^A-Za-z0-9]+", "_", pw)))
      pdf(out, width = 7, height = 7)
      tryCatch(netVisual_aggregate(cc, signaling = pw, layout = "chord"),
               error = function(e) plot.new())
      dev.off()
      figs <- c(figs, out)
    }
  }
  list(figures = figs, version = as.character(packageVersion("CellChat")))
})
