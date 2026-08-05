# CellChat chord diagram. skin.ccc.plot_chord().
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
