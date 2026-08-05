# Shared helpers for every vetted skin-mcp R script.
#
# Contract, enforced by runtimes/bridge.py:
#   argv[1]           work directory, bind-mounted at /work
#   /work/params.json typed parameters from the tool (never R source)
#   /work/input.h5ad  the AnnData, when the tool passed one
#   /work/input_mtx/  fallback layout when the h5ad round-trip failed
#   /work/result.json what the tool reads back. Writing it is mandatory.
suppressPackageStartupMessages({
  library(jsonlite); library(Matrix)
})

skin_work <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) < 1) stop("usage: Rscript <script>.R <work_dir>")
  normalizePath(a[[1]], mustWork = TRUE)
}

skin_params <- function(work) {
  p <- file.path(work, "params.json")
  if (!file.exists(p)) return(list())
  fromJSON(p, simplifyVector = TRUE)
}

# Plain files are the only transport, in both directions.
#
# This used to prefer zellkonverter::readH5AD and fall back to mtx. zellkonverter
# reaches Python through basilisk/reticulate, so it carries an entire second
# runtime to go wrong -- on this machine it did, trying to provision Python
# 3.14.0 through pyenv and failing before it read a single cell. A converter
# whose failure mode is "could not install an interpreter" is the wrong
# dependency for a data handoff between two runtimes that are both already
# installed and working.
skin_read_sce <- function(work) {
  d <- file.path(work, "input_mtx")
  if (!dir.exists(d)) stop("no input object found in ", work)
  suppressPackageStartupMessages(library(SingleCellExperiment))
  m     <- readMM(file.path(d, "matrix.mtx"))           # genes x cells
  genes <- readLines(file.path(d, "genes.txt"))
  bcs   <- readLines(file.path(d, "barcodes.txt"))
  rownames(m) <- genes; colnames(m) <- bcs
  meta  <- read.csv(file.path(d, "metadata.csv"), row.names = 1, check.names = FALSE)
  sce   <- SingleCellExperiment(assays = list(counts = m),
                                colData = meta[colnames(m), , drop = FALSE])
  for (f in list.files(d, pattern = "^reducedDim_", full.names = TRUE)) {
    key <- sub("^reducedDim_", "", tools::file_path_sans_ext(basename(f)))
    reducedDim(sce, key) <- as.matrix(read.csv(f, row.names = 1))
  }
  sce
}

skin_write_result <- function(work, obj) {
  write(toJSON(obj, auto_unbox = TRUE, digits = 8, null = "null", na = "null"),
        file.path(work, "result.json"))
}

# Any error still has to leave a machine-readable result.json behind, otherwise
# the Python side can only report "exited non-zero".
skin_main <- function(fn) {
  work <- skin_work()
  res <- tryCatch(fn(work, skin_params(work)),
                  error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  if (is.null(res$ok)) res$ok <- TRUE
  skin_write_result(work, res)
  if (!isTRUE(res$ok)) quit(status = 1)
}
