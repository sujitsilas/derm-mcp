# Pinned R runtime for skin-mcp.
#
# The base is pinned by DIGEST, not by tag: `rocker/r-ver:4.4.2` is mutable and a
# rebuild six months from now would silently give you a different R. Replace the
# digest below when you deliberately upgrade, and record the change.
#
#   docker build -t skin-mcp-r:0.1.0 src/skinmcp/runtimes/r
#
# Verify the digest for your architecture with:
#   docker buildx imagetools inspect rocker/r-ver:4.4.2
FROM rocker/r-ver:4.4.2@sha256:8a2f1f8dd0c1f6e1e51d63e2eb9b2f8c1b0dbb0f7e7e4ba6e0ba7f0f6f38c2a1 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    R_REMOTES_NO_ERRORS_FROM_WARNINGS=true \
    RENV_CONFIG_REPOS_OVERRIDE=https://packagemanager.posit.co/cran/2025-01-15

RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4-openssl-dev libssl-dev libxml2-dev libhdf5-dev libglpk-dev \
      libpng-dev libjpeg-dev libtiff5-dev libfreetype6-dev libharfbuzz-dev \
      libfribidi-dev libgit2-dev libfontconfig1-dev zlib1g-dev cmake git \
    && rm -rf /var/lib/apt/lists/*

# renv.lock is the source of truth for package versions. Copying it first keeps
# the (slow) restore layer cached across script edits.
WORKDIR /opt/skinmcp
COPY renv.lock /opt/skinmcp/renv.lock

RUN R -q -e "install.packages('renv', repos='https://cloud.r-project.org')" \
 && R -q -e "renv::consent(provided=TRUE); renv::restore(lockfile='/opt/skinmcp/renv.lock', prompt=FALSE)"

# Vetted scripts are bind-mounted read-only at run time (see runtimes/bridge.py),
# so they are deliberately NOT baked into the image — editing one must not require
# a rebuild.
WORKDIR /work
CMD ["R", "--version"]
