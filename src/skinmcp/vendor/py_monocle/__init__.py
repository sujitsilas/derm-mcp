"""A self-contained principal-graph learner, API-compatible with `py_monocle`.

WHY THIS EXISTS
---------------
The spec asked for `github.com/bioturing/py-monocle` to be vendored at a pinned
commit. That repository publishes **no license** (GitHub reports
`NOASSERTION`), so redistributing its source inside this package is not
something we can do. Instead this module implements the same published
algorithm from the primary sources:

  * Mao, Wang, Yan & Zeng, "SimplePPT: A Simple Principal Tree Algorithm",
    SDM 2015 — the reverse-graph-embedding objective and its alternating
    optimisation.
  * Cao et al., Nature 2019 (monocle3) — the k-means-centroid initialisation,
    MST-based tree construction, graph pruning, and geodesic pseudotime from a
    root principal point.

`skin.traj.monocle` prefers the real `py_monocle` when the user has installed it
and falls back here otherwise, tagging the return with which implementation ran.
Results are close but not bit-identical; the tool says so rather than implying
parity.

Public surface mirrors py_monocle:

    projected_points, mst, centroids = learn_graph(matrix, clusters=None,
                                                   n_centroids=..., prune=True,
                                                   p_threshold=...)
    pseudotime = order_cells(matrix, centroids, mst=..., projected_points=...,
                             root_pr_cells=...)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree

__all__ = ["learn_graph", "order_cells", "IMPLEMENTATION"]

IMPLEMENTATION = "skinmcp-simpleppt"


def _init_centroids(matrix: np.ndarray, n_centroids: int, clusters: Any,
                    random_state: int) -> np.ndarray:
    """k-means centroids, seeded per cluster when a partition is supplied."""
    from sklearn.cluster import KMeans

    X = np.asarray(matrix, dtype=float)
    n_centroids = int(min(max(n_centroids, 2), max(2, X.shape[0] // 2)))
    if clusters is None:
        km = KMeans(n_clusters=n_centroids, n_init=10, random_state=random_state)
        return km.fit(X).cluster_centers_

    clu = np.asarray(clusters)
    uniq = np.unique(clu)
    # Allocate centroids proportionally to cluster size, at least one each.
    sizes = np.array([(clu == u).sum() for u in uniq], dtype=float)
    alloc = np.maximum(1, np.round(n_centroids * sizes / sizes.sum()).astype(int))
    out = []
    for u, k in zip(uniq, alloc):
        pts = X[clu == u]
        k = int(min(k, len(pts)))
        if k <= 1:
            out.append(pts.mean(0, keepdims=True))
            continue
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        out.append(km.fit(pts).cluster_centers_)
    return np.vstack(out)


def _soft_assign(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
    """Responsibility matrix of cells to centroids (the SimplePPT E-step)."""
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    logits = -d2 / max(sigma, 1e-12)
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-300)


def _mst_from_centroids(C: np.ndarray) -> sp.csr_matrix:
    """Euclidean MST over the principal points."""
    d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    return minimum_spanning_tree(sp.csr_matrix(d))


def learn_graph(
    matrix: Any,
    clusters: Any = None,
    n_centroids: int = 20,
    prune: bool = True,
    p_threshold: float = 14.0,
    max_iter: int = 30,
    sigma: float | None = None,
    gamma: float = 10.0,
    random_state: int = 0,
    tol: float = 1e-4,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Fit a principal tree to `matrix`.

    Args:
        matrix: (n_cells, n_dims) coordinates. Typically X_umap (2D) or a
            PCA/Harmony embedding.
        clusters: Optional integer cluster labels used to seed the centroids.
        n_centroids: Number of principal points.
        prune: Remove spur branches whose length is below `p_threshold`.
        p_threshold: Pruning length threshold, in the units of `matrix`.
        max_iter: Alternating-optimisation iterations.
        sigma: Soft-assignment bandwidth. None = median nearest-centroid distance.
        gamma: Weight on the tree-length term relative to the reconstruction term.
        random_state: RNG seed.
        tol: Relative convergence tolerance on the objective.

    Returns:
        (projected_points, mst, centroids) — matching the py_monocle signature.
        `projected_points` is each cell's position projected onto the tree.
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("matrix must be (n_cells, n_dims) with at least 3 cells")

    C = _init_centroids(X, n_centroids, clusters, random_state)
    if sigma is None:
        d0 = cKDTree(C).query(X)[0]
        sigma = float(np.median(d0) ** 2) or 1.0

    prev_obj = np.inf
    W = _mst_from_centroids(C)
    for _ in range(max_iter):
        W = _mst_from_centroids(C)
        P = _soft_assign(X, C, sigma)

        # M-step: minimise ||X - P C||^2 + gamma * sum_edges ||c_i - c_j||^2.
        # Closed form via a Laplacian-regularised least squares on the centroids.
        n_c = C.shape[0]
        diagP = np.asarray(P.sum(0)).ravel()
        Wsym = ((W + W.T) > 0).astype(float).toarray()
        L = np.diag(Wsym.sum(1)) - Wsym
        A = np.diag(diagP) + gamma * L + 1e-8 * np.eye(n_c)
        B = P.T @ X
        C = np.linalg.solve(A, B)

        obj = float(((X - P @ C) ** 2).sum() + gamma * (Wsym * _pairdist2(C)).sum() / 2)
        if abs(prev_obj - obj) <= tol * max(abs(prev_obj), 1.0):
            break
        prev_obj = obj

    W = _mst_from_centroids(C)
    if prune:
        C, W = _prune(C, W, X, float(p_threshold))

    projected = _project_to_graph(X, C, W)
    return projected, sp.csr_matrix(W), C


def _pairdist2(C: np.ndarray) -> np.ndarray:
    return ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)


def _prune(C: np.ndarray, W: sp.csr_matrix, X: np.ndarray,
           p_threshold: float) -> tuple[np.ndarray, sp.csr_matrix]:
    """Drop leaf branches shorter than `p_threshold` that own few cells.

    Mirrors monocle3's `prune_graph`: spurs that neither span a meaningful
    distance nor carry cells are noise from the centroid initialisation.
    """
    A = ((W + W.T) > 0).astype(np.int8).toarray()
    owner = cKDTree(C).query(X)[1]
    counts = np.bincount(owner, minlength=C.shape[0])
    min_cells = max(3, int(0.005 * X.shape[0]))

    keep = np.ones(C.shape[0], dtype=bool)
    changed = True
    while changed:
        changed = False
        deg = A[np.ix_(keep, keep)].sum(1)
        idx = np.where(keep)[0]
        for local, node in enumerate(idx):
            if deg[local] != 1:
                continue
            nb = idx[np.where(A[np.ix_([node], idx)].ravel() > 0)[0]]
            if nb.size == 0:
                continue
            length = float(np.linalg.norm(C[node] - C[nb[0]]))
            if length < p_threshold and counts[node] < min_cells:
                keep[node] = False
                changed = True
        if keep.sum() < 3:
            keep[:] = True
            break

    C2 = C[keep]
    return C2, _mst_from_centroids(C2)


def _project_to_graph(X: np.ndarray, C: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """Project each cell onto its closest point on the nearest tree edge."""
    A = sp.triu(((W + W.T) > 0).astype(np.int8), k=1)
    ii, jj = A.nonzero()
    if ii.size == 0:
        return C[cKDTree(C).query(X)[1]]
    P = C[ii]
    Q = C[jj]
    D = Q - P
    denom = (D ** 2).sum(1)
    denom[denom == 0] = 1e-12
    # t = clamp((x - p).d / |d|^2, 0, 1) per edge, vectorised over cells.
    t = np.clip(((X[:, None, :] - P[None, :, :]) * D[None, :, :]).sum(-1) / denom, 0.0, 1.0)
    proj = P[None, :, :] + t[..., None] * D[None, :, :]
    d2 = ((X[:, None, :] - proj) ** 2).sum(-1)
    best = d2.argmin(1)
    return proj[np.arange(X.shape[0]), best]


def order_cells(
    matrix: Any,
    centroids: Any,
    mst: Any = None,
    projected_points: Any = None,
    root_pr_cells: Any = 0,
) -> np.ndarray:
    """Geodesic pseudotime from a root principal point.

    Args:
        matrix: (n_cells, n_dims) coordinates.
        centroids: Principal points from `learn_graph`.
        mst: The tree from `learn_graph`.
        projected_points: Ignored (kept for API compatibility); cells are
            assigned to their nearest principal point.
        root_pr_cells: Index of the root principal point, or an array of root
            cell indices whose nearest principal point becomes the root.

    Returns:
        Per-cell pseudotime. Cells on a component disconnected from the root get
        `inf`, matching py_monocle's behaviour.
    """
    X = np.asarray(matrix, dtype=float)
    C = np.asarray(centroids, dtype=float)
    if mst is None:
        mst = _mst_from_centroids(C)
    W = sp.csr_matrix(mst)
    Wsym = W + W.T
    # Edge weights must be geometric distances, not the MST's stored values.
    ii, jj = sp.triu(Wsym, k=1).nonzero()
    d = np.sqrt(((C[ii] - C[jj]) ** 2).sum(1))
    G = sp.csr_matrix((np.concatenate([d, d]),
                       (np.concatenate([ii, jj]), np.concatenate([jj, ii]))),
                      shape=(C.shape[0], C.shape[0]))

    root = np.atleast_1d(np.asarray(root_pr_cells))
    if root.size == 1 and 0 <= int(root[0]) < C.shape[0]:
        root_node = int(root[0])
    else:
        root_node = int(np.bincount(cKDTree(C).query(X[root.astype(int)])[1],
                                    minlength=C.shape[0]).argmax())

    dist_nodes = dijkstra(G, indices=root_node, directed=False)
    owner = cKDTree(C).query(X)[1]
    resid = np.linalg.norm(X - C[owner], axis=1)
    return dist_nodes[owner] + resid
