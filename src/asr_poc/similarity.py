"""Embedding-space similarity, kNN, and diversity clustering — no training.

Operates purely on the ESM-2 vectors produced by :mod:`embeddings`. The two
public entry points used by the ranking pipeline are :func:`anchor_similarity`
(per-candidate score in [0, 1]) and :func:`diversity_select` (cluster-balanced
top-K selection for the fold step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .io_utils import get_logger

log = get_logger("wp3.similarity")


# ── Core similarity ──────────────────────────────────────────────────────────
def cosine_matrix(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity matrix between every row of ``a`` and every row of ``b``.

    Returns a DataFrame indexed by ``a``'s ids with columns from ``b``'s ids.
    """
    sim = cosine_similarity(a.values, b.values)
    return pd.DataFrame(sim, index=a.index, columns=b.index)


def anchor_similarity(
    candidates: pd.DataFrame, anchors: pd.DataFrame, k: int
) -> pd.Series:
    """Per-candidate mean cosine similarity to the k nearest anchors, rescaled.

    cosine ∈ [-1, 1] → rescaled to [0, 1] via (x + 1) / 2 so the composite-score
    weights operate on a common scale. Top-k (rather than mean over all anchors)
    keeps the score robust to anchor outliers and family diversity.
    """
    sim = cosine_matrix(candidates, anchors)
    # For each row, take the mean of the top-k values.
    topk_mean = np.partition(sim.values, -k, axis=1)[:, -k:].mean(axis=1)
    scaled = (topk_mean + 1.0) / 2.0
    s = pd.Series(scaled, index=candidates.index, name="emb_sim")
    log.info("anchor_similarity", n=len(s), k=k,
             min=float(s.min()), max=float(s.max()), mean=float(s.mean()))
    return s


def nearest_anchors(
    candidates: pd.DataFrame, anchors: pd.DataFrame, k: int = 3
) -> pd.DataFrame:
    """Return the k nearest anchor ids per candidate (long format).

    Useful for the report: "ALT_NodeX is closest to UniProt:O00748".
    """
    sim = cosine_matrix(candidates, anchors)
    rows = []
    for cid, row in sim.iterrows():
        top = row.nlargest(k)
        for rank, (anchor_id, score) in enumerate(top.items(), start=1):
            rows.append({"candidate_id": cid, "rank": rank,
                         "anchor_id": anchor_id, "cosine": float(score)})
    return pd.DataFrame(rows)


# ── Clustering / diversity ──────────────────────────────────────────────────
def cluster_embeddings(
    embeddings: pd.DataFrame, n_clusters: int, random_state: int = 42
) -> pd.Series:
    """KMeans cluster assignments for each embedding row."""
    n = min(n_clusters, len(embeddings))
    if n < 2:
        return pd.Series(np.zeros(len(embeddings), dtype=int), index=embeddings.index,
                         name="cluster")
    km = KMeans(n_clusters=n, n_init=10, random_state=random_state)
    labels = km.fit_predict(embeddings.values)
    log.info("clustered", n_clusters=n, n=len(embeddings))
    return pd.Series(labels, index=embeddings.index, name="cluster")


def diversity_select(
    ranked_ids: list[str], clusters: pd.Series, k: int
) -> list[str]:
    """Pick ``k`` ids from ``ranked_ids`` favouring cluster coverage.

    Iterate the ranking; take the highest-ranked unseen cluster first until
    every cluster has one representative, then fill from the top of the
    remaining ranking. This stops the fold budget being eaten by near-duplicate
    alternates of one ancestor node.
    """
    seen_clusters: set = set()
    picked: list[str] = []
    leftover: list[str] = []
    for cid in ranked_ids:
        c = int(clusters.get(cid, -1))
        if c not in seen_clusters and len(picked) < k:
            picked.append(cid)
            seen_clusters.add(c)
        else:
            leftover.append(cid)
    for cid in leftover:
        if len(picked) >= k:
            break
        picked.append(cid)
    return picked[:k]


# ── Composition / quick helpers ──────────────────────────────────────────────
def normalize_to_unit(s: pd.Series) -> pd.Series:
    """Min-max scale a Series into [0, 1]. Constant series → all zeros."""
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index, name=s.name)
    return ((s - lo) / (hi - lo)).rename(s.name)
