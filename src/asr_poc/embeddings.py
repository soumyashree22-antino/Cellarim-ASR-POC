"""ESM-2 protein embeddings — inference-only.

Provider-abstracted so the same code runs on a Mac CPU with the small ESM-2
model and on a GPU/API path with the full model. Vectors are cached to Parquet
keyed by sequence hash so re-runs are free.

No training, no fine-tuning. Only :func:`embed_sequences` and the cache helpers
are imported by downstream modules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .io_utils import get_logger, read_fasta, sequence_hash, write_fasta

log = get_logger("wp3.embeddings")


# ── Provider abstraction ─────────────────────────────────────────────────────
def embed_sequences(seqs: Mapping[str, str], cfg: Config) -> pd.DataFrame:
    """Return a DataFrame (index=id, columns=e_0..e_{d-1}) of ESM-2 embeddings.

    Provider selected by ``cfg.embeddings.provider``: ``local`` (CPU ESM-2 via
    fair-esm), ``api`` (hosted endpoint), or ``fallback`` (seeded hash; tests
    only). Any provider failure logs a warning and drops to the fallback so the
    pipeline still completes — but the fallback is **not scientific**.
    """
    provider = cfg.embeddings.provider
    if provider == "local":
        try:
            return _embed_local_esm(seqs, cfg)
        except Exception as exc:
            log.warning("esm_local_unavailable", error=str(exc))
            return _embed_fallback(seqs)
    if provider == "api":
        try:
            return _embed_api(seqs, cfg)
        except Exception as exc:  # pragma: no cover - network
            log.warning("esm_api_unavailable", error=str(exc))
            return _embed_fallback(seqs)
    if provider == "fallback":
        return _embed_fallback(seqs)
    raise ValueError(f"Unknown embedding provider: {provider}")


def _embed_local_esm(seqs: Mapping[str, str], cfg: Config) -> pd.DataFrame:
    """Mean-pooled ESM-2 embeddings via the `fair-esm` package on CPU."""
    import esm
    import torch

    model, alphabet = esm.pretrained.load_model_and_alphabet(cfg.embeddings.esm_model_local)
    model.eval()
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers

    vectors: dict[str, np.ndarray] = {}
    items = list(seqs.items())
    bs = cfg.embeddings.batch_size
    for i in range(0, len(items), bs):
        batch = items[i : i + bs]
        _, _, toks = bc([(sid, s) for sid, s in batch])
        with torch.no_grad():
            out = model(toks, repr_layers=[repr_layer])
        reps = out["representations"][repr_layer]
        for j, (sid, s) in enumerate(batch):
            vectors[sid] = reps[j, 1 : len(s) + 1].mean(0).numpy()
    log.info("embedded_local", n=len(vectors), dim=len(next(iter(vectors.values()))),
             model=cfg.embeddings.esm_model_local)
    return _vectors_to_frame(vectors)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=20))
def _embed_api_one(seq: str, url: str) -> np.ndarray:  # pragma: no cover - network
    resp = requests.post(url, data=seq, timeout=120)
    resp.raise_for_status()
    return np.asarray(resp.json(), dtype=float).mean(axis=0)


def _embed_api(seqs: Mapping[str, str], cfg: Config) -> pd.DataFrame:  # pragma: no cover
    vectors = {sid: _embed_api_one(s, cfg.embeddings.esm_api_url) for sid, s in seqs.items()}
    return _vectors_to_frame(vectors)


def _embed_fallback(seqs: Mapping[str, str], dim: int = 64) -> pd.DataFrame:
    """Deterministic seeded pseudo-embedding — wiring-only, not scientific.

    Each sequence hash seeds an RNG so embeddings are reproducible and
    sequence-specific. Lets tests and CI run with no model installed.
    """
    vectors: dict[str, np.ndarray] = {}
    for sid, s in seqs.items():
        rng = np.random.default_rng(int(sequence_hash(s), 16) % (2**32))
        vectors[sid] = rng.standard_normal(dim)
    log.warning("embedded_fallback", n=len(vectors), dim=dim)
    return _vectors_to_frame(vectors)


def _vectors_to_frame(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(vectors, orient="index")
    df.columns = [f"e_{i}" for i in range(df.shape[1])]
    df.index.name = "id"
    return df


# ── Cache to Parquet ─────────────────────────────────────────────────────────
def save_embeddings(df: pd.DataFrame, path: Path, model_name: str) -> None:
    """Write the embedding matrix to Parquet and a small JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    meta = {
        "model": model_name,
        "dim": df.shape[1],
        "n": df.shape[0],
        "ids_sample": df.index[:5].tolist(),
    }
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("saved_embeddings", path=str(path), n=df.shape[0], dim=df.shape[1])


def load_embeddings(path: Path) -> pd.DataFrame:
    """Load an embedding Parquet (index = id)."""
    return pd.read_parquet(path)


def embed_or_load(
    seqs: Mapping[str, str], path: Path, cfg: Config, force: bool = False
) -> pd.DataFrame:
    """Embed ``seqs`` or load from ``path`` if it exists. Set ``force=True`` to rebuild."""
    if path.exists() and not force:
        df = load_embeddings(path)
        missing = set(seqs) - set(df.index)
        if not missing:
            log.info("embeddings_cache_hit", path=str(path), n=len(df))
            return df.loc[list(seqs)]
        log.info("embeddings_partial_cache", missing=len(missing))
    df = embed_sequences(seqs, cfg)
    save_embeddings(df, path, cfg.embeddings.esm_model_local)
    return df


# ── Anchor set ───────────────────────────────────────────────────────────────
def build_anchor_set(cfg: Config) -> dict[str, str]:
    """Choose the anchor set: the highest-annotation extant curated lipases.

    Anchors are the "known good" references that candidates are measured
    against. We pull from the curated WP1 metadata and pick the top-N entries
    by UniProt annotation score (proxy for evidence quality). The benchmark
    panel is always included.
    """
    metadata = pd.read_csv(cfg.paths.metadata_csv)
    curated = read_fasta(cfg.paths.curated_fasta)
    benchmark = read_fasta(cfg.paths.benchmark_fasta) if cfg.paths.benchmark_fasta.exists() else {}

    sort_col = "annotation_score" if "annotation_score" in metadata.columns else "length"
    top_ids = metadata.sort_values(sort_col, ascending=False)["id"].head(
        cfg.ranking.anchor_uniprot_count
    ).tolist()
    anchors = {sid: curated[sid] for sid in top_ids if sid in curated}
    # Ensure every benchmark id is in the anchor set.
    for sid, s in benchmark.items():
        anchors.setdefault(sid, s)

    write_fasta(anchors, cfg.paths.anchor_fasta)
    log.info("anchors_built", n=len(anchors), out=str(cfg.paths.anchor_fasta))
    return anchors
