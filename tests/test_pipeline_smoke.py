"""End-to-end wiring smoke test for the inference-only architecture.

Runs WP1 -> WP2 (from a synthetic state file) -> WP3 (embed + similarity +
motif + conservation + composite ranking) -> WP4 (fallback structure) -> WP5
(templated report). All external tools/networks are bypassed: binaries are
skipped (synthetic state file feeds candidate generation), embeddings use the
seeded fallback, folding uses the placeholder backbone, and the report is the
deterministic local template. Asserts each stage's declared artifacts.
"""

from __future__ import annotations

import pandas as pd

from asr_poc import (
    embeddings,
    motif_conservation,
    phylo,
    ranking,
    report,
    retrieval,
    similarity,
    structure,
)
from asr_poc.config import RankingWeights


def test_wp1_dataset_build(cfg, raw_family):
    summary = retrieval.build_dataset(cfg, df_raw=raw_family)
    assert summary["curated"] >= 10
    assert summary["curated"] < summary["raw"]
    assert cfg.paths.curated_fasta.exists()
    assert cfg.paths.metadata_csv.exists()
    assert cfg.paths.benchmark_fasta.exists()

    meta = pd.read_csv(cfg.paths.metadata_csv)
    assert {"id", "organism", "length", "family"}.issubset(meta.columns)
    assert "NOMOTIF" not in set(meta["id"])


def test_wp2_candidate_pool(cfg, synthetic_state_file):
    summary = phylo.build_candidate_pool(cfg, state_file=synthetic_state_file)
    assert summary["ancestors"] == 2
    assert summary["candidates"] >= summary["ancestors"]
    assert cfg.paths.ancestral_fasta.exists()
    unc = pd.read_csv(cfg.paths.uncertainty_csv)
    assert {"node", "site", "max_pp", "entropy"}.issubset(unc.columns)


# ── Inference-only ranking unit pieces ───────────────────────────────────────
def test_embeddings_fallback_shape(cfg):
    cfg.embeddings.provider = "fallback"
    df = embeddings.embed_sequences({"A": "MKTLG", "B": "MKTAG"}, cfg)
    assert list(df.index) == ["A", "B"]
    assert df.shape[1] == 64  # default fallback dim


def test_similarity_anchor_score(cfg):
    cfg.embeddings.provider = "fallback"
    cands = embeddings.embed_sequences({"c1": "MKTLG", "c2": "MAAAA"}, cfg)
    anch = embeddings.embed_sequences({"a1": "MKTLG", "a2": "MKTAG"}, cfg)
    s = similarity.anchor_similarity(cands, anch, k=2)
    assert set(s.index) == {"c1", "c2"}
    assert (s >= 0).all() and (s <= 1).all()


def test_motif_score_bounds(cfg):
    assert motif_conservation.motif_score("AAAGMSMGAAA", cfg) > 0
    assert motif_conservation.motif_score("AAAAAA", cfg) == 0.0


def test_composite_score_pure():
    w = RankingWeights()
    s = ranking.composite_score(
        emb_sim=0.9, struct_conf=0.8, motif=1.0,
        conservation=0.5, uncertainty=0.1, weights=w,
    )
    expected = 0.35 * 0.9 + 0.30 * 0.8 + 0.15 * 1.0 + 0.10 * 0.5 - 0.10 * 0.1
    assert abs(s - expected) < 1e-9


# ── End-to-end (uses fallback providers, no network) ─────────────────────────
def test_full_pipeline_end_to_end(cfg, raw_family, synthetic_state_file):
    cfg.embeddings.provider = "fallback"
    cfg.structure.folder_provider = "local"   # use the placeholder backbone
    # Cap signals/folds to keep the test tiny.
    cfg.ranking.pre_fold_top_k = 3
    cfg.ranking.final_top_n = 2

    retrieval.build_dataset(cfg, df_raw=raw_family)
    phylo.build_candidate_pool(cfg, state_file=synthetic_state_file)

    signals = ranking.pre_fold_rank(cfg)
    for col in ("emb_sim", "motif", "conservation", "uncertainty",
                "cluster", "pre_score"):
        assert col in signals.columns
    assert signals["pre_score"].is_monotonic_decreasing

    fold_ids = ranking.candidates_to_fold(signals, cfg)
    assert 1 <= len(fold_ids) <= 3

    # Hand-roll a minimal structure_metrics frame using the fallback folder.
    from asr_poc.io_utils import read_fasta
    seqs = read_fasta(cfg.paths.ancestral_fasta)
    rows = []
    for sid in fold_ids:
        pdb = structure.predict_structure(sid, seqs[sid], cfg)
        m = structure.structure_metrics(pdb)
        rows.append({"candidate_id": sid, "kind": "candidate", "pdb": str(pdb), **m})
    # Synthesize a benchmark row so structure_scoring.packing has a denominator.
    rows.append({"candidate_id": "BENCH_X", "kind": "benchmark", "pdb": "",
                 "radius_of_gyration": 20.0, "mean_plddt": 90.0,
                 "contact_density": 5.0})
    metrics = pd.DataFrame(rows)

    final = ranking.final_rank(signals, metrics, cfg)
    assert (cfg.paths.ranking_csv).exists()
    assert (cfg.paths.signals_csv).exists()
    assert "final_score" in final.columns

    report_path = report.write_report(cfg, final)
    assert report_path.endswith("scientific_report.md")
    md = open(report_path).read()
    assert "Top" in md and "candidate" in md.lower()
