"""WP3/WP5 — Hybrid LLM-assisted candidate ranking and scientific validation.

Integrates LLM-assisted biological reasoning with deterministic scientific
validation and structural ESMFold predictions.
"""

from __future__ import annotations

import pandas as pd

from . import embeddings, feature_table, llm_scoring, deterministic_validation, motif_conservation, phylo, similarity, structure_scoring
from .config import Config, RankingWeights
from .io_utils import get_logger, read_fasta

log = get_logger("wp3.ranking")


# ── Stage 1: Pre-fold Hybrid Ranking (All Candidates) ───────────────────────
def pre_fold_rank(cfg: Config, msa_path=None) -> pd.DataFrame:
    """Rank all candidates using a hybrid LLM and deterministic validation workflow.

    1. Embeds candidates and the anchor set with ESM-2.
    2. Builds a comprehensive structured feature table.
    3. Scores candidates using pluggable LLMs (Gemini, OpenAI, Claude, or fallback).
    4. Validates rankings against deterministic evolutionary priors.
    5. Computes a hybrid 'pre_score' to prioritize candidates.
    6. Clusters sequences for diversity selection.
    """
    cfg.paths.ensure_dirs()
    candidates = read_fasta(cfg.paths.ancestral_fasta)
    anchors = embeddings.build_anchor_set(cfg)

    # 1. Embed and calculate cosine similarities
    cand_emb = embeddings.embed_or_load(candidates, cfg.paths.candidate_embeddings, cfg)
    anch_emb = embeddings.embed_or_load(anchors, cfg.paths.anchor_embeddings, cfg)
    emb_sim = similarity.anchor_similarity(cand_emb, anch_emb, k=cfg.ranking.knn_k)

    # 2. Sequence-based motif and conservation features
    motif = motif_conservation.motif_scores(candidates, cfg)

    msa_path = msa_path or (cfg.paths.alignments_dir / "msa_subset60.fasta")
    if not msa_path.exists():
        msa_path = cfg.paths.msa_fasta
    if msa_path.exists():
        conservation = motif_conservation.conservation_scores(candidates, msa_path)
    else:
        log.warning("conservation_skipped", reason="no MSA found")
        conservation = pd.Series({sid: 0.0 for sid in candidates}, name="conservation")

    # 3. Evolutionary posterior uncertainty
    unc_df = phylo.consensus_uncertainty_per_candidate(cfg)
    uncertainty = motif_conservation.uncertainty_penalty(unc_df)

    # 4. Build Structured Feature Table
    feat_df = feature_table.build_feature_table(
        cfg=cfg,
        emb_sim=emb_sim,
        motif_scores=motif,
        conservation_scores=conservation,
        uncertainty_df=unc_df,
        structure_metrics=None
    )

    # 5. LLM-Assisted Stability Scoring
    prompt_json = feature_table.serialize_feature_table_for_prompt(feat_df, top_n=348)
    llm_df = llm_scoring.score_candidates_with_llm(prompt_json, cfg)

    # 6. Deterministic Validation Layer
    rho, tau, validation_df = deterministic_validation.validate_rankings(feat_df, llm_df, cfg)

    # 7. KMeans Clustering for Clade Diversity
    clusters = similarity.cluster_embeddings(
        cand_emb, n_clusters=cfg.ranking.cluster_n, random_state=cfg.ranking.random_state
    )

    # Combine all signals
    signals = validation_df.join(clusters, how="inner")
    
    # Calculate a balanced hybrid pre_score:
    # 70% LLM Stability score (scaled [0,1]) + 30% Deterministic Prior
    signals["pre_score"] = (
        0.70 * (signals["llm_stability_score"] / 10.0)
        + 0.30 * signals["deterministic_score"]
    )
    
    # Add backward compatibility aliases for notebook print statements
    signals["emb_sim"] = signals["embedding_similarity"]
    signals["motif"] = signals["motif_preservation"]
    signals["conservation"] = signals["conservation_score"]
    signals["uncertainty"] = signals["uncertainty_entropy"]
    
    # Add validation metadata
    signals["validation_rho"] = rho
    signals["validation_tau"] = tau

    signals = signals.sort_values("pre_score", ascending=False)
    
    # Save the signals audit trail
    signals.reset_index().to_csv(cfg.paths.signals_csv, index=False)
    
    log.info("pre_fold_ranked", n=len(signals),
             top_llm_score=float(signals["llm_stability_score"].iloc[0]),
             top_pre_score=float(signals["pre_score"].iloc[0]))
             
    return signals


def candidates_to_fold(signals: pd.DataFrame, cfg: Config) -> list[str]:
    """Select a cluster-diverse top-K set of candidates for structural folding."""
    ranked = signals.sort_values("pre_score", ascending=False).index.tolist()
    return similarity.diversity_select(ranked, signals["cluster"], cfg.ranking.pre_fold_top_k)


# ── Stage 2: Final Ranking (Folded Candidates with Structure) ───────────────
def final_rank(
    signals: pd.DataFrame, structure_metrics: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Run the second pass hybrid scoring including structural fold validation.

    1. Re-builds the feature table, integrating ESMFold pLDDT and Rg compactness.
    2. Re-scores the folded candidates with the LLM, providing structural evidence.
    3. Runs deterministic validation correlation checks.
    4. Computes the final composite prioritized ranking.
    """
    candidates = read_fasta(cfg.paths.ancestral_fasta)
    anchors = embeddings.build_anchor_set(cfg)
    
    cand_emb = embeddings.embed_or_load(candidates, cfg.paths.candidate_embeddings, cfg)
    anch_emb = embeddings.embed_or_load(anchors, cfg.paths.anchor_embeddings, cfg)
    emb_sim = similarity.anchor_similarity(cand_emb, anch_emb, k=cfg.ranking.knn_k)
    
    motif = motif_conservation.motif_scores(candidates, cfg)
    
    msa_path = cfg.paths.alignments_dir / "msa_subset60.fasta"
    if not msa_path.exists():
        msa_path = cfg.paths.msa_fasta
    if msa_path.exists():
        conservation = motif_conservation.conservation_scores(candidates, msa_path)
    else:
        conservation = pd.Series({sid: 0.0 for sid in candidates}, name="conservation")
        
    unc_df = phylo.consensus_uncertainty_per_candidate(cfg)
    
    # 1. Re-build Feature Table incorporating ESMFold metrics
    feat_df = feature_table.build_feature_table(
        cfg=cfg,
        emb_sim=emb_sim,
        motif_scores=motif,
        conservation_scores=conservation,
        uncertainty_df=unc_df,
        structure_metrics=structure_metrics
    )
    
    # Restrict to only the folded subset
    folded_ids = structure_metrics[structure_metrics["kind"] == "candidate"]["candidate_id"].tolist()
    folded_feat = feat_df.loc[folded_ids].copy()
    
    # 2. Final LLM Scoring with Structural Evidence
    prompt_json = feature_table.serialize_feature_table_for_prompt(folded_feat, top_n=len(folded_feat))
    llm_df = llm_scoring.score_candidates_with_llm(prompt_json, cfg)
    
    # 3. Final Deterministic Validation correlation check
    rho, tau, validation_df = deterministic_validation.validate_rankings(folded_feat, llm_df, cfg)
    
    # Join with clusters
    clusters = signals[["cluster"]]
    final = validation_df.join(clusters, how="inner")
    
    # Map ESMFold confidence
    struct = structure_scoring.structural_signals(structure_metrics)
    final = final.join(struct, how="inner")
    
    # 4. Final Hybrid Score calculation:
    # 50% LLM Stability score + 30% ESMFold confidence + 20% Deterministic validation score
    final["final_score"] = (
        0.50 * (final["llm_stability_score"] / 10.0)
        + 0.30 * final["struct_conf"]
        + 0.20 * final["deterministic_score"]
    )
    
    # Add backward compatibility aliases for notebook print statements
    final["emb_sim"] = final["embedding_similarity"]
    final["motif"] = final["motif_preservation"]
    final["conservation"] = final["conservation_score"]
    final["uncertainty"] = final["uncertainty_entropy"]
    
    final["validation_rho"] = rho
    final["validation_tau"] = tau
    
    final = final.sort_values("final_score", ascending=False)
    final["rank"] = range(1, len(final) + 1)
    
    # Export compact rankings and detailed audits
    out_cols = ["rank", "final_score", "llm_stability_score", "struct_conf", "motif_preservation",
                "conservation_score", "uncertainty_entropy", "packing", "cluster"]
    final.index.name = "candidate_id"
    final.reset_index()[["candidate_id"] + out_cols].to_csv(cfg.paths.ranking_csv, index=False)
    final.reset_index().to_csv(cfg.paths.signals_csv, index=False)
    
    log.info("final_ranked",
             n=len(final), top=final.index[0],
             top_score=float(final["final_score"].iloc[0]),
             out=str(cfg.paths.ranking_csv))
             
    return final


# ── Continuous Composite Score Helper (Pure, Testable) ──────────────────────
def composite_score(
    emb_sim: float, struct_conf: float, motif: float, conservation: float,
    uncertainty: float, weights: RankingWeights,
) -> float:
    """Mathematical baseline score helper for local testing and validation."""
    return (
        weights.sim * emb_sim
        + weights.structure * struct_conf
        + weights.motif * motif
        + weights.conservation * conservation
        - weights.uncertainty * uncertainty
    )
