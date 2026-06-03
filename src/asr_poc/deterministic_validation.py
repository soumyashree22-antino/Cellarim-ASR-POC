"""Deterministic validation and grounding layer.

Computes mathematically grounded baseline scores, checks rank correlation between
LLM prioritizations and strict evolutionary priors, and flags outliers.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr, kendalltau

from .config import Config
from .io_utils import get_logger

log = get_logger("wp3.validation")


def calculate_deterministic_score(row: pd.Series, cfg: Config) -> float:
    """Standard, rule-based continuous scoring formula using target.yaml weights."""
    w = cfg.ranking.weights
    
    # Positive weights
    score = (
        w.sim * row.get("embedding_similarity", 0.0)
        + w.motif * row.get("motif_preservation", 0.0)
        + w.conservation * row.get("conservation_score", 0.0)
    )
    
    # Subtract uncertainty
    score -= w.uncertainty * row.get("uncertainty_entropy", 0.0)
    
    # Add fold confidence if structure is already folded
    if "fold_confidence" in row and row["fold_confidence"] > 0.0:
        score += w.structure * row["fold_confidence"]
        
    return float(score)


def validate_rankings(
    feature_table: pd.DataFrame,
    llm_scores: pd.DataFrame,
    cfg: Config
) -> tuple[float, float, pd.DataFrame]:
    """Validate rank consistency between LLM scores and evolutionary priors.

    Calculates:
        * Spearman's Rho: rank correlation coefficient [-1, 1].
        * Kendall's Tau: rank correlation coefficient [-1, 1].
        * Outliers: candidates showing > 3.0 points delta (out of 10) between
          the LLM score and the rescaled deterministic prior.

    Returns (spearman_rho, kendall_tau, validation_dataframe).
    """
    df = feature_table.copy()
    
    # Compute deterministic scores
    df["deterministic_score"] = df.apply(lambda r: calculate_deterministic_score(r, cfg), axis=1)
    
    # Rescale deterministic score to a [0, 10] scale to match the LLM scale
    det_min = df["deterministic_score"].min()
    det_max = df["deterministic_score"].max()
    if det_max - det_min > 1e-9:
        df["deterministic_score_10"] = 10.0 * (df["deterministic_score"] - det_min) / (det_max - det_min)
    else:
        df["deterministic_score_10"] = 5.0  # default constant
        
    # Join with LLM scores
    df = df.join(llm_scores, how="inner")
    
    # Spearman rank correlation
    rho, _ = spearmanr(df["deterministic_score_10"], df["llm_stability_score"])
    rho = float(rho) if not pd.isna(rho) else 1.0
    
    # Kendall's Tau correlation
    tau, _ = kendalltau(df["deterministic_score_10"], df["llm_stability_score"])
    tau = float(tau) if not pd.isna(tau) else 1.0
    
    # Calculate score delta and flag outliers (> 3.0 points out of 10)
    df["score_delta"] = (df["llm_stability_score"] - df["deterministic_score_10"]).abs()
    df["is_outlier"] = df["score_delta"] > 3.0
    
    outliers_count = int(df["is_outlier"].sum())
    log.info("validation_complete", 
             spearman_rho=round(rho, 3), 
             kendall_tau=round(tau, 3), 
             outliers=outliers_count)
             
    if outliers_count > 0:
        log.warning("ranking_outliers_detected", 
                    count=outliers_count, 
                    outlier_ids=list(df[df["is_outlier"]].index))
        
    return rho, tau, df
