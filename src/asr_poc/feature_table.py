"""Feature table builder for LLM-assisted candidate evaluation.

Compiles biological, evolutionary, and representational features of candidate
sequences into structured tables suitable for LLM analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .io_utils import get_logger, read_fasta

log = get_logger("wp3.feature_table")

# Kyte-Doolittle Hydrophobicity Scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}


def calculate_hydrophobicity(seq: str) -> float:
    """Calculate mean Kyte-Doolittle hydrophobicity for the sequence."""
    seq = seq.upper().replace("-", "")
    if not seq:
        return 0.0
    scores = [KD_SCALE.get(aa, 0.0) for aa in seq]
    return float(np.mean(scores))


def calculate_mutation_distance(seq: str, ref_seq: str) -> float:
    """Fractional mismatch distance between two sequences."""
    seq = seq.upper().replace("-", "")
    ref_seq = ref_seq.upper().replace("-", "")
    if not seq or not ref_seq:
        return 1.0
    mismatches = sum(1 for a, b in zip(seq, ref_seq, strict=False) if a != b)
    mismatches += abs(len(seq) - len(ref_seq))
    return float(mismatches / max(len(seq), len(ref_seq)))


def build_feature_table(
    cfg: Config,
    emb_sim: pd.Series,
    motif_scores: pd.Series,
    conservation_scores: pd.Series,
    uncertainty_df: pd.DataFrame,
    structure_metrics: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compile structured biological features for all candidates.

    Features include:
        * embedding_similarity: cosine proximity to extant anchors in ESM-2 space.
        * motif_preservation: active-site nucleophile elbow presence [0, 1].
        * conservation: family alignment conservation at invariant columns.
        * entropy: mean site-wise phylogenetic reconstruction uncertainty.
        * hydrophobicity: continuous mean Kyte-Doolittle scale score.
        * evolutionary_distance: mismatch distance vs the reference lipase.
        * active_site_preservation: binary active triad intactness.
        * fold_confidence: ESMFold pLDDT/100 (when folded post-WP4, else 0.0).
    """
    candidates = read_fasta(cfg.paths.ancestral_fasta)
    ref_seq = ""
    if cfg.paths.benchmark_fasta.exists():
        benchmarks = read_fasta(cfg.paths.benchmark_fasta)
        if benchmarks:
            ref_seq = list(benchmarks.values())[0]

    rows = []
    # Map posterior uncertainty from iqtree
    unc_map = {}
    if "node" in uncertainty_df.columns:
        for _, row in uncertainty_df.iterrows():
            node_id = str(row["node"])
            unc_map[node_id] = {
                "mean_entropy": float(row.get("mean_entropy", 0.0)),
                "min_pp": float(row.get("min_pp", 1.0)),
                "mean_pp": float(row.get("mean_pp", 1.0))
            }

    # Map structure metrics if post-fold
    struct_map = {}
    if structure_metrics is not None and "candidate_id" in structure_metrics.columns:
        for _, row in structure_metrics.iterrows():
            cid = row["candidate_id"]
            plddt = float(row.get("mean_plddt", 0.0))
            if plddt <= 1.5:  # standard scale check
                plddt *= 100.0
            struct_map[cid] = {
                "fold_confidence": float(plddt / 100.0),
                "packing_density": float(row.get("contact_density", 0.0)),
                "compactness_rg": float(row.get("radius_of_gyration", 0.0))
            }

    for cid, seq in candidates.items():
        # Extrapolate internal node key (e.g. ALT_Node52_alt2 -> Node52 -> 52)
        node_part = cid.removeprefix("ALT_").removeprefix("ANC_").split("_")[0]
        node_id = node_part.replace("Node", "")
        
        unc = unc_map.get(node_id, {"mean_entropy": 0.0, "min_pp": 1.0, "mean_pp": 1.0})
        struct = struct_map.get(cid, {"fold_confidence": 0.0, "packing_density": 0.0, "compactness_rg": 0.0})

        row = {
            "candidate_id": cid,
            "sequence_length": len(seq),
            "embedding_similarity": float(emb_sim.get(cid, 0.0)),
            "motif_preservation": float(motif_scores.get(cid, 0.0)),
            "conservation_score": float(conservation_scores.get(cid, 0.0)),
            "uncertainty_entropy": float(unc["mean_entropy"]),
            "reconstruction_confidence": float(unc["mean_pp"]),
            "hydrophobicity": calculate_hydrophobicity(seq),
            "evolutionary_distance": calculate_mutation_distance(seq, ref_seq),
            "active_site_preservation": 1.0 if motif_scores.get(cid, 0.0) >= 1.0 else 0.0,
            "fold_confidence": struct["fold_confidence"],
            "packing_density": struct["packing_density"],
            "compactness_rg": struct["compactness_rg"]
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("candidate_id")
    log.info("feature_table_built", candidates=len(df), features=len(df.columns))
    return df


def serialize_feature_table_for_prompt(df: pd.DataFrame, top_n: int = 15) -> str:
    """Format feature table rows into a structured JSON string for LLM input."""
    # Select subset of candidates to fit in prompt context comfortably
    subset = df.head(top_n).copy()
    records = []
    for cid, row in subset.iterrows():
        record = {
            "candidate_id": cid,
            "sequence_length": int(row["sequence_length"]),
            "embedding_similarity": round(float(row["embedding_similarity"]), 3),
            "motif_preservation": round(float(row["motif_preservation"]), 3),
            "conservation_score": round(float(row["conservation_score"]), 3),
            "uncertainty_entropy": round(float(row["uncertainty_entropy"]), 3),
            "hydrophobicity": round(float(row["hydrophobicity"]), 3),
            "evolutionary_distance": round(float(row["evolutionary_distance"]), 3),
            "active_site_preservation": int(row["active_site_preservation"]),
            "fold_confidence": round(float(row["fold_confidence"]), 3),
            "packing_density": round(float(row["packing_density"]), 3)
        }
        records.append(record)
    return json.dumps(records, indent=2)
