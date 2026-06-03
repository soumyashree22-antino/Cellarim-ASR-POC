"""Biological sanity signals for the inference-only ranker.

* **Motif preservation** — does the candidate carry the family's catalytic
  motif (e.g. GxSxG nucleophile elbow for lipases) and is the Ser-Asp-His
  triad plausibly spaced?
* **Conservation score** — at MSA columns where the family is strongly
  conserved, does the candidate match the consensus residue?

Both are computed directly from sequences and the MSA — no models, no training.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .config import Config
from .io_utils import get_logger, read_fasta

log = get_logger("wp3.motif")


# ── Motif preservation ──────────────────────────────────────────────────────
def motif_score(seq: str, cfg: Config) -> float:
    """Return a [0, 1] score: 1.0 if catalytic motif present, 0.5 partial, 0 absent.

    Lipase-style: GxSxG nucleophile elbow plus a *plausible* downstream
    Asp/Glu and a final His ordered in primary sequence. The triad-distance
    check is purposely loose (just requires forward ordering and a minimum
    sequence gap) — geometric verification belongs to WP4.
    """
    motif = cfg.target.catalytic_motif_regex
    if not motif:
        return 1.0
    m = re.search(motif, seq)
    if not m:
        return 0.0
    score = 0.7  # motif present
    triad = cfg.target.catalytic_triad or []
    if len(triad) >= 2 and triad[0].lower().startswith("ser"):
        ser_pos = m.start() + 2  # 'S' in GxSxG
        # Look downstream for a D/E (Asp/Glu) and then an H (His).
        tail = seq[ser_pos + 10:]
        d_match = re.search(r"[DE]", tail)
        if d_match:
            score += 0.15
            h_match = re.search(r"H", tail[d_match.end():])
            if h_match:
                score += 0.15
    return min(score, 1.0)


def motif_scores(seqs: dict[str, str], cfg: Config) -> pd.Series:
    """Vectorised :func:`motif_score` over a mapping of candidates."""
    s = pd.Series(
        {sid: motif_score(seq, cfg) for sid, seq in seqs.items()},
        name="motif",
    )
    log.info("motif_scores", n=len(s), mean=float(s.mean()), n_zero=int((s == 0).sum()))
    return s


# ── Conservation score ──────────────────────────────────────────────────────
def column_consensus(msa_path) -> tuple[list[str], list[str], list[float]]:
    """Return (consensus_residues, columns, conservation_strength) per MSA column.

    conservation_strength = fraction of non-gap sequences matching the column's
    plurality residue (1.0 = perfectly conserved among present residues).
    """
    msa = list(read_fasta(msa_path).values())
    if not msa:
        return [], [], []
    width = len(msa[0])
    consensus: list[str] = []
    strength: list[float] = []
    for col in range(width):
        residues = [s[col] for s in msa if col < len(s) and s[col] != "-"]
        if not residues:
            consensus.append("-")
            strength.append(0.0)
            continue
        most_common, count = Counter(residues).most_common(1)[0]
        consensus.append(most_common)
        strength.append(count / len(residues))
    return consensus, list(range(width)), strength


def conservation_scores(
    seqs: dict[str, str], msa_path, strong_threshold: float = 0.8
) -> pd.Series:
    """For each candidate sequence, fraction of strongly-conserved positions matched.

    Candidates are assumed to be gap-masked (per WP2 fix) so their length equals
    the count of "real" MSA columns. We re-derive the conserved subset from the
    masked MSA: columns where ≥ ``strong_threshold`` of present residues agree.

    Scoring: among those strong columns, the fraction at which the candidate
    matches the column consensus. ∈ [0, 1].
    """
    consensus, _, strength = column_consensus(msa_path)
    # Gap-mask: keep only columns that themselves are not "-" consensus AND meet threshold.
    keep_idx = [i for i, (c, s) in enumerate(zip(consensus, strength, strict=False))
                if c != "-" and s >= strong_threshold]
    if not keep_idx:
        return pd.Series({sid: 0.0 for sid in seqs}, name="conservation")

    scores = {}
    for sid, seq in seqs.items():
        # Pair conserved-position residue with the candidate residue at the same
        # *ungapped* position. Candidates are length L = #real-columns; we expect
        # them to line up with the real columns 1:1.
        if len(seq) != len(consensus) - consensus.count("-"):
            # length mismatch — pad/truncate gracefully by matching first min().
            pass
        matches = 0
        # Build an iterator over real (non-gap-consensus) columns:
        real_positions = [i for i, c in enumerate(consensus) if c != "-"]
        # Map each strong column to its index within real_positions
        for k_pos, ci in enumerate(real_positions):
            if k_pos >= len(seq):
                break
            if ci in keep_idx and seq[k_pos] == consensus[ci]:
                matches += 1
        scores[sid] = matches / len(keep_idx)
    s = pd.Series(scores, name="conservation")
    log.info("conservation_scores", n=len(s), strong_cols=len(keep_idx),
             mean=float(s.mean()))
    return s


# ── Uncertainty (already produced by WP2) ───────────────────────────────────
def uncertainty_penalty(per_candidate: pd.DataFrame) -> pd.Series:
    """Min-max normalised mean-entropy → [0, 1] penalty.

    Takes the per-candidate uncertainty frame produced by
    ``phylo.consensus_uncertainty_per_candidate`` (columns: candidate_id,
    mean_entropy, mean_pp, ...). Higher entropy → higher penalty.
    """
    if "candidate_id" not in per_candidate.columns:
        raise ValueError("per_candidate must include 'candidate_id'")
    s = per_candidate.set_index("candidate_id")["mean_entropy"]
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index, name="uncertainty")
    return ((s - lo) / (hi - lo)).rename("uncertainty")
