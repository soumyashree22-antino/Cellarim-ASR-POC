"""WP5 — Scientific report for the top candidates (deterministic, local only).

Builds a structured *evidence packet* per candidate (numbers from the ranker
and structural metrics, nearest-anchor identities) and renders it into a
markdown report. Pure template — no LLM, no API keys, no network. All numbers
are computed locally by the rest of the pipeline; this module only formats them.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from .config import Config
from .io_utils import get_logger, read_fasta
from .similarity import nearest_anchors

log = get_logger("wp5.report")


def evidence_packet(
    candidate_id: str,
    final: pd.DataFrame,
    nearest: pd.DataFrame,
    candidates: dict[str, str],
) -> dict:
    """Compact dict of the per-candidate signals + nearest anchors."""
    row = final.loc[candidate_id]
    nbrs = nearest[nearest["candidate_id"] == candidate_id].sort_values("rank")
    return {
        "candidate_id": candidate_id,
        "sequence_length": len(candidates.get(candidate_id, "")),
        "rank": int(row.get("rank", 0)),
        "final_score": round(float(row.get("final_score", row.get("pre_score", 0.0))), 4),
        "signals": {
            "emb_sim":      round(float(row.get("emb_sim", 0.0)), 4),
            "struct_conf":  round(float(row.get("struct_conf", 0.0)), 4),
            "motif":        round(float(row.get("motif", 0.0)), 4),
            "conservation": round(float(row.get("conservation", 0.0)), 4),
            "uncertainty":  round(float(row.get("uncertainty", 0.0)), 4),
            "packing":      round(float(row.get("packing", 0.0)), 4),
        },
        "cluster": int(row.get("cluster", 0)),
        "nearest_anchors": [
            {"id": r["anchor_id"], "cosine": round(float(r["cosine"]), 4)}
            for _, r in nbrs.iterrows()
        ],
    }


def write_report(cfg: Config, final: pd.DataFrame) -> str:
    """Produce the scientific report markdown and return its path as a string."""
    cfg.paths.ensure_dirs()
    candidates = read_fasta(cfg.paths.ancestral_fasta)

    # Nearest anchors come from the cached ESM-2 embeddings (cheap re-derivation).
    from . import embeddings

    cand_emb = embeddings.load_embeddings(cfg.paths.candidate_embeddings)
    anch_emb = embeddings.load_embeddings(cfg.paths.anchor_embeddings)
    nearest = nearest_anchors(cand_emb, anch_emb, k=3)

    top = final.head(cfg.ranking.final_top_n).reset_index()
    packets = [
        evidence_packet(row["candidate_id"], final, nearest, candidates)
        for _, row in top.iterrows()
    ]

    header = (
        f"# Scientific Report — Top {len(packets)} Candidates "
        f"({cfg.target.name})\n\n"
        f"Generated: {datetime.now(UTC).isoformat()}\n\n"
        "All numbers are computed locally (ESM-2 embeddings + ESMFold + motif/"
        "conservation checks). See `reports/candidate_signals.csv` for the full "
        "audit trail.\n\n"
    )
    cfg.paths.report_md.write_text(header + _render(packets))
    log.info("report_written", out=str(cfg.paths.report_md), n=len(packets))
    return str(cfg.paths.report_md)


def _render(packets: list[dict]) -> str:
    """Render the evidence packets as markdown sections."""
    blocks = []
    for p in packets:
        s = p["signals"]
        ranked_anchors = ", ".join(
            f"{n['id']} (cos={n['cosine']})" for n in p["nearest_anchors"]
        )
        verdict = []
        if s["struct_conf"] >= 0.7:
            verdict.append("ESMFold is confident in the predicted fold")
        elif s["struct_conf"] >= 0.5:
            verdict.append("ESMFold confidence is moderate")
        else:
            verdict.append("ESMFold confidence is low — verify fold integrity")
        if s["motif"] >= 0.9:
            verdict.append("the catalytic motif is intact")
        elif s["motif"] >= 0.5:
            verdict.append("the catalytic motif is partially intact")
        else:
            verdict.append("the catalytic motif is missing")
        if s["emb_sim"] >= 0.9:
            verdict.append("the sequence sits firmly on the family manifold (ESM-2)")
        elif s["emb_sim"] >= 0.7:
            verdict.append("the sequence sits well on the family manifold (ESM-2)")

        blocks.append(
            f"## #{p['rank']} — `{p['candidate_id']}`\n\n"
            f"**Final score:** {p['final_score']}  ·  length {p['sequence_length']} aa  "
            f"·  cluster {p['cluster']}\n\n"
            "| Signal | Value |\n|---|---|\n"
            f"| Anchor cosine (emb_sim) | {s['emb_sim']} |\n"
            f"| ESMFold pLDDT/100 (struct_conf) | {s['struct_conf']} |\n"
            f"| Catalytic motif | {s['motif']} |\n"
            f"| Conservation match | {s['conservation']} |\n"
            f"| ASR uncertainty penalty | {s['uncertainty']} |\n"
            f"| Packing vs benchmarks | {s['packing']} |\n\n"
            f"**Nearest extant lipases:** {ranked_anchors}\n\n"
            "**Interpretation:** " + "; ".join(verdict) + ".\n"
        )
    return "\n".join(blocks)
