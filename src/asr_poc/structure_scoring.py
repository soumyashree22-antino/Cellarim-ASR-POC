"""Map ESMFold / AlphaFold structural metrics into [0, 1] ranking signals.

Pure post-processing of the per-candidate metrics DataFrame produced by
``structure.analyze_candidates`` — no folding or training happens here.
"""

from __future__ import annotations

import pandas as pd

from .io_utils import get_logger

log = get_logger("wp3.structure_score")


def plddt_score(metrics: pd.DataFrame) -> pd.Series:
    """Mean pLDDT / 100 — the structural-confidence component, ∈ [0, 1]."""
    s = (metrics.set_index("candidate_id")["mean_plddt"].astype(float) / 100.0).clip(0, 1)
    return s.rename("struct_conf")


def packing_score(metrics: pd.DataFrame) -> pd.Series:
    """Per-candidate contact density relative to the benchmark mean, clipped to [0, 1].

    Useful diagnostic / optional ranking signal. Candidates whose packing is
    much weaker than benchmarks lose score; ones matching or exceeding cap at 1.
    """
    bench_mean = metrics.loc[metrics["kind"] == "benchmark", "contact_density"].mean()
    if pd.isna(bench_mean) or bench_mean <= 0:
        return pd.Series(dtype=float, name="packing")
    cand = metrics.loc[metrics["kind"] == "candidate"].set_index("candidate_id")
    s = (cand["contact_density"] / bench_mean).clip(0, 1)
    return s.rename("packing")


def structural_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """Bundle the structural signals for ranking, indexed by candidate_id."""
    cand_only = metrics[metrics["kind"] == "candidate"].copy()
    pl = plddt_score(cand_only)
    pk = packing_score(metrics).reindex(pl.index).fillna(0.0)
    out = pd.concat([pl, pk], axis=1)
    out.index.name = "candidate_id"
    log.info("structural_signals", n=len(out),
             mean_struct_conf=float(out["struct_conf"].mean()),
             mean_packing=float(out["packing"].mean()))
    return out
