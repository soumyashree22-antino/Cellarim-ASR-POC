"""Shared pytest fixtures.

Provides a tiny synthetic "lipase" family and a synthetic IQ-TREE ``.state``
file so the full pipeline can be exercised end-to-end with no external binaries
or network — guarding the wiring before a real run.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from asr_poc.config import load_config

AA = "ACDEFGHIKLMNPQRSTVWY"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_seq(rng: random.Random, length: int, motif_at: int) -> str:
    seq = [rng.choice(AA) for _ in range(length)]
    # Embed the GxSxG lipase motif so the catalytic-motif filter keeps it.
    motif = ["G", rng.choice(AA), "S", rng.choice(AA), "G"]
    seq[motif_at : motif_at + 5] = motif
    return "".join(seq)


@pytest.fixture
def cfg(tmp_path):
    """A Config pointing all outputs at a temp dir (no repo pollution)."""
    c = load_config(PROJECT_ROOT / "config" / "target.yaml")
    c.paths.root = tmp_path  # redirect artifacts into the test sandbox
    c.paths.ensure_dirs()
    return c


@pytest.fixture
def raw_family() -> pd.DataFrame:
    """A 15-sequence synthetic homolog set mimicking a UniProt pull."""
    rng = random.Random(42)
    rows = []
    for i in range(15):
        length = rng.randint(200, 400)
        seq = _make_seq(rng, length, motif_at=rng.randint(80, 120))
        rows.append({
            "id": f"SEQ{i:02d}",
            "organism": f"Test organism {i}",
            "length": length,
            "fragment": "",
            "ec_number": "3.1.1.3",
            "protein_name": "Triacylglycerol lipase",
            "annotation_score": rng.randint(1, 5),
            "sequence": seq,
        })
    # Add a couple of records that should be filtered out.
    rows.append({"id": "FRAG01", "organism": "x", "length": 90, "fragment": "fragment",
                 "ec_number": "", "protein_name": "lipase (Fragment)",
                 "annotation_score": 1, "sequence": "GMSMG" + "A" * 85})
    rows.append({"id": "NOMOTIF", "organism": "y", "length": 300, "fragment": "",
                 "ec_number": "", "protein_name": "unrelated", "annotation_score": 1,
                 "sequence": "A" * 300})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_state_file(tmp_path) -> Path:
    """A minimal IQ-TREE-style .state file for two internal nodes, 12 sites."""
    rng = random.Random(7)
    n_sites = 12
    header = ["Node", "Site", "State"] + [f"p_{a}" for a in AA]
    lines = ["# Synthetic state file for tests", "\t".join(header)]
    for node in ("Node1", "Node2"):
        for site in range(1, n_sites + 1):
            probs = [rng.random() for _ in AA]
            total = sum(probs)
            probs = [p / total for p in probs]
            best = max(range(len(AA)), key=lambda k: probs[k])
            state = AA[best]
            row = [node, str(site), state] + [f"{p:.4f}" for p in probs]
            lines.append("\t".join(row))
    path = tmp_path / "asr.state"
    path.write_text("\n".join(lines) + "\n")
    return path
