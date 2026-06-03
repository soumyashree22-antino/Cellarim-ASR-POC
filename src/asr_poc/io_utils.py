"""IO, logging, and reproducibility helpers shared across all work packages.

Keeps FASTA/CSV/PDB read-write, deterministic seeding, content hashing, and
structured logging in one place so the notebooks stay thin and consistent.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

import structlog

# ── Logging ──────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ]
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger tagged with the stage/module name."""
    return structlog.get_logger().bind(stage=name)


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if present) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy always present in this project, guard anyway
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def sequence_hash(seq: str) -> str:
    """Stable short hash of a sequence — used to cache embeddings/structures."""
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


# ── FASTA IO (no hard Biopython dependency for the simple paths) ─────────────
def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into an ``{id: sequence}`` dict (id = first token)."""
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        records[header] = "".join(chunks)
    return records


def write_fasta(records: Mapping[str, str], path: str | Path, width: int = 60) -> Path:
    """Write an ``{id: sequence}`` mapping to FASTA, wrapping at ``width``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for seq_id, seq in records.items():
            fh.write(f">{seq_id}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
    return path


def count_fasta(path: str | Path) -> int:
    """Number of records in a FASTA file."""
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def iter_fasta(path: str | Path) -> Iterable[tuple[str, str]]:
    """Yield ``(id, sequence)`` pairs lazily."""
    yield from read_fasta(path).items()
