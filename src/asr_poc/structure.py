"""WP4 — Structural modeling & mechanistic analysis.

Predict 3D structures (ESMFold API by default), compute structure-derived metrics
(radius of gyration, secondary-structure content, mean pLDDT, SASA-based packing
proxy), compare ancestral/AI candidates against the extant benchmark, and emit
mutation hypotheses. Structural-confidence features are fed back to WP3.

Folding is provider-abstracted (:func:`predict_structure`) with a deterministic
no-network fallback so the pipeline and tests run without API access. Real runs
use the ESMFold API for CPU-only machines (no local GPU required).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .io_utils import get_logger, read_fasta, sequence_hash

log = get_logger("wp4.structure")

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction"


# ── Structure prediction (provider-abstracted) ───────────────────────────────
def predict_structure(seq_id: str, seq: str, cfg: Config) -> Path:
    """Return a path to the predicted PDB for ``seq``; cached by sequence hash.

    Providers (``structure.folder_provider``):
        * ``api``       — ESMFold REST endpoint.
        * ``alphafold`` — AlphaFold DB lookup for benchmark UniProt IDs, with
          ColabFold local fold for ancestral candidates.
        * ``local``     — uses the deterministic fallback (tests / wiring only).
    Any failure falls back to the placeholder backbone so the pipeline always
    completes; a warning is logged so it's visible.
    """
    cfg.paths.structures_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cfg.paths.structures_dir / f"{seq_id}_{sequence_hash(seq)}.pdb"
    if pdb_path.exists():
        return pdb_path

    provider = cfg.structure.folder_provider
    pdb_text: str | None = None
    if provider == "api":
        try:
            pdb_text = _fold_esmfold_api(seq, cfg.structure.esmfold_api_url)
        except Exception as exc:  # pragma: no cover - network
            log.warning("esmfold_api_unavailable", error=str(exc))
    elif provider == "esmfold_local":
        if seq_id.startswith("BENCH_"):
            uniprot = seq_id.removeprefix("BENCH_")
            try:
                pdb_text = _fold_alphafold_db(uniprot)
                if pdb_text:
                    log.info("afdb_hit", seq_id=seq_id, uniprot=uniprot)
            except Exception as exc:  # pragma: no cover - network
                log.warning("afdb_unavailable", seq_id=seq_id, error=str(exc))
        if pdb_text is None:
            try:
                pdb_text = _fold_esmfold_local(seq)
                log.info("esmfold_local_done", seq_id=seq_id, residues=len(seq))
            except Exception as exc:
                log.warning("esmfold_local_failed", seq_id=seq_id, error=str(exc))
    elif provider == "alphafold":
        if seq_id.startswith("BENCH_"):
            uniprot = seq_id.removeprefix("BENCH_")
            try:
                pdb_text = _fold_alphafold_db(uniprot)
                if pdb_text:
                    log.info("afdb_hit", seq_id=seq_id, uniprot=uniprot)
            except Exception as exc:  # pragma: no cover - network
                log.warning("afdb_unavailable", seq_id=seq_id, error=str(exc))
        if pdb_text is None:
            try:
                pdb_text = _fold_colabfold(seq_id, seq, cfg.paths.structures_dir)
                log.info("colabfold_done", seq_id=seq_id, residues=len(seq))
            except Exception as exc:
                log.warning("colabfold_failed", seq_id=seq_id, error=str(exc))

    if pdb_text is None:
        pdb_text = _fold_fallback(seq)

    pdb_path.write_text(pdb_text)
    return pdb_path


# ── AlphaFold DB (EBI) ───────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _fold_alphafold_db(uniprot_id: str) -> str | None:
    """Fetch the AF2 PDB for ``uniprot_id`` from EBI's AlphaFold DB.

    Returns the PDB text, or ``None`` if no entry exists. Real AF2 predictions,
    no compute required.
    """
    resp = requests.get(f"{AFDB_API}/{uniprot_id}", timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return None
    pdb_url = payload[0].get("pdbUrl")
    if not pdb_url:
        return None
    pdb_resp = requests.get(pdb_url, timeout=180)
    pdb_resp.raise_for_status()
    return pdb_resp.text


# ── ESMFold (local, no MSA, no TF) ──────────────────────────────────────────
def _fold_esmfold_local(seq: str) -> str:
    """Run ESMFold locally via HuggingFace ``transformers`` and return PDB text.

    Loads ``facebook/esmfold_v1`` once per process (cached on the module). Pure
    PyTorch — no TensorFlow, no ColabFold, no MSA server, no openfold dep. The
    first call downloads the model (~3 GB) into the HF cache.
    """
    import torch

    global _ESMFOLD_MODEL, _ESMFOLD_TOKENIZER
    if _ESMFOLD_MODEL is None:
        from transformers import AutoTokenizer, EsmForProteinFolding

        log.info("esmfold_load_start", source="facebook/esmfold_v1")
        _ESMFOLD_TOKENIZER = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        m = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )
        m.eval()
        # Memory-friendly chunking for CPU inference.
        m.trunk.set_chunk_size(64)
        _ESMFOLD_MODEL = m
        log.info("esmfold_load_done")

    tokenized = _ESMFOLD_TOKENIZER([seq], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = _ESMFOLD_MODEL(tokenized["input_ids"])
    pdb_list = _ESMFOLD_MODEL.output_to_pdb(outputs)
    return pdb_list[0]


_ESMFOLD_MODEL = None
_ESMFOLD_TOKENIZER = None


# ── ColabFold (local AlphaFold2) ─────────────────────────────────────────────
def _fold_colabfold(seq_id: str, seq: str, structures_dir: Path) -> str:
    """Run ``colabfold_batch`` locally on a single sequence; return the PDB text.

    Uses a single AF2 model + single recycle to keep CPU runtime tractable. MSA
    is generated via the public ColabFold MMseqs2 server. Raises if
    ``colabfold_batch`` is not installed.
    """
    cb = shutil.which("colabfold_batch")
    if cb is None:
        raise FileNotFoundError(
            "colabfold_batch not on PATH. Install via "
            "`pip install \"colabfold[alphafold-minus-jax] @ "
            "git+https://github.com/sokrypton/ColabFold\"` + `pip install jax[cpu]`."
        )
    structures_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=structures_dir, prefix="colabfold_") as tmp:
        tmp = Path(tmp)
        fasta = tmp / "input.fasta"
        fasta.write_text(f">{seq_id}\n{seq}\n")
        out_dir = tmp / "out"
        out_dir.mkdir()
        cmd = [
            cb, str(fasta), str(out_dir),
            "--num-models", "1",
            "--num-recycle", "1",
            "--model-type", "alphafold2_ptm",
            "--msa-mode", "mmseqs2_uniref_env",
        ]
        log.info("colabfold_run", seq_id=seq_id, residues=len(seq), cmd=" ".join(cmd))
        # Stream stdout/stderr so progress is visible during long runs.
        subprocess.run(cmd, check=True)
        pdbs = sorted(out_dir.glob("*_relaxed_*.pdb")) or sorted(out_dir.glob("*_unrelaxed_*.pdb"))
        if not pdbs:
            raise RuntimeError(f"ColabFold produced no PDB for {seq_id}")
        return pdbs[0].read_text()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
def _fold_esmfold_api(seq: str, url: str) -> str:  # pragma: no cover - network
    resp = requests.post(url, data=seq, timeout=300)
    resp.raise_for_status()
    return resp.text


def _fold_fallback(seq: str) -> str:
    """Deterministic placeholder backbone (extended chain) as a PDB string.

    Generates CA atoms along a straight 3.8 Å-spaced chain with a pseudo-pLDDT in
    the B-factor column derived from the sequence hash. This keeps WP4 runnable
    with no GPU/network; replace with real ESMFold/AlphaFold for science.
    """
    rng = np.random.default_rng(int(sequence_hash(seq), 16) % (2**32))
    lines = ["REMARK  FALLBACK PLACEHOLDER STRUCTURE - NOT A REAL PREDICTION"]
    for i, aa in enumerate(seq):
        x = i * 3.8
        y = math.sin(i / 5.0) * 2.0
        z = math.cos(i / 5.0) * 2.0
        plddt = float(np.clip(60 + rng.standard_normal() * 15, 0, 100))
        lines.append(
            f"ATOM  {i + 1:>5} CA   {_three(aa)} A{i + 1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def _three(aa: str) -> str:
    return _AA3.get(aa, "GLY")


# ── Structural metrics ───────────────────────────────────────────────────────
def parse_ca_coords(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (CA coordinates Nx3, per-residue B-factor/pLDDT array)."""
    coords, bfac = [], []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            bfac.append(float(line[60:66]))
    return np.asarray(coords), np.asarray(bfac)


def structure_metrics(pdb_path: Path) -> dict[str, float]:
    """Compute compactness / confidence metrics from a predicted structure.

    * ``radius_of_gyration`` — compactness proxy for packing/stability.
    * ``mean_plddt`` — mean per-residue confidence (B-factor column), always in
      the 0-100 AFDB convention regardless of source (transformers ESMFold
      writes 0-1; we auto-scale).
    * ``contact_density`` — CA-CA contacts < 8 Å per residue (packing proxy).
    """
    coords, plddt = parse_ca_coords(pdb_path)
    if len(coords) == 0:
        return {"radius_of_gyration": 0.0, "mean_plddt": 0.0, "contact_density": 0.0}
    # Normalize pLDDT scale: AFDB writes 0-100, transformers ESMFold writes 0-1.
    if plddt.size and plddt.max() <= 1.5:
        plddt = plddt * 100.0
    center = coords.mean(axis=0)
    rg = float(np.sqrt(((coords - center) ** 2).sum(axis=1).mean()))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    contacts = ((dist < 8.0) & (dist > 0)).sum() / 2
    return {
        "radius_of_gyration": rg,
        "mean_plddt": float(plddt.mean()),
        "contact_density": float(contacts / len(coords)),
    }


def analyze_candidates(cfg: Config, ranking: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fold + measure the top candidates and the benchmark panel.

    Returns a per-candidate metrics DataFrame and writes it under ``reports/``.
    Caps the number folded at ``structure.max_structures`` to bound API usage.
    """
    cfg.paths.ensure_dirs()
    candidates = read_fasta(cfg.paths.ancestral_fasta)
    benchmark = read_fasta(cfg.paths.benchmark_fasta) if cfg.paths.benchmark_fasta.exists() else {}

    # Prioritise by WP3 ranking when available.
    order = list(candidates)
    if ranking is not None and "candidate_id" in ranking.columns:
        ranked_ids = [c for c in ranking["candidate_id"] if c in candidates]
        order = ranked_ids + [c for c in candidates if c not in ranked_ids]

    selected = order[: cfg.structure.max_structures]
    rows = []
    for sid in selected:
        pdb = predict_structure(sid, candidates[sid], cfg)
        m = structure_metrics(pdb)
        rows.append({"candidate_id": sid, "kind": "candidate", "pdb": str(pdb), **m})
    for sid, seq in list(benchmark.items())[:5]:
        pdb = predict_structure(f"BENCH_{sid}", seq, cfg)
        m = structure_metrics(pdb)
        rows.append({"candidate_id": f"BENCH_{sid}", "kind": "benchmark", "pdb": str(pdb), **m})

    df = pd.DataFrame(rows)
    out = cfg.paths.reports_dir / "structure_metrics.csv"
    df.to_csv(out, index=False)
    log.info("wp4_complete", n=len(df), out=str(out))
    return df


def structural_confidence_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Project WP4 metrics into the WP3 feature schema (index join key)."""
    cols = ["candidate_id", "mean_plddt", "radius_of_gyration", "contact_density"]
    return metrics[metrics["kind"] == "candidate"][cols].copy()


def mutation_hypotheses(metrics: pd.DataFrame, cfg: Config) -> Path:
    """Write simple, transparent mechanistic hypotheses for the top candidates.

    Compares each candidate's compactness/contact density against the benchmark
    mean and notes likely stability direction. A starting point for the round-2
    interpretation report, not a final mechanistic claim.
    """
    bench = metrics[metrics["kind"] == "benchmark"]
    cand = metrics[metrics["kind"] == "candidate"]
    bench_contacts = bench["contact_density"].mean() if len(bench) else float("nan")

    lines = ["# WP4 — Mutation / Stability Hypotheses\n",
             f"Target family: **{cfg.target.name}**\n",
             f"Benchmark mean contact density: {bench_contacts:.3f}\n"]
    for _, r in cand.iterrows():
        direction = ("higher packing — possible stability gain"
                     if r["contact_density"] >= bench_contacts
                     else "lower packing — verify fold integrity")
        lines.append(
            f"- **{r['candidate_id']}**: Rg={r['radius_of_gyration']:.2f} Å, "
            f"contacts/res={r['contact_density']:.3f}, mean pLDDT={r['mean_plddt']:.1f} "
            f"→ {direction}."
        )
    out = cfg.paths.reports_dir / "mutation_hypotheses.md"
    out.write_text("\n".join(lines) + "\n")
    log.info("hypotheses_written", out=str(out))
    return out
