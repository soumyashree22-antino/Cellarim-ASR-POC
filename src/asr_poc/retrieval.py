"""WP1 — Sequence dataset build.

Retrieve homologous enzyme sequences from public databases (UniProt REST, with a
BLAST path stub), clean them (length, fragments, redundancy, catalytic motif),
annotate metadata, and emit the curated FASTA + metadata table + benchmark panel.

The functions are deliberately small and composable so the notebook reads as a
pipeline and each filter can be unit-tested in isolation. Network access is
isolated to :func:`fetch_uniprot` so the rest is testable offline.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .io_utils import get_logger, write_fasta

log = get_logger("wp1.retrieval")

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"


# ── Retrieval ────────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=30))
def _uniprot_request(params: dict) -> str:
    resp = requests.get(UNIPROT_STREAM_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.text


def fetch_uniprot(query: str, max_sequences: int) -> pd.DataFrame:
    """Fetch sequences + metadata from UniProt as a tidy DataFrame.

    Returns one row per accession with: ``id, organism, length, fragment,
    ec_number, protein_name, annotation_score, sequence``.
    """
    fields = ",".join(
        ["accession", "organism_name", "length", "fragment", "ec",
         "protein_name", "annotation_score", "sequence"]
    )
    params = {"query": query, "format": "tsv", "fields": fields, "size": 500}
    log.info("uniprot_fetch", query=query, max_sequences=max_sequences)
    text = _uniprot_request(params)
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df = df.rename(
        columns={
            "Entry": "id",
            "Organism": "organism",
            "Length": "length",
            "Fragment": "fragment",
            "EC number": "ec_number",
            "Protein names": "protein_name",
            "Annotation": "annotation_score",
            "Sequence": "sequence",
        }
    )
    if len(df) > max_sequences:
        df = df.head(max_sequences)
    log.info("uniprot_fetched", n=len(df))
    return df


def fetch_blast(cfg: Config) -> pd.DataFrame:  # pragma: no cover - network/blast
    """BLAST retrieval path (stub).

    Uses Biopython's NCBIWWW for a remote blastp against the configured database,
    seeded by the reference sequence. Kept minimal for the POC; UniProt is the
    default source. Implemented here so the source is swappable via config.
    """
    raise NotImplementedError(
        "BLAST retrieval is stubbed for the POC; set retrieval.source=uniprot."
    )


def retrieve(cfg: Config) -> pd.DataFrame:
    """Dispatch to the configured retrieval source."""
    if cfg.retrieval.source == "uniprot":
        return fetch_uniprot(cfg.retrieval.uniprot_query, cfg.retrieval.max_sequences)
    if cfg.retrieval.source == "blast":
        return fetch_blast(cfg)
    raise ValueError(f"Unknown retrieval source: {cfg.retrieval.source}")


# ── Cleaning filters (pure, testable) ────────────────────────────────────────
def has_catalytic_motif(sequence: str, motif_regex: str) -> bool:
    if not motif_regex:
        return True
    return re.search(motif_regex, sequence) is not None


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply WP1 cleaning rules. Returns a filtered, deduplicated DataFrame.

    Removes: exact-duplicate sequences, fragments/partial entries, sequences
    outside the length window, and (optionally) sequences lacking the catalytic
    motif. Records the row count dropped at each step in the log.
    """
    cur = cfg.curation
    n0 = len(df)
    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].str.upper().str.replace(r"\s+", "", regex=True)

    # Length window
    df = df[(df["length"] >= cur.min_length) & (df["length"] <= cur.max_length)]

    # Fragments / partial
    if cur.drop_fragments and "fragment" in df.columns:
        frag = df["fragment"].fillna("").astype(str).str.strip().str.lower()
        df = df[~frag.isin({"fragment", "fragments"})]
        # also drop descriptions explicitly marked partial
        if "protein_name" in df.columns:
            df = df[~df["protein_name"].fillna("").str.contains("partial", case=False)]

    # Catalytic motif
    if cur.require_catalytic_motif:
        keep = df["sequence"].apply(
            lambda s: has_catalytic_motif(s, cfg.target.catalytic_motif_regex)
        )
        df = df[keep]

    # Exact-duplicate sequences (keep first / highest annotation if available)
    if "annotation_score" in df.columns:
        df = df.sort_values("annotation_score", ascending=False)
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    log.info("cleaned", before=n0, after=len(df), dropped=n0 - len(df))
    return df


def reduce_redundancy(df: pd.DataFrame, identity: float) -> pd.DataFrame:
    """Greedy redundancy reduction to balance diversity vs duplication.

    A lightweight, dependency-free stand-in for MMseqs2/CD-HIT suitable for the
    POC scale: greedily keep a sequence unless it is >= ``identity`` similar
    (k-mer Jaccard proxy) to one already kept. For large datasets, swap in
    MMseqs2 ``easy-cluster`` behind this same interface.
    """

    def kmers(seq: str, k: int = 4) -> set[str]:
        return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}

    kept_rows: list[int] = []
    kept_kmers: list[set[str]] = []
    for idx, seq in df["sequence"].items():
        ks = kmers(seq)
        redundant = False
        for prev in kept_kmers:
            inter = len(ks & prev)
            union = len(ks | prev) or 1
            if inter / union >= identity:
                redundant = True
                break
        if not redundant:
            kept_rows.append(idx)
            kept_kmers.append(ks)
    out = df.loc[kept_rows].reset_index(drop=True)
    log.info("redundancy_reduced", before=len(df), after=len(out), identity=identity)
    return out


def annotate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add derived metadata columns used downstream (WP2/WP3)."""
    df = df.copy()
    df["family"] = cfg.target.name
    df["taxonomy"] = df.get("organism", pd.Series([""] * len(df)))
    # Position of the catalytic nucleophile elbow (motif start), if present.
    motif = cfg.target.catalytic_motif_regex
    df["motif_start"] = df["sequence"].apply(
        lambda s: (m.start() if (m := re.search(motif, s)) else -1) if motif else -1
    )
    return df


# ── Orchestration ────────────────────────────────────────────────────────────
def build_dataset(cfg: Config, df_raw: pd.DataFrame | None = None) -> dict[str, int]:
    """Run the full WP1 pipeline and write artifacts. Returns a summary dict.

    ``df_raw`` lets tests inject a fixture instead of hitting the network.
    """
    cfg.paths.ensure_dirs()
    raw = df_raw if df_raw is not None else retrieve(cfg)

    # Persist the raw pull for provenance.
    write_fasta(_to_records(raw), cfg.paths.raw_fasta)

    curated = clean(raw, cfg)
    curated = reduce_redundancy(curated, cfg.curation.cluster_identity)
    curated = annotate(curated, cfg)

    write_fasta(_to_records(curated), cfg.paths.curated_fasta)
    meta_cols = [c for c in curated.columns if c != "sequence"]
    curated[meta_cols].to_csv(cfg.paths.metadata_csv, index=False)

    benchmark = select_benchmark_panel(curated, cfg)
    write_fasta(_to_records(benchmark), cfg.paths.benchmark_fasta)

    summary = {
        "raw": len(raw),
        "curated": len(curated),
        "benchmark": len(benchmark),
    }
    log.info("wp1_complete", **summary)
    return summary


def select_benchmark_panel(df: pd.DataFrame, cfg: Config, n: int = 5) -> pd.DataFrame:
    """Pick a small panel of well-annotated extant enzymes as comparators."""
    sort_col = "annotation_score" if "annotation_score" in df.columns else "length"
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def _to_records(df: pd.DataFrame) -> Mapping[str, str]:
    return dict(zip(df["id"], df["sequence"], strict=False))
