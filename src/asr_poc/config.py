"""Typed configuration for the ASR POC.

A single YAML (``config/target.yaml`` by default) drives every stage. We parse it
into nested Pydantic models so notebooks get validation, autocompletion, and a
single source of truth for paths and thresholds — instead of scattering magic
numbers across cells.

Usage
-----
    from asr_poc.config import load_config
    cfg = load_config()                 # default config/target.yaml
    cfg = load_config("config/my.yaml") # explicit path
    cfg.paths.curated_fasta             # canonical output locations
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Repository root = two levels up from this file (src/asr_poc/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "target.yaml"


class TargetSpec(BaseModel):
    name: str
    description: str = ""
    reference_uniprot: str | None = None
    pfam_id: str | None = None
    ec_number: str | None = None
    catalytic_motif_regex: str = ""
    catalytic_triad: list[str] = Field(default_factory=list)


class BlastSpec(BaseModel):
    program: str = "blastp"
    database: str = "nr"
    evalue: float = 1e-5
    hitlist_size: int = 500


class RetrievalSpec(BaseModel):
    source: str = "uniprot"
    uniprot_query: str = ""
    max_sequences: int = 500
    blast: BlastSpec = Field(default_factory=BlastSpec)


class CurationSpec(BaseModel):
    min_length: int = 150
    max_length: int = 600
    drop_fragments: bool = True
    cluster_identity: float = 0.90
    require_catalytic_motif: bool = True


class PhylogenySpec(BaseModel):
    aligner: str = "mafft"
    mafft_opts: str = "--auto"
    iqtree_model: str = "MFP"
    ultrafast_bootstrap: int = 1000
    asr_engine: str = "paml"
    ambiguous_pp_threshold: float = 0.80
    max_alternates_per_node: int = 5
    nodes_to_reconstruct: int | str = 10


class EmbeddingsSpec(BaseModel):
    """ESM-2 embedding provider configuration. Inference-only, no training."""

    provider: str = "local"             # local | api | fallback
    esm_model_local: str = "esm2_t12_35M_UR50D"
    esm_api_url: str = ""
    batch_size: int = 8


class RankingWeights(BaseModel):
    """Composite-score weights (priors, not learned). Sum is not enforced."""

    sim: float = 0.35           # anchor-set embedding cosine similarity
    structure: float = 0.30     # ESMFold pLDDT / 100
    motif: float = 0.15         # catalytic motif preservation
    conservation: float = 0.10  # match family consensus at conserved columns
    uncertainty: float = 0.10   # ASR posterior entropy penalty (subtracted)


class RankingSpec(BaseModel):
    """Inference-only ranking configuration (no XGBoost, no training)."""

    anchor_uniprot_count: int = 25      # extant entries used as anchor set
    knn_k: int = 5                       # neighbours per candidate for emb_sim
    pre_fold_top_k: int = 10             # candidates to fold via ESMFold
    final_top_n: int = 3                 # candidates surfaced in the report
    weights: RankingWeights = Field(default_factory=RankingWeights)
    cluster_n: int = 8                   # KMeans clusters for diversity selection
    random_state: int = 42


class StructureSpec(BaseModel):
    folder_provider: str = "api"
    esmfold_api_url: str = ""
    max_structures: int = 20
    active_site_radius_angstrom: float = 8.0


class RunSpec(BaseModel):
    seed: int = 42
    output_root: str = "."


class Paths(BaseModel):
    """Canonical artifact locations, resolved from ``run.output_root``.

    Matches the repo layout in enzymes.docx §12 so every stage reads/writes in
    a predictable place and notebooks never invent their own paths.
    """

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw_sequences"

    @property
    def curated_dir(self) -> Path:
        return self.root / "data" / "curated_sequences"

    @property
    def alignments_dir(self) -> Path:
        return self.root / "data" / "alignments"

    @property
    def phylogeny_dir(self) -> Path:
        return self.root / "data" / "phylogeny"

    @property
    def ancestral_dir(self) -> Path:
        return self.root / "data" / "ancestral_sequences"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def structures_dir(self) -> Path:
        return self.root / "structure_predictions"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    # Frequently used files ----------------------------------------------------
    @property
    def raw_fasta(self) -> Path:
        return self.raw_dir / "homologs.fasta"

    @property
    def curated_fasta(self) -> Path:
        return self.curated_dir / "curated.fasta"

    @property
    def metadata_csv(self) -> Path:
        return self.curated_dir / "metadata.csv"

    @property
    def benchmark_fasta(self) -> Path:
        return self.curated_dir / "benchmark_extant.fasta"

    @property
    def msa_fasta(self) -> Path:
        return self.alignments_dir / "msa.fasta"

    @property
    def tree_file(self) -> Path:
        return self.phylogeny_dir / "tree.treefile"

    @property
    def ancestral_fasta(self) -> Path:
        return self.ancestral_dir / "candidates.fasta"

    @property
    def uncertainty_csv(self) -> Path:
        return self.ancestral_dir / "sitewise_uncertainty.csv"

    @property
    def ranking_csv(self) -> Path:
        return self.reports_dir / "candidate_ranking.csv"

    # Inference-only architecture artifacts ------------------------------------
    @property
    def candidate_embeddings(self) -> Path:
        return self.embeddings_dir / "candidates.parquet"

    @property
    def anchor_embeddings(self) -> Path:
        return self.embeddings_dir / "anchors.parquet"

    @property
    def anchor_fasta(self) -> Path:
        return self.curated_dir / "anchors.fasta"

    @property
    def embeddings_meta(self) -> Path:
        return self.embeddings_dir / "meta.json"

    @property
    def signals_csv(self) -> Path:
        """Per-candidate breakdown of every score component (the audit trail)."""
        return self.reports_dir / "candidate_signals.csv"

    @property
    def report_md(self) -> Path:
        return self.reports_dir / "scientific_report.md"

    def ensure_dirs(self) -> None:
        """Create every output directory (idempotent)."""
        for d in (
            self.raw_dir, self.curated_dir, self.alignments_dir, self.phylogeny_dir,
            self.ancestral_dir, self.embeddings_dir, self.models_dir,
            self.structures_dir, self.reports_dir, self.results_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


class ReportSpec(BaseModel):
    """Scientific report configuration. Pure local templating, no external APIs."""

    top_n: int = 3                  # candidates rendered in the report body


class LlmSpec(BaseModel):
    """Configuration for LLM-assisted scientific evaluation."""

    provider: str = "fallback"      # gemini | openai | claude | fallback
    model: str = "gemini-2.5-flash"
    api_key_env_var: str = "GEMINI_API_KEY"
    temperature: float = 0.1
    max_retries: int = 3


class Config(BaseModel):
    target: TargetSpec
    retrieval: RetrievalSpec = Field(default_factory=RetrievalSpec)
    curation: CurationSpec = Field(default_factory=CurationSpec)
    phylogeny: PhylogenySpec = Field(default_factory=PhylogenySpec)
    embeddings: EmbeddingsSpec = Field(default_factory=EmbeddingsSpec)
    ranking: RankingSpec = Field(default_factory=RankingSpec)
    structure: StructureSpec = Field(default_factory=StructureSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)
    llm: LlmSpec = Field(default_factory=LlmSpec)
    run: RunSpec = Field(default_factory=RunSpec)
    paths: Paths

    model_config = {"arbitrary_types_allowed": True}


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the target YAML into a :class:`Config`.

    ``run.output_root`` is resolved relative to the project root when given as a
    relative path, so notebooks can be launched from anywhere.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    output_root = Path(raw.get("run", {}).get("output_root", "."))
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    raw["paths"] = {"root": output_root}
    return Config.model_validate(raw)
