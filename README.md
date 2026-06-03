# AI-Based Ancestral Sequence Reconstruction (ASR) of Enzymes — POC

A **lean, inference-only** proof-of-concept pipeline that discovers and
prioritises **ancestral enzyme variants** by combining classical phylogenetics
with **pretrained protein foundation models** (ESM-2, ESMFold/AlphaFold) and
similarity-based reasoning. No supervised training. No labels.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale.

```
UniProt/BLAST ─► WP1 curate ─► WP2 align+tree+ASR ─► WP3 ESM-2 embed + similarity ─►
                ─► WP4 fold top-K (local ESMFold) ─► WP5 composite rank + report ─► leads
```

Current target: **lipase** (configurable via `config/target.yaml`).

## What changed vs the XGBoost version

| Component | Before | Now |
|---|---|---|
| WP3 ranker | XGBoost trained on a placeholder thermostability proxy | **No training.** ESM-2 cosine to an anchor set + motif/conservation/uncertainty |
| Top-3 selection | Sequences that happened to maximise an unvalidated proxy | Sequences that sit on the lipase family manifold per ESM-2 |
| Structure step | Decorative — ran *after* the ranking and didn't change it | **Direct ranking input** — pLDDT/100 is 30% of the composite score |
| Audit trail | Just a ranking CSV | `reports/candidate_signals.csv` breaks every score into its components |
| Report | None | `reports/scientific_report.md` — deterministic templated markdown |
| Tuning | Retrain on new labels | Edit `config/target.yaml` weights, rerun in seconds |

## Layout

```
src/asr_poc/
  config.py                Typed configuration (Pydantic) loaded from target.yaml
  io_utils.py              FASTA / logging / seeds / hashing
  retrieval.py             WP1 — UniProt mining + curation + benchmark panel
  phylo.py                 WP2 — MAFFT + IQ-TREE + ASR (gap-masked) + candidate pool
  embeddings.py            WP3 — ESM-2 embeddings (local / API / fallback), parquet cache
  similarity.py            WP3 — cosine, kNN, clustering, diversity selection
  motif_conservation.py    WP3 — catalytic motif + conservation + uncertainty penalty
  structure.py             WP4 — fold (ESMFold/ColabFold/AFDB) + metrics (pLDDT, Rg)
  structure_scoring.py     WP4 — map metrics to [0,1] ranking signals
  ranking.py               WP3/WP5 — pre-fold and final composite scores (no training)
  report.py                WP5 — deterministic templated markdown report (no LLM)
```

## Setup

```bash
# Binary tools (MAFFT, IQ-TREE) via conda OR Homebrew:
brew install mafft iqtree3       # macOS
# or:
conda env create -f environment.yml && conda activate asr-poc

# Python deps:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Optional: AlphaFold2 (ColabFold) — alternative folder for ancestral candidates
# (default is local ESMFold, which is already in requirements.txt)
pip install "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
pip install "jax[cpu]"
```

## Run

```bash
jupyter lab          # then run notebooks 01 → 05 in order
```

Or programmatically:

```python
from asr_poc.config import load_config
from asr_poc import retrieval, phylo, ranking, structure, report

cfg = load_config()

# WP1
retrieval.build_dataset(cfg)

# WP2 (needs MAFFT + IQ-TREE on PATH)
msa = phylo.align(cfg)
phylo.build_tree(cfg, msa)
state = phylo.reconstruct_ancestors(cfg, msa)
phylo.build_candidate_pool(cfg, state_file=state, msa_path=msa)

# WP3 — embeddings + similarity-based pre-fold ranking
signals = ranking.pre_fold_rank(cfg)
fold_ids = ranking.candidates_to_fold(signals, cfg)

# WP4 — fold the K diversity-selected candidates
metrics = structure.analyze_candidates(cfg)

# WP5 — final composite + scientific report
final = ranking.final_rank(signals, metrics, cfg)
report.write_report(cfg, final)
```

To target another enzyme family, edit `config/target.yaml` (UniProt query,
catalytic motif, weights) — no code changes.

## Tuning the ranker (no retraining)

The composite score weights live in `config/target.yaml` under `ranking.weights`.
Defaults total 1.0 on the positive side; the uncertainty term is subtracted.
Edit and rerun `final_rank()` — there is no fitted state to retrain.

```yaml
ranking:
  weights:
    sim: 0.35           # ESM-2 anchor cosine similarity (family manifold)
    structure: 0.30     # ESMFold / AF2 pLDDT / 100 (foldability)
    motif: 0.15         # catalytic GxSxG present + triad order
    conservation: 0.10  # match family consensus at conserved MSA columns
    uncertainty: 0.10   # ASR posterior entropy (subtracted)
```

## Compute notes

Runs on a local Mac (CPU). GPU-heavy steps offload to hosted APIs:

* **Embeddings (WP3)**: local small ESM-2 (`esm2_t12_35M_UR50D`, ~150 MB,
  CPU-fast). Switch `embeddings.provider` to `api` for the 650M / 3B variants.
* **Structures (WP4)**: ESMFold API, or AlphaFold DB lookup for extant
  benchmarks (instant + free), or ColabFold local for ancestral candidates
  (~15 min/seq on M-series CPU).

Both have deterministic, no-network fallbacks so the test suite runs offline.

## Test

```bash
pytest                # offline smoke test on a 15-seq fixture family
pytest -m live        # opt-in: real APIs/tools
ruff check src tests  # lint
```

## Deliverables produced

* `data/curated_sequences/curated.fasta` + `metadata.csv` + `benchmark_extant.fasta` — WP1
* `data/alignments/`, `data/phylogeny/`, `data/ancestral_sequences/` — WP2
* `embeddings/candidates.parquet`, `embeddings/anchors.parquet` — WP3
* `structure_predictions/*.pdb` — WP4 (real AF2/ESMFold predictions)
* `reports/candidate_signals.csv` — full audit trail (every score component)
* `reports/candidate_ranking.csv` — final ranking
* `reports/structure_metrics.csv` — fold quality per structure
* `reports/scientific_report.md` — templated markdown interpretation of the top-N
