# ASR Enzyme Engineering POC — Hybrid LLM & Foundation Model Architecture

**Pivot:** the previous WP3 trained an XGBoost regressor against a hand-crafted
physicochemical proxy label and used those scores to pick candidates to fold.
That coupling was the root cause of the very low pLDDT we observed on the top
picks — the proxy didn't actually predict foldability, so the ranker happily
surfaced sequences AF2 doesn't trust.

**New direction:** no training. Use **pretrained protein foundation models**
(ESM-2 for embeddings, ESMFold for structure) combined with **LLM-assisted stability scoring**
and **deterministic scientific validation**. This hybrid pipeline reasons over structured
biological feature tables (embedding similarities, motif preservation, family conservation,
ASR uncertainty, hydrophobicity, mutation distances) using pluggable LLM models (Gemini, Claude, OpenAI)
while grounding and validating LLM reasoning with a deterministic offline rank-correlation layer.

> Final ranking is a **transparent, hybrid composite of inference-time
> biological metrics and LLM-assisted reasoning** — no labels, no training.
> Every step is mathematically grounded, biologically validated, and optimized for
> structural diversity using K-Means sequence clustering.

---

## 1. Updated workflow

```
                  WP1 — Dataset (unchanged)
   UniProt ─► retrieve ─► clean ─► curated + benchmark
                                       │
                                       ▼
                  WP2 — ASR (unchanged)
                MAFFT ─► IQ-TREE 3 ─► ASR
                                       │
                                       ▼
                ┌──────────────────────────────────────────┐
                │  Candidate Variant Pool                  │
                │  (ML ancestors + alternates, gap-masked) │
                │  + per-candidate ASR uncertainty         │
                └────────────────────┬─────────────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
   WP3a Feature Table       WP3b LLM Stability       WP3c Scientific Grounding
   Builds feature table     Pluggable LLM scoring    Spearman & Kendall rank
   with similarities,       of biological features   correlation checks between
   motifs, conservation,    table (Gemini, Claude,   LLM scores & deterministic
   hydrophobicity, etc.     OpenAI, or fallback)     evolutionary priors
        │                            │                             │
        └────────────────────────────┼─────────────────────────────┘
                                     ▼
                        WP3d Diversity Selection
                        K-Means clustering on embeddings
                        to select cluster-diverse candidates
                                     │
                                     ▼
                  WP3e Pre-fold Rank (Pass 1)
                  70% LLM Stability + 30% Deterministic Prior
                                     │
                                     ▼
                  WP4 Structure (ESMFold, real predictions)
                  → pLDDT, Rg, contact density per candidate
                  → benchmark fetched from AlphaFold DB
                                     │
                                     ▼
                  WP5 Final Composite Hybrid Rank (Pass 2)
                  50% LLM Stability + 30% ESMFold + 20% Deterministic Prior
                                     │
                                     ▼
                  WP5 Scientific Report (LLM-interpreted)
                  - Per-candidate signals table + nearest anchors
                  - Local Markdown report with LLM biological commentary
```

WP1 and WP2 are unchanged. WP3 is reorganized around feature table extraction, pluggable LLM stability scoring, deterministic validation correlation, and K-Means diversity selection. WP4 folds the selected candidates, and WP5 computes the final composite ranking and generates a comprehensive interpretative scientific report.

---

## 2. The 10 design questions, answered

### 2.1 How ESM-2 embeddings replace XGBoost

ESM-2 was trained on ~250M natural protein sequences with a masked-language
objective. Its hidden representations encode evolutionary, structural, and
biophysical features without ever seeing a label. We exploit two properties:

1. **Family manifold.** Real lipases cluster tightly in ESM-2 embedding space.
   Distance to that cluster is a strong "looks like a lipase" prior.
2. **Foldability signal.** Sequences with garbled regions, frameshift-like
   composition, or motif disruption land further from the manifold — and
   ESMFold's pLDDT correlates with that distance.

So instead of training a model to predict "is this enzyme good?", we measure
**how close it sits to known good ones** and how confidently the foundation
model can fold it. Both are zero-training, inference-only signals.

### 2.2 How proteins are ranked without training

A **transparent, hybrid scoring pipeline** that integrates LLM stability reasoning with a deterministic scientific validation layer.

1. **Deterministic Prior (ground truth evolutionary rules):**
   ```
   score(c) = w_sim · emb_sim(c)            ∈ [0, 1]
           + w_mot · motif_preservation(c)  ∈ [0, 1]
           + w_con · conservation(c)        ∈ [0, 1]
           − w_unc · uncertainty(c)         ∈ [0, 1]
   ```
   Default weights (sum to 1 on the positive side, penalty subtracted) are exposed in `config/target.yaml`.

2. **LLM-Assisted Stability Scoring:**
   An LLM (Gemini, OpenAI, Claude, or deterministic offline fallback) reasons over a structured, serialized biological feature table (without raw sequences to prevent hallucination) to assign a biological stability score ∈ [0, 10] and provide detailed biological rationale.

3. **Hybrid Pre-Fold Rank (Pass 1):**
   ```
   pre_score(c) = 0.70 · (llm_stability_score / 10.0) + 0.30 · deterministic_prior(c)
   ```

4. **Hybrid Final Composite Rank (Pass 2):**
   ```
   final_score(c) = 0.50 · (llm_stability_score / 10.0) + 0.30 · struct_conf(c) + 0.20 · deterministic_score(c)
   ```
   This ensures that the final rank integrates:
   * 50% LLM Stability reasoning under structural evidence.
   * 30% ESMFold self-reported structural confidence (pLDDT).
   * 20% Strict, rule-based deterministic scoring.

5. **Rank Correlation Grounding:**
   Spearman's Rho and Kendall's Tau rank-correlation coefficients are calculated between the LLM score and the deterministic priors. If rank correlation drops significantly or outlier deltas exceed 3.0 points out of 10, warning flags are triggered, ensuring transparent grounding.

### 2.3 How embedding similarity works

For each sequence `s` we compute a mean-pooled ESM-2 embedding `e(s) ∈ ℝ^d`
(d = 1280 for ESM-2 650M, 480 for the 150M small model).

Define an **anchor set A** = high-quality extant lipases (the 5-seq benchmark
+ the top annotation-score curated entries; ~25 anchors total).

For each candidate `c`, define:

```
emb_sim(c)  = mean_{a ∈ topK(c, A, k=5)} cos(e(c), e(a))
            = average cosine similarity to its 5 nearest anchors
```

`emb_sim ∈ [-1, 1]` then rescaled to `[0, 1]` via `(x+1)/2` for ranking. Using
**top-k** (not all anchors) makes the score robust to anchor outliers and to
the natural diversity of the family.

### 2.4 How structural confidence contributes

ESMFold (or AlphaFold via the AFDB+ColabFold path we already wired) gives a
per-residue **pLDDT** confidence on the predicted structure.

```
struct_conf(c) = mean(pLDDT_residues(c)) / 100   ∈ [0, 1]
```

Optional refinements (cheap, computed from the same PDB):

* **packing_score**: `contact_density(c) / contact_density(benchmark_mean)`,
  clipped to `[0, 1]`. Penalizes unfolded/extended candidates.
* **active_site_geom**: distance triangle of the Ser-Asp-His catalytic triad
  vs the benchmark median. Drops the score if the triad is geometrically
  disrupted (> 2 Å deviation).

For the v1 MVP, just `mean(pLDDT)` is enough.

### 2.5 How candidate prioritization works (two-pass)

Folding is the expensive step. We use a cheap pre-rank to decide *what* to
fold, then a full composite hybrid ranking to evaluate the folded candidates:

1. **Pre-fold Rank (Pass 1 - all candidates):**
   * Build the structured biological feature table for all candidates.
   * Run LLM stability scoring to obtain a stability rating ∈ [0, 10] and rationale.
   * Compute a balanced hybrid score:
     `pre_score(c) = 0.70 * (llm_stability_score / 10.0) + 0.30 * deterministic_prior(c)`
2. **Diversity Selection (K-Means):**
   * Execute K-Means clustering on ESM-2 embeddings to identify phylogenetic/representational clades.
   * Pick a cluster-diverse top-K (e.g. K = 10) set of candidates for structural folding to ensure biological diversity.
3. **Fold those K candidates** with ESMFold to obtain structural fold metrics (pLDDT, compactness Rg, contact density).
4. **Final Rank (Pass 2 - folded candidates):**
   * Re-build the biological feature table incorporating structural folding evidence.
   * Re-score the folded candidates with the LLM under structural evidence.
   * Calculate rank correlation (Spearman & Kendall) and run validation outlier checks.
   * Compute the final composite hybrid score:
     `final_score(c) = 0.50 * (llm_stability_score / 10.0) + 0.30 * struct_conf(c) + 0.20 * deterministic_score(c)`
5. **Pick top-N** (e.g. N = 3) for the interpretative report.

This keeps total compute bounded at K folds × ~5 min ESMFold, while injecting advanced AI scientist reasoning over structural/evolutionary metrics.

### 2.6 How ESMFold integrates

ESMFold is wrapped behind the same provider abstraction we already use in
`structure.py`. New design:

* `embeddings.py` — produces `e(s)` via ESM-2 (local or HF Inference API).
* `structure.py` — produces `pdb(s) + pLDDT(s)` via ESMFold (local or API).
* Both cache by SHA-1 of the sequence so re-runs are free.

ESMFold's first 36 layers of ESM-2-3B *are* the same backbone that produced
the embedding — so the embedding similarity and the structure confidence are
*coherent measurements from a single model family*, not two independent
oracles. That coherence is a strength of the inference-only design.

### 2.7 The final scoring system

```python
def final_composite_hybrid_score(c, signals, weights):
    # 50% LLM Stability reasoning under structure
    # 30% ESMFold confidence
    # 20% Rule-based evolutionary prior
    return (
        0.50 * (signals.llm_stability_score / 10.0)
      + 0.30 * signals.struct_conf
      + 0.20 * signals.deterministic_score
    )
```

Properties we deliberately want:
* **LLM reasoning**: leverages broad biological knowledge via pluggable models.
* **Deterministic validation**: rank correlation metrics guarantee grounding and flag hallucinations.
* **Tunable without retraining**: change YAML weights, rerun in seconds.
* **Inspectable**: every final ranking contains the LLM reasoning logs, correlation coefficients, and raw feature metrics.

### 2.8 How to compare candidates against known stable enzymes

Two complementary mechanisms:

1. **Anchor set similarity** (already in `emb_sim`). The anchor set IS the
   "known good" reference. Cosine to it = manifold proximity.
2. **Structural delta** (after folding). For each candidate, compute:
   * `Δ pLDDT = pLDDT(c) − mean(pLDDT(anchors))`
   * `Δ Rg = (Rg(c) − Rg(anchors)) / Rg(anchors)` (relative compactness)
   * `Δ contacts = contacts(c) − contacts(anchors)`

These deltas feed both the composite (via `struct_conf` and `packing_score`)
and the templated report (as quantitative claims).

### 2.9 How vector similarity and clustering are used

* **Cosine** is the canonical distance for ESM-2 embeddings (vectors are not
  unit-normalized but cosine is scale-invariant and matches how the model was
  used during pretraining for attention).
* **Nearest-neighbor (k=5)** to anchors → `emb_sim` (see 2.3).
* **KMeans (k≈10) or HDBSCAN** on the full candidate set → cluster IDs.
  Diversity selection: when picking the top-K to fold, prefer one candidate
  per cluster up to K, then fill from the global ranking. Prevents the K
  slots being eaten by near-duplicate alternates of one ancestor.
* **UMAP / PCA → 2D** for the report only (visualization, not ranking).

### 2.10 Why this still produces meaningful rankings without supervised learning

Because **every input to the rank is itself a measurement, not a prediction**:

| Signal | What it actually measures |
|---|---|
| `emb_sim` | Distance to natural-sequence manifold (ESM-2 pretraining) |
| `struct_conf` | Self-reported confidence of a state-of-the-art folder |
| `motif` | Direct check that the catalytic site is intact |
| `conservation` | Direct check at MSA-supported invariant positions |
| `uncertainty` | Direct readout from IQ-TREE's posterior |

Combining honest measurements with sensible weights gives a defensible
ranking even with zero labels. The XGBoost approach hid a placeholder label
inside a model; this approach exposes every assumption in the YAML.

---

## 3. Embedding pipeline & storage

### 3.1 Provider abstraction

```python
class EmbeddingProvider(Protocol):
    def embed(self, sequences: dict[str, str]) -> dict[str, np.ndarray]: ...
```

Two implementations (config-selected):

* **local**: `fair-esm` package, model `esm2_t12_35M_UR50D` (dim=480) for
  development, `esm2_t30_150M_UR50D` (dim=640) for runs that need quality.
  Both run on CPU.
* **api**: HuggingFace Inference API or replicate.com endpoint with the full
  `esm2_t33_650M_UR50D` (dim=1280). Same interface.

The deterministic seeded hash fallback we already have stays in place purely
for offline tests — never used scientifically.

### 3.2 Storage

For ~10³ candidates this is trivial:

```
embeddings/
  candidates.parquet      # one row per id, columns = e_0..e_{d-1}
  anchors.parquet         # same schema for the anchor set
  meta.json               # model name, dim, date, sha-1 → id mapping
```

Parquet because pandas and Arrow read it instantly. **No vector DB needed at
this scale.** For ≥ 10⁵ candidates (out of scope for this POC), drop FAISS
behind the same provider interface — neither caller code nor formulas change.

### 3.3 Similarity metrics

* `cosine(u, v) = u·v / (‖u‖·‖v‖)` — primary.
* `dot(u, v)` — used when vectors are pre-normalized.
* `euclidean(u, v)` — only for clustering robustness checks.

`sklearn.metrics.pairwise.cosine_similarity` for batched computation. No
training, no fitting — just matrix multiplies.

---

## 4. New repository structure

```
project/
├── config/target.yaml          # extended with llm configs, ranking.weights, anchor selection
├── src/asr_poc/
│   ├── config.py               # extended Pydantic types for LLM specs and ranking weights
│   ├── io_utils.py             # unchanged
│   ├── retrieval.py            # WP1 — unchanged
│   ├── phylo.py                # WP2 — unchanged (gap-masking already in)
│   ├── embeddings.py           # NEW — ESM-2 provider, batch embed, cache
│   ├── similarity.py           # NEW — cosine, kNN, anchor selection, clustering
│   ├── feature_table.py        # NEW — structured biological feature table builder
│   ├── llm_scoring.py          # NEW — pluggable LLM client (Gemini, Claude, OpenAI, fallback)
│   ├── deterministic_validation.py  # NEW — Spearman/Kendall rank correlation checks
│   ├── structure.py            # WP4 — fold + pLDDT/Rg/contacts (kept)
│   ├── structure_scoring.py    # NEW — pLDDT→[0,1], packing, active-site geom
│   ├── motif_conservation.py   # NEW — motif + per-column conservation checks
│   ├── ranking.py              # REWRITTEN — composite hybrid score, two-pass
│   ├── vector_analysis.py      # NEW — clustering, UMAP/PCA for the report
│   └── report.py               # NEW — scientific report generator with LLM comments
├── notebooks/
│   ├── 01_wp1_dataset_build.ipynb
│   ├── 02_wp2_align_phylo_asr.ipynb
│   ├── 03_wp3_embed_and_similarity.ipynb     # RENAMED + rewritten for hybrid flow
│   ├── 04_wp4_fold_topK.ipynb                # focuses on top-K folding with diversity selection
│   ├── 05_wp5_rank_and_report.ipynb          # final composite hybrid ranking + report
│   └── 06_explore_embedding_space.ipynb      # UMAP / cluster viz
├── tests/                                    # updated for new modules + hybrid smoke tests
└── reports/, embeddings/, structure_predictions/, ...  # outputs
```

**Removed**: nothing on disk yet (XGBoost dep was an import, model on disk
gets deleted on refactor). **Conceptually removed**: training, hyperparam
tuning, label loaders, model registry, retraining loop.

**Added**: 8 new modules (above). Two new notebooks (`05`, `06`). One new
config block (`llm`, `ranking.weights`, `embeddings.*`, `report.*`).

---

## 5. Tech stack (updated)

**In:**
* **ESM-2** (`fair-esm`) — embeddings + ESMFold backbone.
* **ESMFold (local)** via HuggingFace `transformers`, plus AlphaFold DB lookup
  for extant benchmarks. Real predictions, no external folding APIs.
* **PyTorch** — runtime for ESM-2 and ESMFold (CPU is fine at this scale).
* **Pluggable LLM Providers** (Gemini, Claude, OpenAI) — for biological stability scoring and report interpretation.
* **Biopython, NumPy, Pandas, PyArrow** — IO, tabular work.
* **scikit-learn** — `pairwise.cosine_similarity`, `KMeans` clade clustering, and optional `umap-learn`.
* **SciPy** — Spearman and Kendall rank correlation checking.
* **PyMOL / ChimeraX** — visualization in the report (optional, manual).

**Out:**
* **XGBoost** — removed from `requirements.txt` and all code paths.
* `joblib` model persistence (no models to persist).
* Any `cross_val_score` / `train_test_split` / supervised metric.

---

## 6. MVP roadmap (simplest path to a working POC)

The completed hybrid prioritization pipeline includes the following core modules:

1. **`embeddings.py`** — wraps ESM-2 with cache + parquet IO.
2. **`similarity.py`** — cosine matrix, kNN, anchor selection, and KMeans clade clustering.
3. **`feature_table.py`** — structured biological feature table compiling representation, sequence, evolutionary, and structural signals.
4. **`llm_scoring.py`** — pluggable endpoint client supporting Gemini, OpenAI, Claude, and offline fallback.
5. **`deterministic_validation.py`** — rank correlation coefficients (Spearman's Rho and Kendall's Tau) and outlier flagging.
6. **`motif_conservation.py`** — GxSxG active-site check, family column conservation score.
7. **`structure_scoring.py`** — pLDDT structural confidence mapping and contact packing density.
8. **`ranking.py`** — hybrid pre-fold and final ranking composite scoring.
9. **`report.py`** — scientific report generator with structured signals and LLM comments.
10. **`config.py`** — added `LlmSpec`, `RankingWeights`, `ReportSpec` and Pydantic validation.
11. **`config/target.yaml`** — added pluggable `llm` block and weights sections.
12. **Tests** — complete unit tests for all modules + end-to-end hybrid pipeline smoke test.

Infrastructure footprint: **no GPU required**, **no vector DB**, **no training labels**. The pipeline supports full offline deterministic validation.

---

## 7. Risks & limitations

* **ESM-2 has a max sequence length of ~1024.** Our candidates are ~295 aa, so
  fine. Multi-domain enzymes would need windowed pooling.
* **Anchor-set bias.** The composite is only as good as the anchor set. If
  the anchors are e.g. all bacterial lipases, mammalian-like candidates score
  lower. The config exposes the anchor query so this is auditable.
* **Equal-weighted heuristics are not calibrated.** The weights are
  defensible defaults, not optimal. They cannot be tuned without labels
  (that's the deal). Sensitivity analysis (vary weights ±20%, see if the
  top-3 changes) belongs in the report.
* **ESMFold pLDDT correlates with foldability, not stability.** We use it as
  a foldability proxy and are transparent about that. True thermostability
  prediction needs wet-lab labels.
* **No active learning.** Per the original `enzymes.docx` spec, that's a
  future enhancement and explicitly out of scope.

---

## 8. Practical implementation steps (what I'll do next)

In this order, with a tested commit per step:

1. Rewrite `requirements.txt` — drop XGBoost, add `umap-learn`.
2. Add `embeddings.py` — ESM-2 provider, parquet cache, anchor set builder.
3. Add `similarity.py` — cosine kNN + anchor `emb_sim(c)`.
4. Add `motif_conservation.py` — GxSxG, conservation@conserved-columns.
5. Add `structure_scoring.py` — pLDDT → [0,1], packing score vs benchmark.
6. Rewrite `ranking.py` — composite score, no training, transparent breakdown.
7. Add `report.py` — deterministic templated markdown report (no LLM).
8. Update `config.py` + `config/target.yaml` for new sections.
9. Replace tests: drop `test_wp3_ranking` (XGBoost), add unit tests for each
   new module + a composite end-to-end test on the fixture.
10. Refresh notebooks `03`, `04`, `05`, `06` to drive the new modules.
11. Run on the existing lipase pool (no need to redo WP1/WP2). Verify the
    top-3 changes meaningfully and pLDDT of the new top-3 is materially
    better than the placeholder-proxy top-3 we had (37 → ≥ 60 expected).

When this lands, the POC will be a working inference-only enzyme-engineering
platform that someone with no labels can run end-to-end on a laptop.
