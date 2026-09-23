# Experiment 029 — CC2Vec + edit-size-weighted file aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-23
- **ID:** `exp_029`
- **Status:** executed; order-invariance ablation 1/3 (see §3 for rationale)
- **Varies vs. exp_021:** file-level aggregation only (plain mean → edit-size-weighted mean); model, UMAP-50, KMeans sweep, equal-depth eval identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CC2Vec** per-change vectors (196-dim), same extraction as exp_021 | `/tmp/opencode/s03run_abl/cc2v_file_changes.pt` (1160 files; change vectors saved per file with added/removed line counts) |
| File aggregation | **Edit-size-weighted mean**: `file_emb = Σ wᵢvᵢ / Σ wᵢ`, `wᵢ = addedᵢ + removedᵢ` lines per change (all pairs have ≥1 changed line by construction) | `user_CC2Vec_abl_editsize.jsonl` (65 users) |
| Students evaluated | **62** (same graded set as exp_021) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, cosine, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` |
| Clustering | **KMeans** k 4…20, cosine silhouette | same script |
| Grade discretization | **Equal-depth**, k = 9 | `src/evaluation/evaluation.py` |

Artifacts: `../models/exp_029_best_clustering.pkl`, `../figures/exp_029_umap_kmeans_k9.png`.

## 2. Results

| Metric | exp_029 (edit-size-w) | exp_021 (plain mean) |
|---|---|---|
| Best k | **9** | 5 |
| Silhouette | **0.3412** | 0.2987 |
| NMI | **0.3009** | 0.0764 |
| Purity | 0.3226 | 0.3226 |

- Cluster sizes: `[4, 5, 5, 6, 7, 7, 9, 9, 10]`.
- Grade-bin borders (9 bins): `[27.12, 40.38, 52.75, 60.05, 78.83, 86.17, 96.7, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **Weighting by edit size is the single biggest CC2Vec win in the series:** NMI
  0.076 → 0.301 (4×), best NMI on this embedding by far. Giving large edits more
  voice recovers grade-relevant structure that uniform averaging dilutes — direct
  evidence that *which* changes count matters more than *how* they are encoded here.
- Purity flat (0.3226 both) — the gain is structural (NMI), not majority-vote.
- Caveat: k differs (9 vs. 5 bins); the fixed-bin comparison item still stands, but
  a 4× NMI gap dwarfs the usual bin-count effects seen in exp_001–028.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_abl/aggregate.py  # needs cc2v_file_changes.pt
# then KMeans + eval on user_CC2Vec_abl_editsize.jsonl (sandbox a29 pattern)
```
