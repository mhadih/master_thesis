# Experiment 032 — CC2Vec + decay-weighted (γ=0.9) file aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-26
- **ID:** `exp_032`
- **Status:** executed; sequence-aware ablation (closed-form recency)
- **Varies vs. exp_021:** file-level aggregation only (plain mean → positional decay
  `wᵢ ∝ 0.9^(C-1-i)`); model, UMAP-50, KMeans sweep, equal-depth eval identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CC2Vec** per-change vectors, same store as exp_029 | `/tmp/opencode/s03run_abl/cc2v_file_changes.pt` (durable copy: `data/cc2v_file_changes.pt`, gitignored) |
| File aggregation | **Decay weighting**, γ=0.9 over chronological changes | `user_CC2Vec_abl_decay.jsonl` (65 users) via `src/aggregation/cc2vec_sequence.py` |
| Students evaluated | **62** (same graded set) | `gradesheets/user_grades.csv` |
| Rest | UMAP-50 cosine + KMeans sweep + equal-depth (k = 4) | same scripts |

Artifacts: `../models/exp_032_best_clustering.pkl`, `../figures/exp_032_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_032 (decay γ=0.9) | exp_021 (plain mean) |
|---|---|---|
| Best k | **4** | 5 |
| Silhouette | 0.2939 | 0.2987 |
| NMI | 0.0434 | 0.0764 |
| Purity | 0.3548 | 0.3226 |

- Cluster sizes: `[12, 16, 17, 17]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Smooth recency weighting changes almost nothing** (NMI 0.043 vs. 0.076): with
  γ=0.9 over median-21-change files, effective weights stay flat enough that the
  centroid barely moves — consistent with the near-identical within-file vectors
  (§3 of the ablation table doc). Recency needs a hard cutoff (cf. exp_030's
  last-5, sil 0.38) to bite, and even then it sharpens clusters without adding
  grade signal.
- Decay is fully reproducible (zero parameters beyond γ); report as the null
  recency model against which exp_030/033 are judged.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python src/aggregation/cc2vec_sequence.py  # needs ABL_STORE
# then KMeans + eval on user_CC2Vec_abl_decay.jsonl (sandbox s03run_seq/decay pattern)
```
