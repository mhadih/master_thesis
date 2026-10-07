# Experiment 088 — CodeBERT whole-file + t-SNE-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-10-07
- **ID:** `exp_088`
- **Status:** executed — reducer variant (see reducer_comparison_dim50.md for the head-to-head)
- **Varies vs. exp_038:** reducer only (UMAP-50 → t-SNE-50, exact method, cosine, perplexity 30); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CodeBERT whole-file** (68 × 768-dim), 62 students | `student_embeddings/user_CodeBert_file_embeddings.jsonl` |
| Reduction | **t-SNE-50** (exact, cosine, perplexity 30 < n, `random_state=42`), then L2-normalize — off-label use (t-SNE is designed for 2–3D), included for completeness | `/tmp/opencode/s03run_red/reducer_sweep.py` |
| Clustering | **OPTICS** `xi` grid (ms 2…9 × xi [0.01, 0.05, 0.1, 0.2]), best cosine silhouette subject to ≥4 non-noise clusters | same as exp_038 |
| Eval | equal-depth, bins = winner's k (=8) | same script family |

Artifacts: `../models/exp_088_best_clustering.pkl` (no figure — cluster plot omitted for noise-dominated partitions).

## 2. Results

| Metric | exp_088 (t-SNE) | exp_038 (UMAP) |
|---|---|---|
| Best params | **(ms=2, xi=0.01, inf)** | (ms=2, xi=0.01, inf) |
| Silhouette | **−0.0192** | 0.3903 |
| Clusters | **8 + 31 noise**; sizes `[2, 2, 2, 3, 4, 4, 5, 9]`, noise 31 (50%) | 20 + 6 noise |
| NMI (8 bins) | **0.2934** | 0.6641 |
| Purity | 0.3226 | 0.4194 |

- Grade-bin borders (8 bins): same equal-depth rule (k=8).

## 3. Component observations (for the conclusion)

- Negative silhouette yet NMI 0.29: t-SNE-50 preserves neighborhood structure
  grades partially respect, without forming compact groups — compactness-≠-relevance
  with the sign flipped vs. the CCBERT pattern.
- Half the students labeled noise: density methods on t-SNE space find fragments,
  not communities. Same operating point wins both reducers ((2, 0.01)) — the grid
  agrees on where to look, the spaces differ in what is there.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_red/reducer_sweep.py  # t-SNE section
```
