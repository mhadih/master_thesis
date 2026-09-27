# Experiment 034 — CC2Vec + frozen-LSTM mean-hidden file aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-26
- **ID:** `exp_034`
- **Status:** executed; sequence-aware ablation (order-sensitive, recency-balanced), 3 seeds
- **Varies vs. exp_021/033:** file_emb = mean over LSTM hidden states (not last); rest identical

## 1. Setup

Same store, frozen LSTM (hidden 64, seeds 0, 1, 2, last-200 truncation), UMAP-50,
KMeans sweep, equal-depth eval as exp_033 — only the readout differs (mean vs. last).

Artifacts: `../models/exp_034_best_clustering_s{0,1,2}.pkl`, `../figures/exp_034_umap_kmeans_k7.png` (seed 2).

## 2. Results (per seed)

| Seed | k | Cluster sizes | Silhouette | NMI | Purity |
|---|---|---|---|---|---|
| 0 | 4 | [7, 15, 20, 20] | 0.8134 | 0.0754 | 0.4032 |
| 1 | 4 | [9, 16, 17, 20] | 0.8810 | 0.0867 | 0.3871 |
| 2 | 7 | [6, 7, 8, 8, 9, 10, 14] | 0.7526 | 0.1780 | 0.3065 |
| **mean ± std** | — | — | **0.816 ± 0.052** | **0.113 ± 0.047** | 0.366 ± 0.042 |

- Grade-bin borders: k=4 → `[50.17, 78.83, 98.92, 103.33]`; k=7 (seed 2) →
  `[29.04, 50.58, 60.05, 84.57, 95.0, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **Seed 2 is the interesting outlier:** it alone selects k=7 and reaches NMI 0.178
  — best LSTM result, second-best CC2Vec overall after exp_029 (0.301). But with
  n=3 seeds the NMI spread (±0.047, driven by one seed) is too wide to claim a
  real win: report mean ± std (0.113 ± 0.047), not the max. If this direction is
  pursued, it needs 10+ seeds.
- Mean-readout keeps extreme compactness (sil 0.75–0.88, slightly below last-readout's
  0.85–0.92) with the same flat NMI band — recency balancing does not rescue
  grade relevance either.
- Combined with exp_033: neural recency (last) vs. balanced (mean) differ in
  compactness, not in grade signal. The sequence-ablation series as a whole
  (029–034) says: *how* you pool the trajectory moves NMI only via importance
  weighting (edit-size), never via order.

## 4. Reproduce

As exp_033 (`user_CC2Vec_abl_lstm_mean_s{seed}.jsonl`, sandbox s03run_seq/lstm_mean_s{seed} pattern).
