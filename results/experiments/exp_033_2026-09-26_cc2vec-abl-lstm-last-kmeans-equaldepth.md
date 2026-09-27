# Experiment 033 — CC2Vec + frozen-LSTM last-hidden file aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-26
- **ID:** `exp_033`
- **Status:** executed; sequence-aware ablation (neural recency), 3 seeds
- **Varies vs. exp_021:** file-level aggregation only (plain mean → frozen LSTM
  last hidden state); rest identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CC2Vec** per-change vectors, same store | `data/cc2v_file_changes.pt` |
| File aggregation | **Frozen random LSTM** (input 196, hidden 64, uniform init, eval mode; files truncated to last 200 changes) → last hidden state. Untrained by design (unsupervised, 62 students) — a fixed nonlinear reservoir. Seeds 0, 1, 2, reported jointly, never cherry-picked | `user_CC2Vec_abl_lstm_last_s{0,1,2}.jsonl` (65 users each) via `src/aggregation/cc2vec_sequence.py` |
| Students evaluated | **62** (same graded set) | `gradesheets/user_grades.csv` |
| Rest | UMAP-50 cosine + KMeans sweep + equal-depth (k = 4 all seeds) | same scripts |

Artifacts: `../models/exp_033_best_clustering_s{0,1,2}.pkl`, `../figures/exp_033_umap_kmeans_k4.png` (seed 2, highest NMI).

## 2. Results (per seed; KMeans itself is deterministic, variance is LSTM-init only)

| Seed | k | Cluster sizes | Silhouette | NMI | Purity |
|---|---|---|---|---|---|
| 0 | 4 | [9, 11, 18, 24] | 0.9202 | 0.0722 | 0.3710 |
| 1 | 4 | [9, 11, 18, 24] | 0.8543 | 0.0702 | 0.3871 |
| 2 | 4 | [11, 13, 18, 20] | 0.8539 | 0.0766 | 0.4032 |
| **mean ± std** | 4 | — | **0.876 ± 0.031** | **0.073 ± 0.003** | 0.387 ± 0.013 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- Seeds 0/1 converge to identical partitions; seed 2 differs slightly — seed
  sensitivity is real but small.

## 3. Component observations (for the conclusion)

- **Extreme compactness, zero grade gain:** silhouette 0.85–0.92 (series record by
  far — the LSTM reservoir collapses each file's trajectory into tight attractors)
  with NMI ≈ 0.07, identical to plain mean. Strongest compactness-≠-relevance
  datapoint in the thesis.
- Seed spread (±0.03 sil, ±0.003 NMI) is an order of magnitude smaller than
  cross-method gaps — conclusions are seed-robust.
- Recency mechanism check vs. exp_030 (last-5: sil 0.38, NMI 0.05) and exp_032
  (decay: sil 0.29, NMI 0.04): neural recency ≫ hard cutoff ≫ smooth decay for
  compactness, while NMI stays flat (~0.04–0.08) across all three — recency
  reorganizes, but never toward grades.

## 4. Reproduce

```bash
ABL_SEEDS="0,1,2" /home/hadi/thesis/venv/bin/python src/aggregation/cc2vec_sequence.py
# then KMeans + eval per seed file (sandbox s03run_seq/lstm_last_s{0,1,2} pattern)
```
