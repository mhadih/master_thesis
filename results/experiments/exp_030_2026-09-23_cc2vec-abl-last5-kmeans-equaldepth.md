# Experiment 030 — CC2Vec + last-5-changes file aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-23
- **ID:** `exp_030`
- **Status:** executed; order-invariance ablation 2/3
- **Varies vs. exp_021:** file-level aggregation only (plain mean over all changes → mean over final 5); rest identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CC2Vec** per-change vectors, same store as exp_029 | `/tmp/opencode/s03run_abl/cc2v_file_changes.pt` |
| File aggregation | **Last-5 mean**: `file_emb = mean(v[-5:])` — final polishing phase only. K=5 chosen because the median file has 21 changes, so the last 5 ≈ its final quarter; files with <5 changes contribute all of theirs | `user_CC2Vec_abl_last5.jsonl` (65 users) |
| Students evaluated | **62** (same graded set) | `gradesheets/user_grades.csv` |
| Dimension reduction / clustering / eval | UMAP-50 cosine + KMeans sweep + equal-depth (k = 4) | same scripts |

Artifacts: `../models/exp_030_best_clustering.pkl`, `../figures/exp_030_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_030 (last-5) | exp_021 (plain mean) |
|---|---|---|
| Best k | **4** | 5 |
| Silhouette | **0.3776** | 0.2987 |
| NMI | 0.0539 | 0.0764 |
| Purity | **0.3710** | 0.3226 |

- Cluster sizes: `[8, 15, 19, 20]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Recency sharpens clusters but not grade signal:** best silhouette of the three
  ablations (0.3776 > 0.3412 > 0.3067) — end-state code is more mutually separable —
  yet NMI drops slightly (0.054 vs. 0.076). Final snapshots discriminate *style*,
  not *grades*.
- Purity uptick (0.371) is a 4-bin coarseness effect, consistent with the series.

## 4. Reproduce

As exp_029 (sandbox a30 pattern, `user_CC2Vec_abl_last5.jsonl`).
