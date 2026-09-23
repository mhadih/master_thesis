# Experiment 031 — CC2Vec + ignore-one-line-changes aggregation + KMeans (S03)

- **Date (UTC):** 2026-09-23
- **ID:** `exp_031`
- **Status:** executed; order-invariance ablation 3/3
- **Varies vs. exp_021:** file-level aggregation only (drop changes with added+removed ≤ 1 line, then mean); rest identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Embeddings | **CC2Vec** per-change vectors, same store as exp_029 | `/tmp/opencode/s03run_abl/cc2v_file_changes.pt` |
| File aggregation | **Ignore one-line changes**: typo-level edits (17.0% of all changes) removed before averaging; 16/1160 files left empty and skipped | `user_CC2Vec_abl_no1line.jsonl` (65 users) |
| Students evaluated | **62** (same graded set) | `gradesheets/user_grades.csv` |
| Dimension reduction / clustering / eval | UMAP-50 cosine + KMeans sweep + equal-depth (k = 4) | same scripts |

Artifacts: `../models/exp_031_best_clustering.pkl`, `../figures/exp_031_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_031 (no-1-line) | exp_021 (plain mean) |
|---|---|---|
| Best k | **4** | 5 |
| Silhouette | 0.3067 | 0.2987 |
| NMI | 0.0447 | 0.0764 |
| Purity | 0.3387 | 0.3226 |

- Cluster sizes: `[11, 11, 18, 22]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **One-line edits are load-bearing after all:** removing just 17% of changes
  *hurts* NMI (0.076 → 0.045) instead of denoising — typo-level edits apparently
  carry authorial signal (e.g., careful vs. sloppy editing traces). Do not discard
  small edits; weight them (cf. exp_029's success) rather than filtering.
- Combined ablation moral: *re-weight* changes (029: NMI 0.30) ≫ *recency*
  (030: 0.05) ≫ *filtering small edits* (031: 0.04). The order-invariance problem
  is better solved by importance weighting than by subsetting.

## 4. Reproduce

As exp_029 (sandbox a31 pattern, `user_CC2Vec_abl_no1line.jsonl`).
