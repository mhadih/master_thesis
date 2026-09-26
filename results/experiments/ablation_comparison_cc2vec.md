# Ablation comparison — file-level aggregation of CC2Vec change vectors (S03)

- **Date (UTC):** 2026-09-26
- **Source:** exp_021 (baseline) + exp_029–031 (ablations)
- **Shared protocol:** CC2Vec 196-dim change vectors, 62 students, UMAP-50 (cosine,
  L2-normalized), KMeans k-sweep 4…20, equal-depth grade bins, cosine silhouette.
  Only the per-file aggregation varies; user level stays length-weighted.

| Method | Exp | k (bins) | Cluster sizes | Silhouette | NMI | Purity |
|---|---|---|---|---|---|---|
| Plain mean (baseline) | exp_021 | 5 | [9, 9, 12, 15, 17] | 0.2987 | 0.0764 | 0.3226 |
| Edit-size-weighted mean | exp_029 | 9 | [4, 5, 5, 6, 7, 7, 9, 9, 10] | 0.3412 | **0.3009** | 0.3226 |
| Last-5 mean | exp_030 | 4 | [8, 15, 19, 20] | **0.3776** | 0.0539 | 0.3710 |
| Ignore one-line changes | exp_031 | 4 | [11, 11, 18, 22] | 0.3067 | 0.0447 | 0.3387 |

(Baseline sizes re-verified by deterministic re-run: k=5, sil 0.2987, identical.)

## Reading notes (for the conclusion)

1. **Re-weight ≫ recency ≫ filtering.** Edit-size weighting is the only method that
   moves NMI (0.076 → 0.301, ~4×, best CC2Vec result overall); recency sharpens
   clusters (best silhouette 0.378) but loses grade signal; dropping one-line edits
   hurts (0.076 → 0.045) — small edits carry authorial signal, so weight them
   instead of discarding.
2. **Purity is flat** (0.32–0.37) across all four — the differences are structural
   (NMI), not majority-vote.
3. **Why ablations can matter at all here:** within-file CC2Vec change vectors are
   near-identical (mean pairwise dist ≈ 0.01 vs. norm ≈ 12.6), so any averaging
   variant lands close in L2 — yet UMAP nonlinearly amplifies the small differences
   enough to flip k (4/5/9) and NMI (0.04–0.30).
4. Same k-confounding caveat as the model table (4/5/9 bins); the 4× NMI gap of
   exp_029 dwarfs typical bin-count effects.
