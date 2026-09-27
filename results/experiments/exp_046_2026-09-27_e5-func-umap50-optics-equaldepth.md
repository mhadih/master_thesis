# Experiment 046 — E5 function-level + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-27
- **ID:** `exp_046`
- **Status:** full pipeline — DB-generated function-level E5 embeddings (see exp_043 §1)
- **Varies vs. exp_008:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_043. Clustering: **OPTICS**
(`cluster_method='xi'`) grid (min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine
silhouette) via `clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_ef/op/`. Artifacts: `../models/exp_046_best_clustering.pkl`,
`../figures/exp_046_umap_optics.png`.

## 2. Results

| Metric | exp_046 (function-level) | exp_008 (whole-file) |
|---|---|---|
| Best params | **(ms=8, xi=0.01, inf)** | (ms=2, xi=0.01, inf) |
| Silhouette | 0.5174 | 0.4603 |
| Clusters | **4 + 13 noise** (script stores best_k=4, counting noise); cluster sizes `[8, 10, 12, 19]`, noise 13 | 17 + 4 noise |
| NMI | 0.0970 | **0.5485** |
| Purity | 0.4355 | 0.3710 |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- **The protocol swing that crowns the series argument:** OPTICS/e5 drops from the
  series-best NMI (0.5485, 17 fine clusters) to 0.0970 (4 coarse clusters + heavy
  noise) purely by switching whole-file → function-level. Same model, same
  students, same pipeline — input granularity decides everything.
- min_samples jumps 2 → 8: function space needs much denser cores to form groups,
  and only 4 survive.
- With this, every model family has a same-model protocol pair except GraphCodeBERT
  (function-only): CodeT5 n/a (file-only), CodeBERT ±, E5 ±, CCBERT/CC2Vec/CCT5 n/a
  (change-only by nature).

## 4.–5. Blocked steps / reproduce

As exp_043 (replace `km/kmeans_ef.py` with `op/optics_ef.py`).
