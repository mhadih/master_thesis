# Experiment 050 — CodeT5 function-level + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_050`
- **Status:** full pipeline — DB-generated function-level CodeT5 embeddings (see exp_047 §1)
- **Varies vs. exp_004:** embedding protocol only (whole-file → function-level); all else identical

## 1. Setup

Same embeddings (67 users × 768-dim), students (62, user 143 absent), UMAP-50 cosine
+ L2-normalize, and equal-depth eval as exp_047. Clustering: **OPTICS**
(`cluster_method='xi'`) grid (min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine
silhouette) via `clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_ct5f/op/`. Artifacts: `../models/exp_050_best_clustering.pkl`,
`../figures/exp_050_umap_optics.png`.

## 2. Results

| Metric | exp_050 (function-level) | exp_004 (whole-file) |
|---|---|---|
| Best params | **(ms=5, xi=0.01, inf)** | (ms=5, xi=0.01, inf) |
| Silhouette | 0.6360 | 0.4663 |
| Clusters | **6 + 3 noise** (script stores best_k=6, counting noise); cluster sizes `[6, 6, 8, 8, 13, 18]`, noise 3 | 6 + 12 noise |
| NMI | 0.1645 | 0.2231 |
| Purity | 0.3065 | 0.3387 |

- Grade-bin borders (6 bins): `[34.66, 55.39, 78.83, 95.0, 101.17, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- Identical operating point (ms=5, xi=0.01), closest protocol pair in the CodeT5
  comparison (NMI 0.16 vs. 0.22): OPTICS is the least protocol-sensitive algorithm
  here — both readings weak, both honest. Thinnest noise fringe in this pair
  (3 vs. 12).
- With this, every model family now has a same-model protocol pair: CodeT5 ±,
  CodeBERT ±, E5 ± (GraphCodeBERT whole-file only + function script ready).

## 4.–5. Blocked steps / reproduce

As exp_047 (replace `km/kmeans_ct5f.py` with `op/optics_ct5f.py`).
