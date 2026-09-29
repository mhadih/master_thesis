# Experiment 054 — CCT5 flow-conditioned + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_054`
- **Status:** partial — Kaggle-generated flow embeddings (see exp_051 §1)
- **Varies vs. exp_028:** embedding view only (diff-string → DFG-conditioned); all else identical

## 1. Setup

Same embeddings (65 users × 768-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_051. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_cct5f/op/`. Artifacts: `../models/exp_054_best_clustering.pkl`,
`../figures/exp_054_umap_optics.png`.

## 2. Results

| Metric | exp_054 (flow view) | exp_028 (diff view) |
|---|---|---|
| Best params | **(ms=9, xi=0.01, inf)** | (ms=9, xi=0.01, inf) |
| Silhouette | 0.4916 | 0.4631 |
| Clusters | **3 + 4 noise** (script stores best_k=3, counting noise); sizes `[4(noise), 14, 22, 22]` | 3 + 18 noise |
| NMI | 0.0511 | 0.0551 |
| Purity | **0.4516** | 0.4839 |

- Grade-bin borders (3 bins): `[55.39, 95.0, 103.33]` — identical to exp_028 (same k).
- NMI/purity treat noise as an ordinary label.
- Identical operating point (ms=9, xi=0.01) with a far thinner noise fringe
  (4 vs. 18): flow space is denser at the same density threshold.

## 3. Component observations (for the conclusion)

- NMI identical to two decimals (0.051 vs. 0.055): OPTICS cannot distinguish the
  two CCT5 views grade-wise — the protocol-invariant null, mirroring BIRCH on
  GraphCodeBERT (exp_015/041).
- Purity 0.45 on 3 bins is the coarseness artifact documented since exp_012.
- OPTICS stays the most embedding-sensitive algorithm overall (NMI 0.04–0.66);
  within CCT5 it is view-insensitive.

## 4.–5. Blocked steps / reproduce

As exp_051 (replace `km/kmeans_cf.py` with `op/optics_cf.py`).
