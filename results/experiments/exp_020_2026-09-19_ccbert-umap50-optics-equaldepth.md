# Experiment 020 — CCBERT + UMAP-50 + OPTICS + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_020`
- **Status:** full pipeline — fresh CCBERT embeddings (see exp_017 §1)
- **Varies vs. exp_004/008/012/016:** embedding model only (→ CCBERT); all else identical

## 1. Setup

Same embeddings (66 users × 512-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_017. Clustering: **OPTICS** (`cluster_method='xi'`) grid
(min_samples 2…9 × xi [0.01, 0.05, 0.1, 0.2], cosine silhouette) via
`clustering/optics_clustering.py` + alignment patch.

Run sandbox: `/tmp/opencode/s03run_ccb/op/`. Artifacts: `../models/exp_020_best_clustering.pkl`,
`../figures/exp_020_umap_optics.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (min_samples, xi, max_eps)** | **(7, 0.05, inf)** |
| **Silhouette (cosine)** | **0.7132** |
| **Clusters** | **3 + 12 noise** (script stores best_k=4, counting noise); sizes `[7, 12(noise), 15, 16]` — the 3 real clusters match DBSCAN/BIRCH exactly (ARI = 0.79 vs. DBSCAN incl. noise) |
| **NMI** (vs. 4 equal-depth grade bins) | **0.1570** |
| **Purity** | **0.4516** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- OPTICS peels 12 students off as noise from the same 4-group skeleton the other
  three algorithms agree on — consensus structure with a soft fringe. NMI 0.1570 is
  the best on this embedding but still far below CodeT5/e5 levels.
- OPTICS stays the most embedding-sensitive algorithm (NMI 0.09–0.55), CCBERT sitting
  at its low end with CodeBERT.

## 4.–5. Blocked steps / reproduce

As exp_017 (replace `km/kmeans_ccb.py` with `op/optics_ccb.py`).
