# Experiment 022 — CC2Vec + UMAP-50 + DBSCAN + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_022`
- **Status:** full pipeline — fresh CC2Vec embeddings (see exp_021 §1)
- **Varies vs. exp_002/006/010/014/018:** embedding model only (→ CC2Vec); all else identical

## 1. Setup

Same embeddings (65 users × 196-dim), students (62), UMAP-50 cosine + L2-normalize,
and equal-depth eval as exp_021. Clustering: **DBSCAN** grid (20 eps × min_samples
2…9, <4-cluster skip, cosine silhouette) via `clustering/dbscan_clustering.py` +
alignment patch.

Run sandbox: `/tmp/opencode/s03run_cc2v/db/`. Artifacts: `../models/exp_022_best_clustering.pkl`,
`../figures/exp_022_umap_dbscan.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best params (eps, min_samples)** | **(0.0197, 5)** |
| **Silhouette (cosine)** | **0.2334** |
| **Clusters** | **3 + 18 noise** (script stores best_k=4, counting noise); sizes `[12, 15, 17, 18(noise)]` — 29% noise |
| **NMI** (vs. 4 equal-depth grade bins) | **0.0223** ← series floor |
| **Purity** | **0.3387** |

- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.
- NMI/purity treat noise as an ordinary label.

## 3. Component observations (for the conclusion)

- NMI 0.0223 ≈ chance level — the weakest grade agreement in all 24 experiments.
  Combined with exp_021: CC2Vec space has neither compactness nor grade signal.
- Unlike GraphCodeBERT/CCBERT, no DBSCAN≡BIRCH agreement here (BIRCH finds 6
  clusters, exp_023) — the space is unstructured enough that algorithms disagree.

## 4.–5. Blocked steps / reproduce

As exp_021 (replace `km/kmeans_cc2v.py` with `db/dbscan_cc2v.py`).
