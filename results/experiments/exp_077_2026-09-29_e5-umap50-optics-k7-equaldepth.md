# Experiment 077 — e5 + UMAP-50 + OPTICS (7 clusters, fixed) + equal-depth grades (S03)

- **Date (UTC):** 2026-09-29
- **ID:** `exp_077`
- **Status:** executed — K=7 counterpart of exp_056/063/070
- **Varies vs. exp_070:** exactly-6 → exactly-7 non-noise clusters; all else identical

## Setup

e5 embeddings (68 × 768-dim), 62 students, UMAP-50 cosine + L2-normalize, OPTICS
`xi` grid constrained to 7 non-noise clusters (noise allowed), cosine
silhouette, equal-depth (7 bins). Sandbox: `/tmp/opencode/s03run_fx7/i77/`.

Artifacts: `../models/exp_077_best_clustering.pkl`, `../figures/exp_077_umap_optics_k7.png`.

## Results (k = 4 → 5 → 6 → 7 → 17)

| k | Winner (ms, xi) | Silhouette | NMI | Purity | Source |
|---|---|---|---|---|---|
| 4 | (7, 0.01) | 0.3856 | 0.1106 | 0.4355 | exp_056 |
| 5 | (5, 0.01) | 0.2664 | 0.1905 | 0.4194 | exp_063 |
| 6 | (4, 0.1) | 0.2026 | 0.1658 | 0.3387 | exp_070 |
| 7 | (4, 0.05) | 0.3050 | **0.2499** | 0.3871 | exp_077 |
| 17 | (2, 0.01) | 0.4603 | 0.5485 | 0.3710 | exp_008 |

- Clusters at k=7: 7 + 23 noise `[4, 5, 5, 5, 6, 7, 7]`, noise 23 (37%).

## Reading

Monotone-ish climb with a k=6 dip (0.11 → 0.19 → 0.17 → 0.25): e5 keeps paying
for fineness despite ballooning noise (13 → 19 → 30 → 23). The operating point
wanders ((7,0.01) → (5,0.01) → (4,0.1) → (4,0.05)) — no stable density regime,
consistent with a space whose "clusters" are threshold artifacts as much as
structure. Caveat for citing exp_008: its 0.55 rests on the same fragile regime.
