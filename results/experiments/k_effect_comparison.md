# k-effect comparison — NMI at k = 4, 5, 6, 7, open (S03)

- **Date (UTC):** 2026-09-29 (k=7 column added same day)
- **Source:** fixed-K reruns exp_055–061 (k=4), exp_062–068 (k=5), exp_069–075 (k=6), exp_076–082 (k=7), open-k originals.
- **Shared protocol:** 62 students, UMAP-50 (cosine, L2-normalized), equal-depth
  bins = k (k=4: `[50.17, 78.83, 98.92, 103.33]`; k=5: `[40.38, 60.05, 86.17,
  100.0, 103.33]`), cosine silhouette. DBSCAN/OPTICS constrained to exactly
  k non-noise clusters (noise allowed, treated ordinarily).

| Model (setup) | NMI k=4 | NMI k=5 | NMI k=6 | NMI k=7 | NMI open k | Shape |
|---|---|---|---|---|---|
| CodeT5 (KMeans) | 0.1237 (055) | 0.1273 (062) | 0.1505 (069) | 0.2734 (076) | 0.5356, k=17 (001) | slow climb, then jumps |
| e5 (OPTICS) | 0.1106 (056) | 0.1905 (063) | 0.1658 (070) | 0.2499 (077) | 0.5485, k=17 (008) | rises, dips, climbs, jumps |
| CodeBERT (DBSCAN) | 0.1279 (057) | 0.1507 (064) | 0.1937 (071) | 0.2186 (078) | 0.2085, k=7 (010)* | monotone climb |
| GraphCodeBERT (KMeans) | 0.0932 (058) | 0.1156 (065) | 0.1213 (072) | 0.2041 (079) | — (best is k=4) | gentle climb, k=7 step |
| CCBERT (OPTICS) | 0.1570 (059) | 0.1833 (066) | **0.2180** (073) | 0.2822 (080) | 0.1570, k=7 (020) | peak-dip-peak, best at 6–7 |
| CC2Vec (BIRCH) | 0.0408 (060) | 0.0740 (067) | 0.1148 (074) | 0.1638 (081) | 0.1148, k=6 (023) | monotone climb |
| CCT5 (BIRCH) | 0.1197 (061) | 0.1332 (068) | 0.2038 (075) | =k=6 (082†) | 0.2038, k=6 (027) | monotone climb, capped |

## Reading notes (for the conclusion)

1. **Four shapes, one lesson.** Monotone climbers (CodeBERT, CC2Vec, CCT5,
   GraphCodeBERT): finer cuts keep paying — report argmax-k for these.
   Peak-dip-peak (CCBERT: 0.157 → 0.183 → 0.218 → **0.282** → 0.157): k=7 is now
   CCBERT's best anywhere, but at silhouette 0.03 with 44% noise — compactness
   and grade relevance diverge completely here. Slow-climb-then-jump
   (CodeT5, e5): 4↔7 interpolation explains nothing about the k=17
   outliers — their signal lives *only* at fine granularity.
2. **Fixed-k compresses all models to NMI 0.04–0.28**, erasing the e5/CodeT5
   lead (their best fixed-k cells: e5 0.25 at k=7, CodeT5 0.27 at k=7).
   The model ranking is k-dependent, so every ranking claim in
   this thesis must name its k — the fixed_k_comparison.md (k=4) and this table
   jointly make that point.
3. **Purity falls as NMI rises almost everywhere** (e.g., CCT5 0.42→0.35→0.31):
   the two metrics trade off across k by construction — never report one without
   the other.

\* exp_010 ran repo-default euclidean/unnormalized space (all other cells use the
aligned cosine-normalized space); close but not strictly comparable (0.2085 vs.
fixed-space trajectory ending 0.2186 at k=7).
† exp_082 requested nc=7 but Birch returned 6 labels (nc is an upper bound when
subclusters ≤ nc); ARI=1.0 vs. exp_075, so the k=7 cell duplicates k=6.
