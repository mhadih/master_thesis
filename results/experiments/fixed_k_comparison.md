# Fixed-K comparison — best setup per model at K=4 (S03)

- **Date (UTC):** 2026-09-29
- **Source:** exp_055–061 (fixed-K reruns of exp_001/008/010/013/020/023/027)
- **Shared protocol:** 62 students, UMAP-50 (cosine, L2-normalized), KMeans/BIRCH
  k=4 or DBSCAN/OPTICS constrained to exactly 4 non-noise clusters (noise
  allowed as extra label), equal-depth grade bins = 4 for all runs
  (`[50.17, 78.83, 98.92, 103.33]`), cosine silhouette.
- **Why K=4:** modal winner-k across the 46 open-k runs (KMeans 7/13, BIRCH 7/11
  at k=4); ~15 students per bin (vs. ~3.6 at k=17); 4 interpretable bands.

| Model (best setup) | Exp | Silhouette | Clusters (+noise) | NMI (k=4) | NMI (open k) | Purity |
|---|---|---|---|---|---|---|
| CCBERT (OPTICS ms=7, xi=0.05) | 059 | 0.7132 | 4+12n [7,12,15,16] | **0.1570** | 0.1570 (k=7) | 0.4516 |
| CodeBERT (DBSCAN eps≈0.0222, ms=7) | 057 | 0.5888 | 4+9n [6,6,20,21] | **0.1279** | 0.2085 (k=7) | 0.3871 |
| CodeT5 (KMeans k=4) | 055 | 0.3889 | 4 [10,14,17,21] | **0.1237** | 0.5356 (k=17) | 0.4516 |
| CCT5 (BIRCH thr=0.0225, nc=4) | 061 | 0.4909 | 4 [9,14,15,24] | **0.1197** | 0.2038 (k=6) | 0.4194 |
| e5 (OPTICS ms=7, xi=0.01) | 056 | 0.3856 | 4+13n [11,11,12,15] | **0.1106** | 0.5485 (k=17) | 0.4355 |
| GraphCodeBERT (KMeans k=4) | 058 | 0.6540 | 4 [10,14,15,23] | **0.0932** | 0.0932 (k=4) | 0.4032 |
| CC2Vec (BIRCH thr=0.005, nc=4) | 060 | 0.2846 | 4 [9,14,19,20] | **0.0408** | 0.1148 (k=6) | 0.3548 |

## Reading notes (for the conclusion)

1. **Ranking flattens and reshuffles.** Open-k spread was 0.09–0.55 with e5/CodeT5
   dominant; at k=4 the spread is 0.04–0.16 with CCBERT on top — the celebrated
   e5/CodeT5 grade alignment lived almost entirely in fine granularity.
2. **Resolution-stable vs. resolution-fragile.** CCBERT identical to 4 decimals
   across 4/7 bins (0.1570); GraphCodeBERT identical too (0.0932, bit-reproducible
   control). CodeT5 (0.54→0.12) and e5 (0.55→0.11) collapse — their signal *is*
   fine structure.
3. **Purity inverts honestly:** now highest where NMI is middling (CodeT5 0.45,
   CCBERT 0.45) — coarse-bin majority votes, reported as such, not as agreement.
4. **Method note for replicability:** DBSCAN repo script stores total labels incl.
   noise as best_k; fixed runs store the non-noise count so eval bins = 4
   (one-line change, documented in exp_057).
