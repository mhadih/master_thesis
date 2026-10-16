# Fixed-K comparison — best setup per model at K=4, scored with AMI (S03)

- **Date (UTC):** 2026-10-16
- **Source:** exp_055–061 pkls rescored (no reclustering); identical protocol to
  `fixed_k_comparison.md`, metric NMI → AMI
- **Shared protocol:** 62 students, UMAP-50 (cosine, L2-normalized), KMeans/BIRCH
  k=4 or DBSCAN/OPTICS constrained to exactly 4 non-noise clusters (noise
  allowed as extra label), equal-depth grade bins = 4 for all runs
  (`[50.17, 78.83, 98.92, 103.33]`), cosine silhouette.
- **Metric:** `sklearn.metrics.adjusted_mutual_info_score` (default arithmetic
  averaging, same convention as the NMI scores). All 7 NMIs reproduced to 4
  decimals before scoring, so the embedding→pkl mapping is verified, not assumed.
- **Why AMI:** NMI is uncorrected — random partitions of 62 students into 4
  uneven groups already score >0, so "high NMI" alone cannot evidence a
  code↔grade relation. AMI subtracts expected chance agreement (0 = random,
  negative = worse than random); it tests agreement, still not causality.

| Model (best setup) | Exp | Silhouette | Clusters (+noise) | AMI (k=4) | NMI (k=4) | Purity |
|---|---|---|---|---|---|---|
| CCBERT (OPTICS ms=7, xi=0.05) | 059 | 0.7132 | 4+12n [7,12,15,16] | **0.0896** | 0.1570 | 0.4516 |
| CodeT5 (KMeans k=4) | 055 | 0.3889 | 4 [10,14,17,21] | **0.0693** | 0.1237 | 0.4516 |
| CCT5 (BIRCH thr=0.0225, nc=4) | 061 | 0.4909 | 4 [9,14,15,24] | **0.0641** | 0.1197 | 0.4194 |
| CodeBERT (DBSCAN eps≈0.0222, ms=7) | 057 | 0.5888 | 4+9n [6,6,20,21] | **0.0525** | 0.1279 | 0.3871 |
| e5 (OPTICS ms=7, xi=0.01) | 056 | 0.3856 | 4+13n [11,11,12,15] | **0.0409** | 0.1106 | 0.4355 |
| GraphCodeBERT (KMeans k=4) | 058 | 0.6540 | 4 [10,14,15,23] | **0.0365** | 0.0932 | 0.4032 |
| CC2Vec (BIRCH thr=0.005, nc=4) | 060 | 0.2846 | 4 [9,14,19,20] | **−0.0192** | 0.0408 | 0.3548 |

## Reading notes (for the conclusion)

1. **Chance correction deflates everything 40–60%.** Best score falls 0.157 → 0.090;
   the whole field sits at AMI < 0.09 — barely above random. The honest thesis
   sentence is not "CCBERT tracks grades" but "no embedding tracks grades at k=4;
   CCBERT is the least-bad at essentially-chance level."
2. **Ranking reshuffles under correction.** CodeBERT drops 2nd → 4th (its NMI was
   the most chance-inflated: uneven 6/6/20/21 sizes + 9 noise); CodeT5 rises to
   2nd. NMI-ranked model selection at this n/k would have picked a ranking that
   chance correction overturns — the methodological justification for this rerun.
3. **CC2Vec goes negative (−0.019):** worse-than-random agreement. Its NMI 0.04
   looked like weak signal; AMI says noise. Report as such.
4. **Consistent with Exp-A, not redundant with it:** Exp-A proved separation
   quality differs significantly by embedding; this proves that even the
   better-separated spaces carry ~zero grade information at coarse resolution.
   Together: embeddings differ strongly in geometry, none in grade relevance.
5. **Limits (state explicitly):** AMI still measures partition agreement, not
   causation or even correlation direction; k=4 coarse bins cap all scores;
   noise-as-extra-label handling is inherited unchanged from the NMI runs.
