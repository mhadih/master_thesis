# Model comparison — best NMI per embedding model (S03)

- **Date (UTC):** 2026-09-26
- **Source:** exp_001–028 (standard 4-algorithm suites) + exp_029–031 (CC2Vec ablations)
- **Shared protocol:** 62 graded students, UMAP-50 (cosine, L2-normalized), equal-depth
  grade bins with bin count = winner's cluster count, cosine silhouette.

| # | Model (dim) | Best exp | Winning setup | k (bins) | Silhouette | NMI | Purity |
|---|---|---|---|---|---|---|---|
| 1 | multilingual-e5 (768) | exp_008 | OPTICS (ms=2, xi=0.01), 17 clusters + 4 noise | 17 | 0.4603 | **0.5485** | 0.3710 |
| 2 | CodeT5 (768) | exp_001 | KMeans, k=17 | 17 | 0.5179 | **0.5356** | 0.3548 |
| 3 | CodeBERT (768) | exp_010 | DBSCAN (eps≈0.019, ms=5), 6 clusters + 3 noise | 7 | 0.6479 | **0.2085** | 0.3387 |
| 4 | CCT5 (768) | exp_027 | BIRCH (thr=0.0225, nc=6) | 6 | 0.5347 | **0.2038** | 0.3065 |
| 5 | CCBERT (512) | exp_020 | OPTICS (ms=7, xi=0.05), 3 clusters + 12 noise | 4 | 0.7132 | **0.1570** | 0.4516 |
| 6 | CC2Vec (196) | exp_023 | BIRCH (thr=0.0125, nc=6) | 6 | 0.3070 | **0.1148** | 0.3065 |
| 7 | GraphCodeBERT (768) | exp_013 | KMeans, k=4 | 4 | 0.6540 | **0.0932** | 0.4032 |

## Reading notes (for the conclusion)

1. **Bin-count confounding.** Winners use 4–17 bins and both metrics respond to bin
   count: CCBERT's 0.4516 and GraphCodeBERT's 0.4032 purity are inflated by coarse
   4-bin setups, while e5/CodeT5 earned ~0.54 NMI at the hardest resolution
   (17 bins). A fixed-bin re-evaluation is open future work.
2. **CC2Vec asterisk.** Table shows the best *standard* setup (exp_023, 0.1148);
   the edit-size-weighted ablation exp_029 reaches NMI **0.3009** (KMeans k=9,
   sil 0.3412) — an aggregation finding, not a model finding.
3. **Headline pattern.** Snapshot models split (e5/CodeT5 ≈ 0.54 vs.
   CodeBERT/GraphCodeBERT ≈ 0.1–0.2); all change-sequence models sit at 0.02–0.20
   (CCBERT 0.1570, CCT5 0.2038, CC2Vec 0.1148/0.3009-abl). Yet change models own
   the silhouette ranking (0.63–0.71 vs. 0.46–0.53): compactness ≠ grade relevance.
4. **No universal algorithm winner.** Best-per-model algorithms span all four
   (KMeans ×2, OPTICS ×2, DBSCAN ×1, BIRCH ×2); OPTICS is the most
   embedding-sensitive (NMI 0.04–0.55 across embeddings).
