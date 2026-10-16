# Ablation comparison — CC2Vec aggregation at fixed k=4 (S03)

- **Date (UTC):** 2026-10-16
- **Source:** fresh rerun, no reclustering inherited — all 11 embeddings clustered
  identically (`/tmp/opencode/abl_k4.py`, sandbox)
- **Shared protocol:** CC2Vec 196-dim change vectors, 62 students, UMAP-50 (cosine,
  L2-normalized), **KMeans k=4 fixed** (`random_state=42`), cosine silhouette,
  equal-depth grade bins = 4 (`[50.17, 78.83, 98.92, 103.33]`), NMI + AMI + purity.
  Only the per-file aggregation varies; user level stays length-weighted.
- **Why:** the `ablation_comparison_cc2vec.md` table mixes k=4/5/7/9, so its
  headline (edit-size NMI 0.30) may be a bin-count artifact. This is the fair setup.

| Method | Silhouette | NMI (k=4) | AMI (k=4) | Purity | Cluster sizes |
|---|---|---|---|---|---|
| Plain mean (baseline) | 0.2233 | 0.0545 | −0.0039 | 0.3548 | [10, 16, 17, 19] |
| Edit-size-weighted mean | 0.2813 | 0.0055 | **−0.0556** | 0.2742 | [11, 15, 16, 20] |
| Last-5 mean | 0.3776 | 0.0539 | −0.0058 | 0.3710 | [8, 15, 19, 20] |
| Ignore one-line changes | 0.3067 | 0.0447 | −0.0151 | 0.3387 | [11, 11, 18, 22] |
| Decay weighting (γ=0.9) | 0.2939 | 0.0434 | −0.0149 | 0.3548 | [12, 16, 17, 17] |
| Frozen-LSTM last-hidden (3 seeds) | 0.876±0.031 | 0.073±0.003 | 0.014±0.004 | 0.387±0.013 | s0/s1 [9,11,18,24]; s2 [11,13,18,20] |
| Frozen-LSTM mean-hidden (3 seeds) | 0.805±0.066 | 0.075±0.010 | 0.017±0.010 | 0.381±0.020 | s0 [7,15,20,20]; s1 [9,16,17,20]; s2 [13,14,16,19] |

## Reading notes (for the conclusion)

1. **The edit-size win evaporates.** 0.301 NMI at k=9 → NMI 0.006 / AMI −0.056 at
   k=4 — worse than random, the worst of all seven. The old table's note 4
   ("k-confounding caveat… dwarfs typical bin-count effects") turns out to have
   been load-bearing, not incidental: the 4× gap *was* the bin count. The
   "re-weight ≫ recency" conclusion is retracted; at fair k every method is noise.
2. **Nothing carries grade signal at this resolution.** AMI spans −0.056…+0.030,
   NMI 0.006…0.087 — all chance-band. The fair ablation table is a null result,
   and a null result is the finding: file-level aggregation choice does not move
   grade agreement.
3. **Compactness-≠-relevance, third instance.** Frozen-LSTM silhouette 0.72–0.92
   (vs. baseline 0.22) buys AMI ≤ 0.03. Same moral as the model table and Exp-A,
   now inside one embedding family.
4. **Purity flat** (0.27–0.40) — majority-vote noise on 4 bins, reported, not used.
5. **Determinism checks:** last-5 / no-1-line / decay reproduce the open-k docs
   bit-identically (their winners were already k=4); LSTM-last sil mean 0.876±0.031
   matches exp_033 exactly. (LSTM-mean sil mean differs from exp_034's 0.816
   because seed-2's open-k winner was k=7, sil 0.753 → 0.721 at k=4 — expected.)
6. **Limits:** one algorithm (KMeans), one k, coarse bins — same as the NMI rerun;
   AMI measures agreement, not causation. What this kills is specifically the
   claim that aggregation choice matters for grade tracking, not that it matters
   for representation quality generally.
