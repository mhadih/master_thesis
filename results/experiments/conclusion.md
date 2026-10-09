# Conclusion — transformer embeddings of student code evolution vs. grades (S03)

- **Date (UTC):** 2026-10-07
- **Source:** exp_001–082, exp_088–092, CC2Vec ablations (029–034), fixed-K reruns
  (055–082), k-effect table, dim sweep (083–087), reducer comparison (088–092),
  supervised pairwise LSTM, CCT5 flow view (051–054), cluster aspect tables,
  top-down design features. Shared base protocol unless noted: 62 students,
  UMAP-50 (cosine, L2-normalized), equal-depth grade bins, cosine silhouette.

## 1. Headline answers

1. **Best grade alignment overall:** CodeBERT whole-file + OPTICS (exp_038,
   NMI 0.6641, k=20) and e5 whole-file + OPTICS (exp_008, NMI 0.5485, k=17).
   Both live at fine granularity (17–20 clusters for 62 students).
2. **At fixed resolution the ranking inverts:** at k=4 every model scores
   NMI 0.04–0.16 with CCBERT on top (0.1570); e5/CodeT5 collapse from ~0.54.
   Every ranking claim in this thesis must name its k.
3. **Compactness ≠ grade relevance.** Change models own silhouette (0.63–0.71)
   but trail on NMI (0.02–0.21); raw-space runs score NMI 0.32 with *negative*
   silhouette. Report both metrics together, always.
4. **File-level beats function-level in all four paired models** (0.54>0.16,
   0.66>0.21, 0.55>0.19, 0.12>0.09): document coherence wins for grade
   relevance; chopping helps nothing except CodeBERT-density edge cases.
5. **Change-sequence input buys nothing here:** CCBERT/CC2Vec/CCT5 NMI 0.02–0.21
   vs. snapshot models up to 0.66 — except edit-size-weighted CC2Vec (0.301),
   where the *aggregation*, not the model, does the work.
6. **How you pool matters as much as what you encode:** CC2Vec NMI spans
   0.04–0.30 across aggregations (plain mean → edit-size weights); recency
   (last-5, decay, LSTM) sharpens clusters without adding grade signal; dropping
   one-line edits hurts (small edits carry authorial signal).
7. **Supervised check agrees weakly:** pairwise-ranking LSTM beats chance 0.58
   vs. 0.50 with Spearman ≈ 0.08 — order carries faint ranking signal, no
   absolute-grade signal.
8. **Behavioral correlates:** top-down design score correlates +0.29 with grades
   (strongest single behavioral feature); edit size matters more than edit count
   (users 49 vs. 75: ~6k snapshots each, grades 96.7 vs. 70.7); clusters separate
   on Correctness (0–40 spread) while style aspects sit near ceiling.

## 2. Per-axis summary

- **Models (best NMI):** e5 0.5485 (OPTICS-17) ≈ CodeT5 0.5356 (KMeans-17) >
  CodeBERT 0.6641* (OPTICS-20, k=20 caveat) > CCT5 0.2038 > CodeBERT-func 0.2085
  > CCBERT 0.2822 (k=7, sil 0.03 caveat) > CC2Vec 0.1645 (k=7) > GraphCodeBERT 0.2041 (k=7) >
  CC2Vec-base 0.1148. \*See caveat: micro-clusters.
- **Algorithms:** no universal winner; OPTICS most embedding-sensitive
  (NMI 0.04–0.66); KMeans/BIRCH stable; DBSCAN≡BIRCH partitions coincide twice
  (ARI 1.0, GraphCodeBERT + CCBERT).
- **k:** four curve shapes — monotone climbers (report argmax-k), CCBERT peak at
  6–7, CodeT5/e5 flat-then-jump (signal only at fine k). Fixed-k (4/5/6/7)
  tables + curve figures in results/figures/k_effect_*.png.
- **Reduction:** UMAP-50 validated (only dim reaching series-max NMI); d=5/raw
  destroy fine structure; reducer matters as much as embedding (UMAP 0.66 vs.
  PCA 0.15); AE seed-unstable (report mean±std); LDA excluded (supervised +
  classes−1 cap — documented, not omitted).
- **Aspects:** clusters separate on Correctness, not style; failure pocket
  (cluster 18, raw 66.6) vs. top cluster (raw ~151); Web reads as strategic
  effort allocation; skill profiles per cluster in exp029_cluster_skill_profiles.md.

## 3. Threats to validity

1. n=62 students, one assignment, one semester — no generalizability claim.
2. Grades conflate effort, prior knowledge, and behavior; association ≠ causation;
   no intervention is licensed by these results.
3. NMI/purity couple algorithm, embedding, and bin count — fixed-k tables
   mitigate, not eliminate, this.
4. DFG flow view rests on a from-scratch C++ port (13.7% snapshots contain ERROR
   nodes; recovery succeeds on all but 0.01% — measured, see dfg_error_scan.md).
5. Identifiable grade sheets stay local (gitignored); committed artifacts print
   only aggregates.
6. The k=20 headline cell (exp_038) is near-1:1 matching territory — an
   upper-bound demonstration pending the fixed-bin re-evaluation.

## 4. Future work

1. Fixed-bin (e.g., k=5 for all) re-evaluation of the headline ranking.
2. 10+ seeds for LSTM aggregators; GRU/BiLSTM and trained (not frozen) variants.
3. VAE/fixed-projection control for autoencoder reducers.
4. PDG-based (CCS2Vec-style) models — dropped for lack of public weights/Java-only
   pipeline; needs a Joern C++ pipeline + from-scratch training.
5. Top-down/bottom-up classifier validation against hand labels; S04 replication
   of the full matrix (stats docs already prepared: s03/s04_snapshot_statistics.md).
