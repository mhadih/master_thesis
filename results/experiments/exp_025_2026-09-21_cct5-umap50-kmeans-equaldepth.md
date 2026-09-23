# Experiment 025 — CCT5 + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-21
- **ID:** `exp_025`
- **Status:** partial — embeddings generated on Kaggle (see §1); clustering/evaluation executed locally; DB-backed repo scripts still blocked
- **Varies vs. exp_001/005/009/013/017/021:** embedding model only (→ CCT5); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, consecutive per-(user, file) change pairs, cpp files | `my_ccbert/cpp_traces.csv` via Drive → Kaggle (DB down) |
| Embedding model | **CCT5** (CodeT5-based, Gen pre-training checkpoint, 768-dim): unified-diff `<add>`/`<del>`/`<keep>` strings per pair → T5 encoder mean-pool per change → mean per file → token-length-weighted mean per user — mirrors `generate_emb/cct5_inference_csv.py` | Kaggle GPU run → `student_embeddings/user_CCT5_embeddings.jsonl` (66 users × 768-dim; same 68-user target set, minus `7, 148` with no usable pairs) |
| Students evaluated | **62** (4 ungraded excluded: `142, 143, 145, 146` — identical graded set to exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=5) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_cct5/km/`. Artifacts: `../models/exp_025_best_clustering.pkl`,
`../figures/exp_025_umap_kmeans_k5.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **5** |
| **Silhouette (cosine)** | **0.5304** |
| **NMI** (vs. 5 equal-depth grade bins) | **0.1461** |
| **Purity** | **0.3387** |

- Sweep: k=4: 0.5079, k=5: **0.5304** ✅, 6: 0.4760, 7–20: 0.38–0.48 — single clear peak.
- Cluster sizes: `[9, 9, 12, 16, 16]`.
- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **Embedding (CCT5 vs. rest):** silhouette 0.5304 sits with CodeT5/e5 (0.52–0.53),
  above CC2Vec (0.15–0.31) and below CodeBERT-family (0.65–0.71) — CCT5 space is
  moderately structured. But NMI 0.1461 joins the low band (CodeBERT/CCBERT/CC2Vec
  0.02–0.21), far from CodeT5/e5 (0.54): change-oriented pretraining did not transfer
  to grade relevance here.
- Same k-confounding/purity caveats as ever; fixed-bin variant still open.

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked; embeddings generated off-DB (Kaggle).
Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings on Kaggle/Colab GPU (see generate_emb/cct5_inference_csv.py header)
CCT5_BATCH=64 python generate_emb/cct5_inference_csv.py
# 2. clustering locally
mkdir -p /tmp/opencode/s03run_cct5/km && cd /tmp/opencode/s03run_cct5/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_cc5.py / evaluate_cc5.py = repo scripts with embeddings → .../user_CCT5_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_cc5.py
/home/hadi/thesis/venv/bin/python evaluate_cc5.py
```
