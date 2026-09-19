# Experiment 021 — CC2Vec + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-19
- **ID:** `exp_021`
- **Status:** full pipeline — embeddings freshly generated (see §1); DB-backed repo scripts still blocked
- **Varies vs. exp_001/005/009/013/017:** embedding model only (→ CC2Vec, HAN over change hunks); all else identical

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, consecutive per-(user, file) change pairs, cpp files | `my_ccbert/cpp_traces.csv` (DB down, so offline CSV used) |
| Embedding model | **CC2Vec** (Hierarchical Attention Network, embed 64 / hidden 32, OpenStack-pretrained `openstack_cc2ftr.pt`): whitespace-tokenized added/removed lines per pair → upstream padding (2 files × 10 lines × 64 tokens) → `forward_commit_embeds_diff` 196-dim per change → mean per file → token-length-weighted mean per user — mirrors `generate_emb/cc2vec_inference.py` | 68 CodeT5 users → `student_embeddings/user_CC2Vec_embeddings.jsonl` (65 users × 196-dim; ~113.7k pairs, ~25 min on MX150, batched 32) |
| Students evaluated | **62** (3 targets had no usable pairs: `7, 142, 148`; 3 present but ungraded: `143, 145, 146` — identical graded set to exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=5) | `src/evaluation/evaluation.py` (unchanged) |

Notes: (a) weights + dict vendored under `cc2vec/` from Zenodo record 3965149
(git-ignored, 32 MB). (b) Upstream `forward_code` hardcodes CUDA tensors — GPU required.

Run sandbox: `/tmp/opencode/s03run_cc2v/km/`. Artifacts: `../models/exp_021_best_clustering.pkl`,
`../figures/exp_021_umap_kmeans_k5.png`.

## 2. Results

| Metric | Value |
|---|---|
| **Best k** | **5** |
| **Silhouette (cosine)** | **0.2987** |
| **NMI** (vs. 5 equal-depth grade bins) | **0.0764** |
| **Purity** | **0.3226** |

- Sweep: k=4: 0.2233, k=5: **0.2987** ✅, 6: 0.2898, 7: 0.2843, 8–20: 0.20–0.27 —
  flat and weak throughout, no competitive peak.
- Cluster sizes: `[9, 9, 12, 15, 17]`.
- Grade-bin borders (5 bins): `[40.38, 60.05, 86.17, 100.0, 103.33]`.

## 3. Component observations (for the conclusion)

- **CC2Vec is the outlier on compactness:** silhouette 0.30 vs. 0.46–0.71 for every
  other embedding — the HAN change vectors form no tight groups in UMAP-50 space.
  Possible causes for discussion: (i) truncation to 10 changed lines × 64 tokens
  discards most of each student edit; (ii) OpenStack-vocabulary OOV on student code;
  (iii) JIT-defect-pretraining objective mismatched to authorship/style grouping.
- NMI 0.0764 ≈ floor (cf. CCBERT 0.06–0.16): neither change-sequence model captures
  grades — but CC2Vec additionally lacks even cluster structure.

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked; embeddings generated from the offline
CSV instead. Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings (needs cc2vec weights+dict; CUDA required by upstream forward)
/home/hadi/thesis/venv/bin/python /tmp/opencode/s03run_cc2v/cc2v_from_csv.py
# 2. clustering
mkdir -p /tmp/opencode/s03run_cc2v/km && cd /tmp/opencode/s03run_cc2v/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_cc2v.py / evaluate_cc2v.py = repo scripts with embeddings → .../user_CC2Vec_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_cc2v.py
/home/hadi/thesis/venv/bin/python evaluate_cc2v.py
```
