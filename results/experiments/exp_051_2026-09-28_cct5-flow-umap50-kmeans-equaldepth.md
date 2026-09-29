# Experiment 051 — CCT5 flow-conditioned + UMAP-50 + KMeans + equal-depth grades (S03)

- **Date (UTC):** 2026-09-28
- **ID:** `exp_051`
- **Status:** partial — embeddings generated on Kaggle (see §1); clustering/evaluation executed locally; DB-backed repo scripts still blocked
- **Varies vs. exp_025:** embedding view only (diff-string → DFG-conditioned); model identical (CCT5 Gen checkpoint)

## 1. Setup

| Component | Choice | Source / config |
|---|---|---|
| Dataset | S03 `code_recorder` snapshots, consecutive per-(user, file) change pairs, cpp files | `my_ccbert/cpp_traces.csv` via Drive → Kaggle (DB down) |
| Embedding model | **CCT5** (Gen checkpoint, 768-dim): per-hunk CDG-format input = `<old DFG edges, hunk-scoped> <extra_id_0> <new DFG edges> <extra_id_0> <del>-marked deleted lines`, T5 encoder mean-pool per hunk → mean per file → token-length-weighted mean per user — mirrors `generate_emb/cct5_flow_inference_csv.py` + `cct5/dfg_cpp.py` (C++ tree-sitter port of the official C# DFG walker) | Kaggle GPU run → `student_embeddings/user_CCT5_flow_embeddings.jsonl` (65 users × 768-dim; same 68-user target set, minus `7, 142, 148`) |
| Students evaluated | **62** (3 ungraded excluded: `143, 145, 146` — identical graded set to exp_001) | `gradesheets/user_grades.csv` |
| Dimension reduction | **UMAP-50**, `metric=cosine`, `random_state=42`, L2-normalized | `clustering/kmeans_clustering.py` (defaults) |
| Clustering | **KMeans**, k sweep 4…20, `random_state=42`, cosine silhouette | same script |
| Grade discretization | **Equal-depth** with k = stored best_k (=4) | `src/evaluation/evaluation.py` (unchanged) |

Run sandbox: `/tmp/opencode/s03run_cct5f/km/`. Artifacts: `../models/exp_051_best_clustering.pkl`,
`../figures/exp_051_umap_kmeans_k4.png`.

## 2. Results

| Metric | exp_051 (flow view) | exp_025 (diff view) |
|---|---|---|
| Best k | **4** | 5 |
| Silhouette | 0.5879 | 0.5304 |
| NMI | 0.1097 | 0.1461 |
| Purity | 0.3871 | 0.3387 |

- Cluster sizes: `[6, 15, 18, 23]`.
- Grade-bin borders (4 bins): `[50.17, 78.83, 98.92, 103.33]`.

## 3. Component observations (for the conclusion)

- **Flow conditioning neither helps nor hurts much** (NMI 0.110 vs. 0.146, both
  weak): hunk-scoped data-flow adds structure (sil 0.59 vs. 0.53, coarser k=4)
  but no grade signal beyond the diff view. The two CCT5 views agree with each
  other more than either agrees with grades.
- Purity uptick (0.39 vs. 0.34) is the 4-vs-5-bin effect, not a real gain.
- Methods note: DFGs come from a from-scratch C++ port (`cct5/dfg_cpp.py`) of the
  official C# walker — port fidelity (C#≈C++ statement syntax) is good for the
  simple student code here, but edge cases (templates, macros, `>>` streams)
  fall back to generic recursion, same blind spots as upstream.

## 4. Blocked steps

Repo DB scripts (Steps 0/1/3) still blocked; embeddings generated off-DB (Kaggle).
Steps 2/4 skipped per workflow note.

## 5. Reproduce

```bash
# 1. embeddings on Kaggle/Colab GPU: CCT5_BATCH=64 python generate_emb/cct5_flow_inference_csv.py
#    (needs tree_sitter==0.26.0 + tree_sitter_cpp==0.23.4)
# 2. clustering locally
mkdir -p /tmp/opencode/s03run_cct5f/km && cd /tmp/opencode/s03run_cct5f/km
ln -sfn /home/hadi/thesis/student_embeddings student_embeddings
ln -sfn /home/hadi/thesis/gradesheets gradesheets
# kmeans_cf.py / evaluate_cf.py = repo scripts with embeddings → .../user_CCT5_flow_embeddings.jsonl
MPLBACKEND=Agg /home/hadi/thesis/venv/bin/python kmeans_cf.py
/home/hadi/thesis/venv/bin/python evaluate_cf.py
```
