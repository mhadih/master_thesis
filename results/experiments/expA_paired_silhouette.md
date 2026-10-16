# Exp-A: paired inference on per-student cluster quality (CodeT5 vs. GraphCodeBERT)

- **Date (UTC):** 2026-10-11 (v2 — see §5 for what v1 got wrong)
- **ID:** `exp-A` (statistical-inference companion to exp_055/039, not a clustering run)
- **Status:** executed — script `/tmp/opencode/expA_v2.py` (sandbox; method fully described below)

## 1. Design

| Element | Choice |
|---|---|
| Question | Does the embedding model change cluster quality? |
| Data | Per-student silhouette scores (cosine, UMAP-50 space), paired by student (n=62) |
| Arm A | exp_055 setup: CodeT5 **whole-file** + KMeans k=4 (mean sil 0.3889) |
| Arm B | exp_039 setup: GraphCodeBERT **whole-file** + KMeans k=4 (mean sil 0.5966) |
| Fixed | students, input protocol (whole-file both arms), algorithm, k=4, UMAP pipeline, seeds — only the embedding model varies |
| Verification | Both spaces reproduced bit-identically (labels match committed exp_055/exp_039 pkls; means match exp docs to 4 decimals); file orderings differed, so pairing is by student ID, not position |

## 2. Results

- Mean paired difference (CodeT5 − GraphCodeBERT): **−0.2077** — GraphCodeBERT separates 50/62 students better.
- Normality of differences (Shapiro–Wilk): W=0.9757, **p=0.254** → normality holds; t-test fully licensed, Wilcoxon run regardless.
- Paired t-test: t=−6.24, **p≈4.6×10⁻⁸**; Wilcoxon signed-rank: **p≈4.7×10⁻⁷**. Agreeing.
- Effect size: Cohen's d_z = **−0.793** (large); 95% Student-t CI (t_{0.975,61}=2.000, **not** the 1.96 normal approximation) = **[−0.2742, −0.1411]** (excludes 0).

## 3. Descriptive analysis (for the thesis)

The difference is real, large, and consistent across students — 50 of 62 are
individually better separated under GraphCodeBERT, and the interval sits far
from zero. Distributional assumptions are comfortable here (Shapiro p=0.25, both
tests agree to overwhelming significance), so the conclusion does not hinge on
them.

The substantive reading cuts against naive interpretation, and that is the
point to stress: the *significantly better-clustered* embedding is GraphCodeBERT
— whose grade agreement trails CodeT5's (NMI 0.104 vs. 0.124 at k=4; 0.09–0.20
vs. 0.09–0.54 across k). This is the compactness-≠-relevance thesis with
inferential backing: we can now state not only that the better-separated space
tracks grades worse, but that its superiority in separation is itself
statistically significant (p<10⁻⁷, d_z≈0.8). A reviewer cannot dismiss the
divergence as noise.

Limits: (i) one algorithm (KMeans) and one k (4) — the paired design buys
internal validity at the cost of generality; (ii) silhouette is the tested
quantity, not NMI (single numbers per setup cannot be tested — see methods);
(iii) n=62 from one assignment/semester, as throughout.

## 4. Reproduce

```bash
/home/hadi/thesis/venv/bin/python /tmp/opencode/expA_v2.py
# needs results/models/exp_055_best_clustering.pkl + exp_039_best_clustering.pkl
# (verification step), student_embeddings/*, gradesheets/user_grades.csv; CPU, ~2 min
```

## 5. Correction log (v1 → v2)

- **v1 compared setups, not models:** CodeT5 arm used whole-file embeddings while
  the GraphCodeBERT arm used function-level ones — the 0.265 gap confounded
  model with input protocol. v2 reruns both arms whole-file (exp_055 vs.
  exp_039); the gap narrows to 0.208 but the verdict is unchanged.
- **v1 CI used 1.96 (normal); v2 uses t_{0.975,61} = 2.000.** Numerically
  negligible here (CI [−0.274, −0.141] either way) but methodologically required;
  the doc now states the critical value explicitly.
