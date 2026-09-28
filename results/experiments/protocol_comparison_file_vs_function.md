# Protocol comparison — file-level vs. function-level input (S03)

- **Date (UTC):** 2026-09-28
- **Source:** best-NMI experiment per (method, model) pair: CodeT5 file exp_001 /
  func exp_050; CodeBERT file exp_038 / func exp_010; E5 file exp_008 / func
  exp_044; GraphCodeBERT file exp_040 / func exp_013.
- **Shared protocol:** 62 students (E5/CodeT5-func sets minus user 143), UMAP-50
  (cosine, L2-normalized), equal-depth grade bins, cosine silhouette.

| Model | Method | Best exp (algorithm) | k (bins) | Silhouette | NMI | Purity |
|---|---|---|---|---|---|---|
| CodeT5 | file-level | exp_001 (KMeans, k=17) | 17 | 0.5179 | **0.5356** | 0.3548 |
| CodeT5 | function-level | exp_050 (OPTICS ms=5, xi=0.01; 6+3n) | 6 | 0.6360 | **0.1645** | 0.3065 |
| CodeBERT | file-level | exp_038 (OPTICS ms=2, xi=0.01; 20+6n) | 20 | 0.3903 | **0.6641** | 0.4194 |
| CodeBERT | function-level | exp_010 (DBSCAN eps≈0.019, ms=5; 6+3n) | 7 | 0.6479 | **0.2085** | 0.3387 |
| E5 | file-level | exp_008 (OPTICS ms=2, xi=0.01; 17+4n) | 17 | 0.4603 | **0.5485** | 0.3710 |
| E5 | function-level | exp_044 (DBSCAN eps≈0.021, ms=6; 5+2n) | 6 | 0.5737 | **0.1919** | 0.3548 |
| GraphCodeBERT | file-level | exp_040 (DBSCAN eps≈0.0236, ms=9; 3+18n) | 4 | 0.5825 | **0.1161** | 0.4516 |
| GraphCodeBERT | function-level | exp_013 (KMeans, k=4) | 4 | 0.6540 | **0.0932** | 0.4032 |

## Reading notes (for the conclusion)

1. **File-level wins all four pairs** (0.54>0.16, 0.66>0.21, 0.55>0.19, 0.12>0.09):
   document coherence beats function chopping for grade relevance in every model.
2. **CodeBERT-file's 0.6641 rides on k=20 micro-clusters** (~3 students each vs.
   20 bins) — series best but least comparable cell; pending the fixed-bin
   re-evaluation (open since exp_002).
3. **Asymmetry of the protocol effect:** document models collapse under chopping
   (CodeT5 0.54→0.16, E5 0.55→0.19) while the token-local CodeBERT gains its
   best score at file level anyway (0.66) — granularity interacts with
   pretraining objective, not just input length.
4. GraphCodeBERT is flat either way (0.12/0.09) — protocol-insensitive null.
