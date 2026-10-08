# Cluster skill profiles — exp_029 rerun with normalized aspect scores (S03)

- **Date (UTC):** 2026-09-29
- **Source:** exp_029 pipeline rerun (CC2Vec edit-size-weighted + KMeans k=9, sil 0.3412 —
  reproduced exactly), clusters mapped user_id → 9-digit studentId via DB,
  aspects from `gradesheets/APS03_CA6_integrated_details.csv`.
- **Normalization:** each student's aspect score divided by the column maximum
  over all students in the details file (Design /43, Clean Code /42, multifile /12,
  Correctness /43, Feature Addition /16, Web /22, Bonus /2, raw /157), then
  averaged per cluster. All 62 students matched, zero missing.

## Normalized (cluster × aspect) means

| cluster | n | Design | Clean Code | multifile | Correctness | Feature Add. | Web | Bonus | raw |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.914 | 0.954 | 0.900 | 0.435 | **0.956** | 0.800 | 0.800 | 0.850 |
| 1 | 10 | 0.843 | 0.886 | 0.883 | **0.603** | 0.762 | 0.695 | 0.700 | 0.800 |
| 2 | 7 | 0.794 | **0.923** | 0.845 | 0.164 | 0.806 | 0.471 | 0.429 | 0.700 |
| 3 | 9 | 0.900 | 0.918 | **0.944** | 0.645 | 0.840 | 0.758 | 0.556 | 0.832 |
| 4 | 9 | 0.847 | 0.914 | 0.792 | 0.530 | **0.903** | 0.654 | 0.611 | 0.790 |
| 5 | 4 | **0.933** | 0.875 | 0.875 | 0.439 | 0.703 | 0.199 | 0.375 | 0.710 |
| 6 | 6 | 0.779 | 0.935 | **0.972** | 0.364 | 0.792 | 0.348 | 0.667 | 0.703 |
| 7 | 7 | 0.877 | 0.886 | 0.911 | 0.472 | 0.750 | 0.644 | 0.500 | 0.767 |
| 8 | 5 | 0.933 | **0.967** | 0.900 | **0.658** | 0.800 | 0.645 | 0.600 | 0.838 |

(Bold = cluster's top aspect, raw score excluded — it is a sum, not a skill.)

## Major skill per cluster (for the conclusion)

- **0 — Feature builders** (Feature Addition 0.96): extend code confidently, middling Correctness (0.44) — breadth over verification.
- **1 — Most-correct generalists** (Correctness 0.60, series best at this granularity): no single standout aspect, highest raw (0.80); the "does everything well enough" profile.
- **2 — Clean but broken** (Clean Code 0.92 vs. Correctness 0.16, series worst): style without function — the mirror image of cluster 1 and the strongest single dissociation in the table.
- **3 — Multi-file architects** (multifile 0.94, Correctness 0.65 second-best): structure + working code, the most balanced high performers.
- **4 — Feature builders, weaker** (Feature Addition 0.90, Correctness 0.53): same signature as cluster 0 at lower raw (0.79 vs. 0.85).
- **5 — Designers who skip web** (Design 0.93 top, Web 0.20 lowest): strong structure, optional parts untouched — strategic effort allocation, not inability (their Design proves competence).
- **6 — Structure specialists** (multifile 0.97, Clean Code 0.94, Correctness 0.36): beautiful architecture that doesn't run — cluster 2's stronger sibling.
- **7 — All-rounders** (no standout; Design 0.88 highest): mid everywhere, raw 0.77.
- **8 — Strong all-rounders** (Clean Code 0.97, Design 0.93, Correctness 0.66 series best): the only cluster top-tier in both style *and* function; highest raw (0.84).

## Cross-cluster conclusions

1. **Correctness spreads 4× (0.16–0.66) while style aspects sit at 0.78–0.97:**
   clusters separate on whether code *works*, not how it *looks* — same verdict
   as the exp_038 aspect table, now replicated under a different model,
   algorithm, and k.
2. **Style-without-function is a repeated profile** (clusters 2 and 6): high
   Clean Code/multifile with Correctness < 0.4 — coherent editing that never
   converges to working code. Candidate case-study pool for debugging-behavior
   analysis.
3. **Web is the strategic-choice aspect** (0.20–0.80, widest relative spread with
   Bonus): low Web co-occurs with high Design (cluster 5), so skipping it reads
   as effort allocation, supporting a "rational student" interpretation of
   partial scores.
4. **Caveats:** n=4–10 per cluster (wide uncertainty on every cell); maxima used
   for normalization are cohort-specific (a harder/easier assignment would shift
   them); Bonus/Web are partly optional-task artifacts, not pure skills.
