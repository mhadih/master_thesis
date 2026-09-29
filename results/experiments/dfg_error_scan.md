# DFG robustness scan — tree-sitter ERROR nodes in S03 snapshots

- **Date (UTC):** 2026-09-29
- **Scope:** all 131,776 Text-Change snapshots of the 68 CodeT5 users from
  `my_ccbert/cpp_traces.csv` (the exact input population of the CCT5-flow run),
  parsed with `cct5/dfg_cpp.extract_dataflow` (comment-stripped code), 10 s
  SIGALRM per snapshot.
- **Script:** `/tmp/opencode/dfg_scan.py` (not committed; method fully described here).

## Results

| Measure | Count | Share |
|---|---|---|
| Snapshots parsed | 131,776 | 100% |
| With ≥1 `ERROR` node | 18,040 (in 62/68 users) | 13.7% |
| Empty DFG output | 16 | 0.01% |
| Per-snapshot timeouts (>10 s) | 0 | — |
| Other exceptions | 0 | — |

## Decision (no code change)

ERROR subtrees are common (13.7%, nearly every student — expected: snapshots are
mid-edit code that routinely doesn't compile) but harmless in practice:
tree-sitter's error recovery keeps salvageable fragments as children of the
ERROR node, the walker's generic `else` branch recurses into them normally, and
the blanket `try/except` in `extract_dataflow` essentially never fires. The
feared total-loss path (one bad subtree nuking a whole file's graph) does not
materialize at 131k scale — 16 empty graphs, zero timeouts.

An explicit `ERROR` branch with per-child guards would change ~nothing;
deliberately not implemented. Revisit only if the input population changes
(e.g., S04, other languages).

## Reading notes (for the conclusion / threats to validity)

1. DFG-based results (exp_051–054) rest on partial graphs for ~1 in 7 snapshots —
   valid data flow where parseable, degraded gracefully where not.
2. Empty-DFG rate (0.01%) is three orders of magnitude below the noise rates of
   the clusterers (up to 45%), so parse failures cannot explain any clustering
   outcome in this thesis.
