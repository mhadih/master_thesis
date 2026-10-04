"""Top-down vs. bottom-up design features per student (S03).

For each of the 68 CodeT5 users, walks code snapshots in date order (from
my_ccbert/cpp_traces.csv) and derives, per student:

  stub_ratio_early  mean stub share over first-quartile snapshots
                    (stub = function with <=1 ';' in body)
  call_before_def   share of calls to ever-user-defined functions whose target
                    was not yet defined at call time
  mean_impl_lag     mean snapshots from a function's first sighting to its
                    first non-stub body (0 if never a stub)
  breadth_25        distinct functions present at 25% of timeline /
                    distinct functions ever (skeleton-early ~= top-down)
  main_first        1 if main() exists at/before the snapshot where >=50% of
                    helpers are defined, else 0

plus topdown_score = mean(stub_ratio_early, call_before_def,
min(mean_impl_lag/10,1), breadth_25, main_first) in [0,1] (provisional composite;
thresholds documented, tune against hand labels).

Functions/calls come from tree-sitter-cpp (same parser as cct5/dfg_cpp.py),
not regexes, so incomplete snapshots still yield signatures.

Writes data/student_design_features.csv (68 rows).
"""

import json
import os
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "cct5"))
from dfg_cpp import get_parser

CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
TARGET_USERS = {
    json.loads(l)["user_id"]
    for l in open(os.path.join(REPO, "student_embeddings/user_CodeT5_file_embeddings.jsonl"))
}
OUT = os.path.join(REPO, "data", "student_design_features.csv")


def iter_identifiers(node):
    if node.type == "identifier" and not node.children:
        yield node
    for c in node.children:
        yield from iter_identifiers(c)


def func_name_and_body(func_node, code):
    """(name, body_text or None, is_stub)."""
    decl = func_node.child_by_field_name("declarator")
    name = None
    if decl is not None:
        for c in decl.children:
            if c.type == "identifier" and not c.children:
                name = code[c.start_byte:c.end_byte]
                break
        if name is None:  # pointer/reference declarators: first identifier descendant
            for c in iter_identifiers(decl):
                name = code[c.start_byte:c.end_byte]
                break
    body = None
    for c in func_node.children:
        if c.type == "compound_statement":
            body = code[c.start_byte:c.end_byte]
    if name is None or body is None:
        return None, None, True
    inner = body[1:-1].strip()
    is_stub = inner.count(";") <= 1
    return name, body, is_stub


def snapshot_features(code, parser):
    """One snapshot -> (funcs {name: is_stub}, called_names [in order])."""
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception:
        return {}, []
    funcs, called = {}, []

    def walk(node):
        if node.type == "function_definition":
            name, body, is_stub = func_name_and_body(node, code)
            if name is not None:
                funcs[name] = is_stub
            for c in node.children:  # still walk body for call sites
                walk_calls(c)
            return
        for c in node.children:
            walk(c)

    def walk_calls(node):
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier" and not fn.children:
                    called.append(code[fn.start_byte:fn.end_byte])
                else:  # method calls etc.: record inner identifiers
                    for c in iter_identifiers(fn):
                        called.append(code[c.start_byte:c.end_byte])
                        break
        for c in node.children:
            walk_calls(c)

    walk(tree.root_node)
    return funcs, called


def main():
    parser = get_parser()
    user_snaps = defaultdict(list)  # uid -> [(date, funcs, called)]
    for ch in pd.read_csv(CSV_PATH, usecols=["user_id", "filename", "content", "date"],
                          chunksize=200000, dtype={"user_id": str}, low_memory=False):
        ch = ch[ch["user_id"].isin(TARGET_USERS)]
        if ch.empty:
            continue
        ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
        for (uid, fn), grp in ch.groupby(["user_id", "filename"], sort=False):
            grp = grp.sort_values("date")
            for _, row in grp.iterrows():
                content = row["content"]
                if not isinstance(content, str) or not content.strip():
                    continue
                funcs, called = snapshot_features(content, parser)
                user_snaps[uid].append((row["date"], funcs, called))
    print("users:", len(user_snaps), flush=True)

    rows = []
    for uid in sorted(user_snaps, key=int):
        snaps = sorted(user_snaps[uid], key=lambda s: (s[0] is pd.NaT, s[0]))
        n = len(snaps)
        q = max(1, n // 4)
        early = snaps[:q]
        stub_ratios = []
        for _, funcs, _ in early:
            if funcs:
                stub_ratios.append(sum(1 for s in funcs.values() if s) / len(funcs))
        stub_early = sum(stub_ratios) / len(stub_ratios) if stub_ratios else 0.0

        ever_defined = set()
        for _, funcs, _ in snaps:
            ever_defined.update(funcs)
        n_calls = n_cbd = 0
        defined_so_far = set()
        first_seen, first_full = {}, {}
        for i, (_, funcs, called) in enumerate(snaps):
            for name in funcs:
                first_seen.setdefault(name, i)
                if not funcs[name]:
                    first_full.setdefault(name, i)
            for name in called:
                if name in ever_defined:
                    n_calls += 1
                    if name not in defined_so_far:
                        n_cbd += 1
            defined_so_far.update(funcs)
        call_before_def = (n_cbd / n_calls) if n_calls else 0.0
        lags = [first_full.get(name, i) - i
                for name, i in first_seen.items() if name in first_full]
        mean_lag = sum(lags) / len(lags) if lags else 0.0

        seen_at_q, total_funcs = set(), set(ever_defined)
        for _, funcs, _ in snaps[:q]:
            seen_at_q.update(funcs)
        breadth_25 = (len(seen_at_q) / len(total_funcs)) if total_funcs else 0.0

        helpers = {name for name in ever_defined if name != "main"}
        main_idx = next((i for i, (_, funcs, _) in enumerate(snaps) if "main" in funcs), None)
        if main_idx is None or not helpers:
            main_first = 0
        else:
            seen_helpers, half_idx = set(), None
            for i, (_, funcs, _) in enumerate(snaps):
                seen_helpers.update(n for n in funcs if n != "main")
                if len(seen_helpers) >= len(helpers) / 2:
                    half_idx = i
                    break
            main_first = 1 if (half_idx is not None and main_idx <= half_idx) else 0

        score = (stub_early + call_before_def + min(mean_lag / 10.0, 1.0)
                 + breadth_25 + main_first) / 5.0
        rows.append({
            "user_id": uid, "snapshots": n,
            "stub_ratio_early": round(stub_early, 3),
            "call_before_def": round(call_before_def, 3),
            "mean_impl_lag": round(mean_lag, 2),
            "breadth_25": round(breadth_25, 3),
            "main_first": main_first,
            "topdown_score": round(score, 3),
        })

    import csv
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
