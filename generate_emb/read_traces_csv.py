"""Read code snapshots from cpp_traces.csv into per-user Polars frames.

Snapshot content holds raw C++ (quotes, commas, newlines), and some rows break
the strict CSV parsing of Polars' scanner (ComputeError: 'found more fields
than defined in Schema'). Do NOT work around that with
truncate_ragged_lines/ignore_errors — both silently corrupt or drop code.
Instead this module reads with pandas chunks (verified tolerant of this file)
and converts to a single filtered Polars frame.

Usage:
    from read_traces_csv import load_traces
    user_dfs = load_traces("/kaggle/input/s03-traces/cpp_traces.csv", TARGET_USERS)
    # user_dfs: {user_id: pl.DataFrame} with columns [user_id, filename, content, date]

Notes:
- Missing `content` cells arrive as Polars null: use .fill_null("") or check
  `is None` after .to_list().
- Group values are Polars frames: group["content"].to_list(),
  group.group_by("filename"), etc. — or .to_pandas() per group to keep
  pandas-style logic.
"""

import pandas as pd
import polars as pl

COLUMNS = ["user_id", "filename", "content", "date"]


def load_traces(csv_path, target_users, chunksize=200000):
    """Read + filter snapshots. Returns {user_id: pl.DataFrame}."""
    target = {str(u) for u in target_users}
    chunks = []
    for ch in pd.read_csv(
        csv_path,
        usecols=COLUMNS,
        chunksize=chunksize,
        dtype={"user_id": str},
        low_memory=False,
    ):
        ch = ch[ch["user_id"].isin(target)]
        if not ch.empty:
            chunks.append(pl.from_pandas(ch.reset_index(drop=True)))
    if not chunks:
        return {}
    df_filtered = pl.concat(chunks, how="vertical")
    # NOTE: group_by yields 1-tuple keys for a single key column — unwrap them.
    return {
        (uid[0] if isinstance(uid, tuple) else uid): group
        for uid, group in df_filtered.group_by("user_id", maintain_order=True)
    }


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "my_ccbert/cpp_traces.csv"
    # demo filter (full S03 set lives in cct5_inference_csv.TARGET_USERS;
    # not imported here to avoid loading the CCT5 model on import)
    user_dfs = load_traces(csv_path, {"49", "75"})
    print("users found:", len(user_dfs), flush=True)
    for uid, g in user_dfs.items():
        print(uid, "rows:", g.height)
