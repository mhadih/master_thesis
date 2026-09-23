"""Order-invariance ablations over per-file CC2Vec change vectors.

Reads a {(user_id, filename): {vecs[C,196], added, removed, flen}} torch store
(produced per my_ccbert-style change extraction; see exp_029 doc for how
/tmp/opencode/s03run_abl/cc2v_file_changes.pt was built) and writes per-user
embeddings under three file-level aggregation schemes (user level stays
length-weighted, as in the base setup):

  editsize : file_emb = mean over changes weighted by (added + removed) lines.
  last-K   : file_emb = mean over final K changes (K via ABL_LAST_K, default 5).
  no1line  : drop changes with added+removed <= 1, then mean (emptied files skipped).

Env vars:
  ABL_STORE  input .pt store (default: /tmp/opencode/s03run_abl/cc2v_file_changes.pt)
  ABL_OUTDIR output dir for user_*.jsonl (default: student_embeddings/ in repo)
  ABL_LAST_K K for last-K (default: 5)

Output files: user_CC2Vec_abl_editsize.jsonl, user_CC2Vec_abl_last<K>.jsonl,
              user_CC2Vec_abl_no1line.jsonl
"""

import json
import os
from collections import defaultdict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.environ.get("ABL_STORE", os.path.join(REPO, "data", "cc2v_file_changes.pt"))
OUTDIR = os.environ.get("ABL_OUTDIR", os.path.join(REPO, "student_embeddings"))
K = int(os.environ.get("ABL_LAST_K", "5"))


def main():
    store = torch.load(STORE, map_location="cpu", weights_only=False)
    print("files:", len(store), flush=True)

    # method -> uid -> list of (file_emb, flen), deterministic store order
    per_user = defaultdict(list)
    empty_files = 0
    for (uid, fn), d in store.items():
        vecs = d["vecs"].float()
        sizes = [a + r for a, r in zip(d["added"], d["removed"])]
        flen = d["flen"]

        w = torch.tensor(sizes, dtype=torch.float32)
        per_user[("editsize", uid)].append(((vecs * (w / w.sum()).unsqueeze(1)).sum(dim=0), flen))
        per_user[("lastk", uid)].append((vecs[-K:].mean(dim=0), flen))
        keep = [i for i, s in enumerate(sizes) if s > 1]
        if keep:
            per_user[("no1line", uid)].append((vecs[keep].mean(dim=0), flen))
        else:
            empty_files += 1

    print("files emptied by ignore-1-line:", empty_files, flush=True)

    out_names = {
        "editsize": os.path.join(OUTDIR, "user_CC2Vec_abl_editsize.jsonl"),
        f"last{K}": os.path.join(OUTDIR, f"user_CC2Vec_abl_last{K}.jsonl"),
        "no1line": os.path.join(OUTDIR, "user_CC2Vec_abl_no1line.jsonl"),
    }
    # NOTE: committed exp_030 artifacts used K=5 -> user_CC2Vec_abl_last5.jsonl
    for method, out in out_names.items():
        key = "lastk" if method.startswith("last") else method
        uids = sorted({u for (m, u) in per_user if m == key}, key=int)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for uid in uids:
                items = per_user[(key, uid)]
                embs = torch.stack([e for e, _ in items])
                w = torch.tensor([fl for _, fl in items], dtype=torch.float32)
                w = w / w.sum()
                emb = (embs * w.unsqueeze(1)).sum(dim=0)
                f.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
        print(method, "users:", len(uids), "->", out, flush=True)


if __name__ == "__main__":
    main()
