"""Whole-file CodeBERT embeddings from a local CSV (no database needed).

Same logic as codeBert_file_inference.py (final snapshot per (user_id, filename)
embedded whole in one forward pass, [CLS] readout, word-count-weighted mean per
user), but reads snapshots from a CSV file with columns
[user_id, filename, content, date] instead of PostgreSQL.

Designed for cloud runners (Kaggle / Colab) where the DB is unavailable.
Weights load via transformers from_pretrained (needs torch>=2.6 or network
access to HF hub; Kaggle/Colab images satisfy this).

Env vars:
  CPP_TRACES_CSV       snapshots CSV (default: <repo>/my_ccbert/cpp_traces.csv)
  CODEBERT_BATCH       inference batch size (default: 16; raise to 64 on >=8GB GPUs)
  CODEBERT_LIMIT_USERS embed only first N users (smoke tests)

Writes student_embeddings/user_CodeBert_file_embeddings.jsonl under the repo
root. Incremental append + resume support.
"""

import json
import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
# S03 target users (= user_ids in student_embeddings/user_CodeT5_file_embeddings.jsonl,
# inlined here so other code can reuse them without reading that file).
TARGET_USERS = {
    "7", "49", "75", "77", "83", "142", "143", "145", "146", "148", "150", "154",
    "159", "160", "162", "163", "164", "165", "166", "167", "169", "175", "179",
    "180", "181", "182", "183", "184", "189", "191", "195", "198", "199", "204",
    "205", "212", "218", "219", "222", "223", "227", "229", "230", "231", "232",
    "233", "234", "238", "240", "242", "243", "245", "247", "248", "249", "250",
    "252", "255", "259", "261", "262", "267", "269", "273", "279", "281", "283",
    "284",
}
MAX_LENGTH = 512
BATCH = int(os.environ.get("CODEBERT_BATCH", "16"))
LIMIT = int(os.environ.get("CODEBERT_LIMIT_USERS", "0"))
OUT = os.path.join(REPO, "student_embeddings", "user_CodeBert_file_embeddings.jsonl")


def get_codebert_file_embedding(code_snippet):
    """Embed one entire file (truncated to 512 tokens) with [CLS] readout."""
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        return cls_embedding.squeeze().cpu()  # return as 1D tensor


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()  # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CodeBERT model is completed on", device, flush=True)


@torch.no_grad()
def encode_batch(snippets):
    toks = tokenizer(snippets, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=MAX_LENGTH)
    toks = {k: v.to(device) for k, v in toks.items()}
    return model(**toks)[0][:, 0, :].cpu()


DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        line = line.strip()
        if line:
            DONE.add(json.loads(line)["user_id"])
    print("resume: %d users already embedded, skipping" % len(DONE), flush=True)

user_dfs = {}
for ch in pd.read_csv(CSV_PATH, usecols=["user_id", "filename", "content", "date"],
                      chunksize=200000, dtype={"user_id": str}, low_memory=False):
    ch = ch[ch["user_id"].isin(TARGET_USERS)]
    if ch.empty:
        continue
    for uid, grp in ch.groupby("user_id", sort=False):
        user_dfs.setdefault(uid, []).append(grp)
print("users found:", len(user_dfs), flush=True)

uids = [u for u in sorted(user_dfs, key=int) if u not in DONE]
if LIMIT:
    uids = uids[:LIMIT]
n_files = 0
for uid in tqdm(uids, desc="Users"):
    df = pd.concat(user_dfs[uid])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    file_snips, file_lens = [], []
    for fn, fdf in df.groupby("filename", sort=False):
        if not (fn.endswith("cpp") or fn.endswith("hpp")):
            continue
        final_code = fdf.sort_values("date").iloc[-1]["content"]
        if not isinstance(final_code, str) or not final_code.strip():
            continue
        file_snips.append(final_code)
        file_lens.append(final_code.count(" ") + 1)
    if not file_snips:
        continue
    n_files += len(file_snips)
    vecs = torch.cat([encode_batch(file_snips[j:j + BATCH]) for j in range(0, len(file_snips), BATCH)], dim=0)
    w = torch.tensor(file_lens, dtype=torch.float32)
    emb = (vecs * (w / w.sum()).unsqueeze(1)).sum(dim=0)
    with open(OUT, "a") as af:  # incremental append: aborts lose nothing
        af.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
        af.flush()

print("files: %d, users embedded this run: %d" % (n_files, len(uids)), flush=True)
print("output:", OUT, flush=True)
