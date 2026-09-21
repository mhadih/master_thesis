"""CCT5 embeddings from a local CSV of code snapshots (no database needed).

Same logic as cct5_inference.py (unified-diff <add>/<del>/<keep> strings per
consecutive snapshot pair -> T5 encoder mean-pool per change -> mean per file ->
token-length-weighted mean per user), but reads snapshots from a CSV file with
columns [user_id, filename, content, date] instead of PostgreSQL.

Designed for cloud runners (Kaggle / Colab) where the DB is unavailable:
  - put cpp_traces.csv anywhere and point CPP_TRACES_CSV at it,
  - needs the CCT5 checkpoint in CCT5_DIR (or set CCT5_DIR),
  - writes student_embeddings/user_CCT5_embeddings.jsonl under the repo root.

Env vars:
  CPP_TRACES_CSV  path to snapshots CSV (default: <repo>/my_ccbert/cpp_traces.csv)
  CCT5_DIR        dir with config.json + model.safetensors (default: <repo>/cct5)
  CCT5_BATCH      inference batch size (default: 16; raise to 64 on >=8GB GPUs)
  CCT5_LIMIT_USERS  embed only first N users (smoke tests)
"""

import difflib
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
CCT5_DIR = os.environ.get("CCT5_DIR", os.path.join(REPO, "cct5"))
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
BATCH = int(os.environ.get("CCT5_BATCH", "16"))
LIMIT = int(os.environ.get("CCT5_LIMIT_USERS", "0"))
OUT = os.path.join(REPO, "student_embeddings", "user_CCT5_embeddings.jsonl")


def build_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    base = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<keep>", "<add>", "<del>",
            "<start>", "<end>", "<issue_id>", "<version_id>", "<commit_id>"]
    extra = ["<extra_id_{}>".format(i) for i in range(99, -1, -1)]
    extra += ["<e{}>".format(i) for i in range(99, -1, -1)]
    extra += ["<msg>"]
    add_tokens = base + extra
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [t for t in add_tokens if t not in tokenizer.get_vocab()]})
    return tokenizer


def encode_change(old_code, new_code):
    """One consecutive snapshot pair -> CCT5 diff string (faithful to upstream)."""
    diff = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines()))[2:]
    parts = []
    for line in diff[1:]:
        if not line:
            continue
        if line[0] == '+':
            parts.append("<add>" + line[1:])
        elif line[0] == '-':
            parts.append("<del>" + line[1:])
        else:
            parts.append("<keep>" + line[1:])
    return "".join(parts)


tokenizer = build_tokenizer()
# NOTE: use_safetensors=True avoids torch.load, which transformers>=4.57 blocks
# for torch<2.6 (CVE-2025-32434). Place a safetensors conversion of the
# CCT5 pre-training checkpoint next to pytorch_model.bin in CCT5_DIR first.
model = T5ForConditionalGeneration.from_pretrained(CCT5_DIR, use_safetensors=True)
model.eval()  # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CCT5 model is completed on", device, flush=True)


@torch.no_grad()
def encode_batch(diff_list):
    toks = tokenizer(diff_list, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=MAX_LENGTH)
    toks = {k: v.to(device) for k, v in toks.items()}
    h = model.encoder(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])[0]
    return h.mean(dim=1).cpu()


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
n_pairs = 0
n_done = 0
for uid in tqdm(uids, desc="Users"):
    df = pd.concat(user_dfs[uid])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    file_reprs, file_lens = [], []
    for fn, fdf in df.groupby("filename", sort=False):
        fdf = fdf.reset_index(drop=True)
        contents = [c if isinstance(c, str) else "" for c in fdf["content"].tolist()]
        diffs = []
        for i in range(1, len(contents)):
            if not contents[i - 1].strip() or not contents[i].strip():
                continue
            d = encode_change(contents[i - 1], contents[i])
            if d.strip():
                diffs.append(d)
        if not diffs:
            continue
        n_pairs += len(diffs)
        vecs = torch.cat([encode_batch(diffs[j:j + BATCH]) for j in range(0, len(diffs), BATCH)], dim=0)
        file_reprs.append(vecs.mean(dim=0))
        file_lens.append(len(contents[-1].split()) if contents[-1].strip() else 0)
    if not file_reprs:
        continue
    w = torch.tensor(file_lens, dtype=torch.float32)
    w = w / w.sum()
    emb = (torch.stack(file_reprs) * w.unsqueeze(1)).sum(dim=0)
    n_done += 1
    with open(OUT, "a") as af:  # incremental append: aborts lose nothing
        af.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
        af.flush()

print("pairs: %d, users embedded this run: %d" % (n_pairs, n_done), flush=True)
print("output:", OUT, flush=True)
