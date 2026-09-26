"""CCT5 flow-conditioned embeddings from a local CSV (no database needed).

Second view alongside generate_emb/cct5_inference_csv.py (plain diff strings).
Faithful to the official CCT5 repo's CDG pre-training input (gen_CDG_example):
per hunk, input =
    <old DFG edges, hunk-scoped> <extra_id_0> <new DFG edges, hunk-scoped>
    <extra_id_0> <old code = <del>-prefixed deleted lines of the hunk>
Edges serialize as 'src dst' (comesFrom) / 'dst src' (computedFrom), joined by
<extra_id_0>. Pairs are skipped exactly like upstream: empty old tokens,
unparseable hunk header, or equal old/new scoped DFGs.

C++ data-flow comes from cct5/dfg_cpp.py (port of the official C# walker to
tree-sitter-cpp node types). Representation = T5 encoder mean-pool (768-dim),
same as the diff view (the paper uses the first token for defect prediction;
mean-pool keeps this thesis' exp_025-028 protocol comparable).

Writes student_embeddings/user_CCT5_flow_embeddings.jsonl under the repo root.
Env vars: CPP_TRACES_CSV, CCT5_DIR, CCT5_BATCH, CCT5_LIMIT_USERS (as in
cct5_inference_csv.py). Incremental append + resume support.
"""

import difflib
import json
import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cct5"))
from dfg_cpp import extract_dataflow, filter_dfg, serialize_edges, get_parser

CSV_PATH = os.environ.get("CPP_TRACES_CSV", os.path.join(REPO, "my_ccbert", "cpp_traces.csv"))
CCT5_DIR = os.environ.get("CCT5_DIR", os.path.join(REPO, "cct5"))
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
OUT = os.path.join(REPO, "student_embeddings", "user_CCT5_flow_embeddings.jsonl")
SEP = "<extra_id_0>"

TAG_MATCHER = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")


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


def split_hunks(diff_lines):
    """difflib.unified_diff output (no headers) -> [(header, body_lines)]."""
    hunks, cur = [], None
    for line in diff_lines:
        if line.startswith("@@"):
            if cur is not None:
                hunks.append(cur)
            cur = (line, [])
        elif cur is not None and not line.startswith("\\"):
            cur[1].append(line)
    if cur is not None:
        hunks.append(cur)
    return hunks


def flow_inputs_for_pair(old_code, new_code, parser):
    """One snapshot pair -> list of CDG-format input strings (one per hunk)."""
    diff = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines()))
    if not diff:
        return []
    diff = diff[2:]  # drop ---/+++ headers; split_hunks handles @@ headers
    old_tokens, old_dfg, old_i2c = extract_dataflow(old_code, parser)
    new_tokens, new_dfg, new_i2c = extract_dataflow(new_code, parser)
    if not old_tokens:
        return []
    inputs = []
    for header, body in split_hunks(diff):
        m = TAG_MATCHER.match(header)
        if not m:
            continue
        ss, sl, ts, tl = m.groups()
        ss, ts = int(ss), int(ts)
        sl = int(sl) if sl else 1
        tl = int(tl) if tl else 1
        old_scoped = filter_dfg(old_dfg, old_i2c, (ss - 1, ss - 1 + sl))
        new_scoped = filter_dfg(new_dfg, new_i2c, (ts - 1, ts - 1 + tl))
        if sorted(map(str, old_scoped)) == sorted(map(str, new_scoped)):
            continue  # upstream is_equal_dfg rule
        old_code_str = "".join("<del>" + l[1:] for l in body if l.startswith("-"))
        text = serialize_edges(old_scoped) + SEP + serialize_edges(new_scoped) + SEP + old_code_str
        if text.strip(SEP).strip():
            inputs.append(text)
    return inputs


tokenizer = build_tokenizer()
model = T5ForConditionalGeneration.from_pretrained(CCT5_DIR, use_safetensors=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CCT5 model is completed on", device, flush=True)
parser = get_parser()


@torch.no_grad()
def encode_batch(text_list):
    toks = tokenizer(text_list, return_tensors="pt", truncation=True,
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
n_hunks = 0
n_done = 0
for uid in tqdm(uids, desc="Users"):
    df = pd.concat(user_dfs[uid])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    file_reprs, file_lens = [], []
    for fn, fdf in df.groupby("filename", sort=False):
        fdf = fdf.reset_index(drop=True)
        contents = [c if isinstance(c, str) else "" for c in fdf["content"].tolist()]
        hunks = []
        for i in range(1, len(contents)):
            if not contents[i - 1].strip() or not contents[i].strip():
                continue
            hunks.extend(flow_inputs_for_pair(contents[i - 1], contents[i], parser))
        if not hunks:
            continue
        n_hunks += len(hunks)
        vecs = torch.cat([encode_batch(hunks[j:j + BATCH]) for j in range(0, len(hunks), BATCH)], dim=0)
        file_reprs.append(vecs.mean(dim=0))
        file_lens.append(len(contents[-1].split()) if contents[-1].strip() else 0)
    if not file_reprs:
        continue
    w = torch.tensor(file_lens, dtype=torch.float32)
    w = w / w.sum()
    emb = (torch.stack(file_reprs) * w.unsqueeze(1)).sum(dim=0)
    n_done += 1
    with open(OUT, "a") as af:
        af.write(json.dumps({"user_id": str(uid), "embedding": [float(x) for x in emb.tolist()]}) + "\n")
        af.flush()

print("hunks: %d, users embedded this run: %d" % (n_hunks, n_done), flush=True)
print("output:", OUT, flush=True)
