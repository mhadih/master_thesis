import psycopg2
import pandas as pd
import json
import torch
from uer.layers.layer_norm import LayerNorm
from uer.model_builder import build_model
from uer.model_loader import load_model
from uer.utils.tokenizers import BertTokenizer
from uer.opts import tokenizer_opts
import argparse
import difflib
from transformers import RobertaTokenizer
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
tokenizer_opts(parser)
args = parser.parse_args()

# ---- Step 3: Create tokenizer and attach to aargsrgs ----
args.vocab_path = "/home/hadi/thesis/my_ccbert/codebert_vocab.txt"
args.tokenizer = BertTokenizer(args)
args.embedding = "word_pos_seg"
args.remove_embedding_layernorm = False
args.dropout = 0.1
args.emb_size = 512
args.max_seq_length = 512
args.encoder = "transformer"
args.mask = "fully_visible"
args.layers_num = 4
args.heads_num = 8
args.parameter_sharing = False
args.factorized_embedding_parameterization = False
args.layernorm_positioning = "post"
args.relative_position_embedding = False
args.has_residual_attention = False
args.remove_transformer_bias = False
args.hidden_size = 512
args.feedforward_size = 2048
args.remove_attention_scale = False
args.feed_forward = "ffn"
args.hidden_act = "gelu"
args.relative_attention_buckets_num = 32
args.layernorm = LayerNorm(args.hidden_size)
args.decoder = None
args.data_processor = "bert"
args.target = ["mlm"]
args.tie_weights = True

# ---- Step 4: Build model ----
model = build_model(args)

# ---- Step 5: Load weights ----
model = load_model(model, "ccbert_small.bin-avg")
model.eval()

print("✅ Model loaded successfully!")

# Read from database
conn = psycopg2.connect(
    dbname="code_recorder",
    user="hadi",
    password="***REMOVED***",
    host="localhost",
    port=5432
)
query = """
    SELECT user_id, filename, content, date
    FROM codetrace
    WHERE identifier = 'Text Change'
"""

traces_df = pd.read_sql_query(query, conn)
sorted_df = traces_df.sort_values(by='date')
# Assuming sorted_df is already a DataFrame
filtered_df = sorted_df[sorted_df["filename"].str.endswith("cpp")]
filtered_df.to_csv("cpp_traces.csv", index=False)

# Assume df is already in correct chronological order
# Columns: user_id, filename, content
print("### Structured data (dataframe) is ready!")

# Load CodeBERT tokenizer (same vocab as CCBERT)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Edit action vocabulary (fixed)
EDIT_ACTIONS = {"equal": 0, "replace": 1, "insert": 2, "delete": 3, "<NULL>": 4}

def preprocess_code_change(old_code: str, new_code: str, max_len: int = 256):
    """
    Preprocess two versions of code (C++) into old_ids, new_ids, and edit_ids for CCBERT.
    
    Args:
        old_code (str): Old version of code (before changes).
        new_code (str): New version of code (after changes).
        max_len (int): Max sequence length (truncate if too long).
        
    Returns:
        dict with tensors (old_ids, new_ids, edit_ids)
    """
    # Step 1: Tokenize old and new code
    old_tokens = tokenizer.tokenize(old_code)
    new_tokens = tokenizer.tokenize(new_code)

    # Step 2: Align with difflib
    sm = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    old_aligned, new_aligned, edit_actions = [], [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            old_aligned.extend(old_tokens[i1:i2])
            new_aligned.extend(new_tokens[j1:j2])
            edit_actions.extend(["equal"] * (i2 - i1))
        elif tag == "replace":
            max_len_replace = max(i2 - i1, j2 - j1)
            old_chunk = old_tokens[i1:i2] + ["<NULL>"] * (max_len_replace - (i2 - i1))
            new_chunk = new_tokens[j1:j2] + ["<NULL>"] * (max_len_replace - (j2 - j1))
            old_aligned.extend(old_chunk)
            new_aligned.extend(new_chunk)
            edit_actions.extend(["replace"] * max_len_replace)
        elif tag == "insert":
            new_chunk = new_tokens[j1:j2]
            old_chunk = ["<NULL>"] * len(new_chunk)
            old_aligned.extend(old_chunk)
            new_aligned.extend(new_chunk)
            edit_actions.extend(["insert"] * len(new_chunk))
        elif tag == "delete":
            old_chunk = old_tokens[i1:i2]
            new_chunk = ["<NULL>"] * len(old_chunk)
            old_aligned.extend(old_chunk)
            new_aligned.extend(new_chunk)
            edit_actions.extend(["delete"] * len(old_chunk))

    # Step 3: Convert to IDs
    old_ids = tokenizer.convert_tokens_to_ids(old_aligned)
    new_ids = tokenizer.convert_tokens_to_ids(new_aligned)
    edit_ids = [EDIT_ACTIONS.get(act, EDIT_ACTIONS["<NULL>"]) for act in edit_actions]

    # Step 4: Truncate or pad to max_len
    def pad(seq, pad_val=tokenizer.pad_token_id):
        return seq[:max_len] + [pad_val] * (max_len - len(seq))

    old_ids = pad(old_ids)
    new_ids = pad(new_ids)
    edit_ids = pad(edit_ids, pad_val=EDIT_ACTIONS["<NULL>"])

    return {
        "old_ids": old_ids,
        "new_ids": new_ids,
        "edit_ids": edit_ids
    }

final_user_embeddings = {}
for user_id in tqdm(filtered_df["user_id"].unique(), desc="Users"):
    user_df = filtered_df[filtered_df["user_id"] == user_id]
    
    file_embeddings = []  # list of [H] tensors
    file_lengths = []     # list of ints (lengths for each file)
    for filename in tqdm(user_df["filename"].unique(), desc=f"Files of user {user_id}", leave=False):
        file_df = user_df[user_df["filename"] == filename].reset_index(drop=True)
        file_reprs = []
        for i in tqdm(range(1, len(file_df)), desc=f"Changes in {filename}", leave=False):
            old_code = file_df.iloc[i - 1]["content"]
            new_code = file_df.iloc[i]["content"]
            if not old_code.strip() or not new_code.strip():
                continue

            processed = preprocess_code_change(old_code, new_code, max_len=256)
            old_ids = torch.tensor([processed['old_ids']], dtype=torch.long)
            new_ids = torch.tensor([processed['new_ids']], dtype=torch.long)
            edit_ids = torch.tensor([processed['edit_ids']], dtype=torch.long)

            # Prepare segment tensor (all zeros if not using segments)
            seg = torch.zeros_like(old_ids)

            with torch.no_grad():
                # bypass target, get encoder hidden states
                emb = model.embedding(old_ids, seg)
                memory_bank = model.encoder(emb, seg)   # [1, L, H]

                # aggregate into a single vector for this change
                change_repr = memory_bank.mean(dim=1)   # [1, H]
                file_reprs.append(change_repr)
        if len(file_reprs) == 0:
            continue
        # Stack all change-level embeddings -> [num_changes, H]
        file_reprs = torch.cat(file_reprs, dim=0)
        print("file_reprs shape:", file_reprs.shape)
        # average over all changes
        file_embedding = file_reprs.mean(dim=0)  # shape: [H]
        file_embeddings.append(file_embedding)

        final_code = file_df.iloc[-1]['content']
        file_length = len(tokenizer.tokenize(final_code))
        file_lengths.append(file_length)
    
    # After processing all files for a user:
    if len(file_embeddings) > 0:
        file_embeddings_tensor = torch.stack(file_embeddings, dim=0)  # [num_files, H]
        file_lengths_tensor = torch.tensor(file_lengths, dtype=torch.float32)  # [num_files]
        weights = file_lengths_tensor / file_lengths_tensor.sum()  # [num_files]
        project_embedding = (file_embeddings_tensor * weights.unsqueeze(1)).sum(dim=0)  # [H]
        final_user_embeddings[user_id] = project_embedding
        # To test
        break

print("### Aggregate student's embeddings is completed!")

output_file = "../user_CCBERT_file_embeddings.jsonl"

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]  # Ensure nahted sum of their code emtive float
        }
        f.write(json.dumps(json_line) + '\n')
print(f"### Saved user embeddings to {output_file}")