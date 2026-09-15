import psycopg2
import pandas as pd
import os
import torch
import re
import tempfile
from clang.cindex import Index, CursorKind, Config
from transformers import AutoTokenizer, T5EncoderModel
from collections import defaultdict
from tqdm import tqdm
import json


# Path to save embeddings
output_file = "user_CodeT5_parser_embeddings.jsonl"

# set libclang path if not found automatically
Config.set_library_file("/home/hadi/libclang-minimal/lib/libclang.so.16.0.4")

# Load CodeT5 model
MODEL_NAME = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5EncoderModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CodeT5 model is completed!")

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

# Assume df is already in correct chronological order
# Columns: user_id, filename, content
print("### Structured data (dataframe) is ready!")

# Parameters
MAX_LENGTH = 512  # Truncate long code files


def get_codet5_embedding(code_snippet):
    tokens = tokenizer(code_snippet, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
        # Mean-pooling of token embeddings
        last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        embedding = last_hidden_state.mean(dim=1).squeeze(0)  # shape: (hidden_dim,)
    return embedding.cpu()

def extract_functions_from_string(code_str):
    # Write the string to a temporary file because libclang works on files
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode='w', encoding='utf-8') as tmp:
        tmp.write(code_str)
        tmp_path = tmp.name

    index = Index.create()
    tu = index.parse(tmp_path, args=['-std=c++17'])  # use C++17 standard;

    functions = []

    def visit(node):
        if node.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD]:
            # Get the function source code text by extent
            start = node.extent.start.offset
            end = node.extent.end.offset
            func_code = code_str[start:end]
            functions.append(func_code)
        for c in node.get_children():
            visit(c)

    visit(tu.cursor)

    # Clean up temp file
    os.unlink(tmp_path)

    return functions


# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings
# Assuming sorted_df is already a DataFrame
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not(filename.endswith("cpp") or filename.endswith("hpp")):
        continue
    final_code = group.iloc[-1]['content']
    if not final_code.strip():
        continue
    functions = extract_functions_from_string(final_code)
    for func in functions:
        embedding = get_codet5_embedding(func)
        user_embeddings[user_id].append(embedding)
        word_count = func.count(' ') + 1
        lengths[user_id].append(word_count)

print("### Generate embeddings is completed!")

# Aggregate per user (weighted sum of their code embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    final_user_embeddings[user_id] = torch.sum(weights[:, None] * torch.stack(emb_list), dim=0)  # shape: (hidden_dim=768,)

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]  # Ensure nahted sum of their code emtive float
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")