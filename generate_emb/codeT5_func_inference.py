import psycopg2
import pandas as pd
import os
import torch
import re
from transformers import AutoTokenizer, T5EncoderModel
from collections import defaultdict
from tqdm import tqdm
import json

# Path to save embeddings
output_file = "user_CodeT5_func_embeddings.jsonl"

# Load CodeT5 model
MODEL_NAME = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5EncoderModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("### Load CodeT5 model is completed!")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.db_config import get_code_recorder_config

conn = psycopg2.connect(**get_code_recorder_config())

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
MAX_LENGTH = 512  # Truncate long functions


def get_codet5_embedding(code_snippet):
    tokens = tokenizer(code_snippet, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
        # Mean-pooling of token embeddings
        last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        embedding = last_hidden_state.mean(dim=1).squeeze(0)  # shape: (hidden_dim,)
    return embedding.cpu()


def extract_functions_from_file(content):
    FUNC_PATTERN = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_:<>~*&\s]*?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE
    )
    functions = []
    for match in FUNC_PATTERN.finditer(content):
        start = match.start()
        brace_count = 1
        i = content.find('{', start) + 1
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1
        func_code = content[start:i]
        functions.append(func_code)
    return functions

# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings (functions of the final snapshot per file)
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not(filename.endswith("cpp") or filename.endswith("hpp")):
        continue
    final_code = group.iloc[-1]['content']
    if not final_code.strip():
        continue
    functions = extract_functions_from_file(final_code)
    for func in functions:
        embedding = get_codet5_embedding(func)
        user_embeddings[user_id].append(embedding)
        word_count = func.count(' ') + 1
        lengths[user_id].append(word_count)

print("### Generate embeddings is completed!")


# Aggregate per user (weighted sum of their function embeddings)
final_user_embeddings = {}
for user_id, emb_list in user_embeddings.items():
    weights = torch.tensor(lengths[user_id], dtype=torch.float32)
    weights = weights / weights.sum()
    stacked = torch.stack(emb_list)
    final_user_embeddings[user_id] = torch.sum(weights[:, None] * stacked, dim=0)

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")

conn.close()
