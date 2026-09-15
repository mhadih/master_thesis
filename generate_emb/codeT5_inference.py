import psycopg2
import pandas as pd
import os
import torch
import torch.nn as nn
import re
# import tempfile
# from clang.cindex import Index, CursorKind, Config
from transformers import AutoTokenizer, T5EncoderModel
from collections import defaultdict
from tqdm import tqdm
import json


# Path to save embeddings
output_file = "user_CodeT5_file_with_LSTM.jsonl"

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


# Storage for user embeddings
user_embeddings = defaultdict(list)
lengths = defaultdict(list)

# Generate embeddings
# Assuming sorted_df is already a DataFrame
grouped = sorted_df.groupby(['user_id', 'filename'], sort=False)

for (user_id, filename), group in tqdm(grouped, desc="Processing groups", unit="group"):
    if not(filename.endswith("cpp")):
        continue
    final_code = group.iloc[-1]['content']
    if not final_code.strip():
        continue
    embedding = get_codet5_embedding(final_code)
    user_embeddings[user_id].append(embedding)
    word_count = final_code.count(' ') + 1
    lengths[user_id].append(word_count)

print("### Generate embeddings is completed!")

# # Aggregate per user (weighted sum of their code embeddings)
# final_user_embeddings = {}
# for user_id, emb_list in user_embeddings.items():
#     weights = torch.tensor(lengths[user_id], dtype=torch.float32)
#     weights = weights / weights.sum()
#     final_user_embeddings[user_id] = torch.sum(weights[:, None] * torch.stack(emb_list), dim=0)  # shape: (hidden_dim=768,)

# use LSTM
EMBEDDING_SIZE = 768

class ChangeSequenceAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, sequence_embeddings):
        # sequence_embeddings: [1, num_files, input_dim]
        _, (h_n, _) = self.lstm(sequence_embeddings)
        return h_n[-1].squeeze(0)  # shape [hidden_dim]

final_user_embeddings = {}
agg_model = ChangeSequenceAggregator(input_dim=EMBEDDING_SIZE)

for user_id, emb_list in tqdm(user_embeddings.items(), desc="LSTM inference", unit="user"):
    # emb_list: List[Tensor] with shape [input_dim]
    sequence_tensor = torch.stack(emb_list).unsqueeze(0)  # [1, num_files, input_dim]
    project_embedding = agg_model(sequence_tensor)  # [hidden_dim]
    final_user_embeddings[user_id] = project_embedding
print("### Aggregate student's embeddings is completed!")

with open(output_file, 'w') as f:
    for user_id, emb in final_user_embeddings.items():
        emb_list = emb.tolist()  # Convert tensor to native Python list
        json_line = {
            "user_id": str(user_id),   # Convert to string to avoid int64 issues
            "embedding": [float(x) for x in emb_list]  # Ensure nahted sum of their code emtive float
        }
        f.write(json.dumps(json_line) + '\n')

print(f"### Saved user embeddings to {output_file}")