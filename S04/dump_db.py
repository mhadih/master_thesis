import os
import sys
import psycopg2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.db_config import get_remote_config, get_s04_config

# --- Remote DB connection info (from environment variables / .env) ---
REMOTE_CONFIG = get_remote_config()

LOCAL_CONFIG = get_s04_config()

# Tables to copy
TABLES = ["UserAttributes"]

# Connect to remote DB
remote_conn = psycopg2.connect(**REMOTE_CONFIG)
remote_cur = remote_conn.cursor()

# Connect to local DB
local_conn = psycopg2.connect(**LOCAL_CONFIG)
local_cur = local_conn.cursor()

for table in TABLES:
    print(f"📥 Fetching data from remote table '{table}'...")

    batch_size = 1000
    all_rows = []
    remote_cur.execute(f'SELECT COUNT(*) FROM \"{table}\";')
    total_rows = remote_cur.fetchone()[0]

    remote_cur.execute(f'SELECT * FROM \"{table}\";')

    with tqdm(total=total_rows) as pbar:
        while True:
            rows = remote_cur.fetchmany(batch_size)
            if not rows:
                break
            all_rows.extend(rows)
            pbar.update(len(rows))

    colnames = [f"\"{desc[0]}\"" for desc in remote_cur.description]
    placeholders = ", ".join(["%s"] * len(colnames))
    collist = ", ".join(colnames)

    # print(f"🗑 Creating local table '{table}'...")
    # local_cur.execute(f"CREATE TABLE {table};")
    
    print(f"🗑 Clearing local table '{table}'...")
    local_cur.execute(f"TRUNCATE TABLE \"{table}\" RESTART IDENTITY CASCADE;")

    print(f"📤 Inserting {len(all_rows)} rows into local table '{table}'...")
    for row in all_rows:
        local_cur.execute(
            f"INSERT INTO \"{table}\" ({collist}) VALUES ({placeholders});",
            row
        )

    local_conn.commit()
    print(f"✅ Table '{table}' copied successfully.\n")

# Close connections
remote_cur.close()
remote_conn.close()
local_cur.close()
local_conn.close()
