import psycopg2
from tqdm import tqdm

# --- Remote DB connection info ---
REMOTE_CONFIG = {
    'dbname': 'lmscore',
    'user': 'postgres',
    'password': '***REMOVED***',
    'host': '2af95d8b-2d37-4feb-9925-9517d7905eb3.hsvc.ir',
    'port': '31808'
}

LOCAL_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "code_recorder_s04",
    "user": "postgres",
    "password": "***REMOVED***"
}

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
