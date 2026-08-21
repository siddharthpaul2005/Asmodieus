import json
import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "asmodieus")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY environment variable is missing.")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Path selection: chunk_meta.jsonl if present, else corpus.jsonl
meta_path = os.path.join("data", "processed", "chunk_meta.jsonl")
corpus_path = os.path.join("data", "processed", "corpus.jsonl")

file_to_load = meta_path if os.path.exists(meta_path) else corpus_path
print(f"Loading data from: {file_to_load}", flush=True)

records = []
with open(file_to_load, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = row.get("chunk_id") or str(row.get("idx"))
        text = row.get("text", "").strip()
        if not text or not cid:
            continue
        records.append({
            "_id": str(cid),
            "text": text,
            "language": row.get("language", "eng"),
            "strategy": row.get("strategy", "passage"),
        })

print(f"Total records loaded: {len(records)}", flush=True)

BATCH_SIZE = 90
total_uploaded = 0

t0 = time.time()
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    try:
        response = index.upsert_records(namespace=NAMESPACE, records=batch)
        total_uploaded += len(batch)
        if (i + BATCH_SIZE) % 1000 == 0 or (i + BATCH_SIZE) >= len(records):
            print(f"Uploaded {total_uploaded}/{len(records)} records...", flush=True)
    except Exception as e:
        print(f"Error uploading batch at index {i}: {e}", flush=True)

t1 = time.time()
print(f"Done! Successfully uploaded {total_uploaded} records in {t1 - t0:.2f} seconds.", flush=True)
