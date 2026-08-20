'''

This script does the subsample + dedupe + eval-set creation from hintrain.parquet (English fields), producing the actual working corpus for everything downstream.

What it needs to do:

Read N rows from the parquet (English fields only, per our decision).
From each row, pull out the 10 passages + the is_selected mask + the query + the gold answer.
Dedupe passages — same passage text often appears across many different rows as a distractor, we only want to store it once, but remember every query that treats it as gold.
Save two artifacts:
corpus.jsonl — every unique passage, with a stable chunk_id, ready for step 3 (chunking) and step 4 (embedding).
eval_queries.jsonl — every query, its gold chunk_id(s), and the gold answer text — this is what step 9 (latency benchmarking) and retrieval-accuracy testing will run against.
'''

"""
Step 2.5 — Ingest MSMARCO-XI (Hindi-split file, English fields).

Reads N rows from the cached hintrain.parquet, extracts the English
query/passages/answer fields, dedupes passages into a flat corpus,
and writes two files:

  data/processed/corpus.jsonl        -> one line per unique passage
  data/processed/eval_queries.jsonl  -> one line per query, with gold chunk_id(s)

Run:
  python src/ingest.py --n-rows 5000
"""

import argparse
import hashlib
import json
import os

import duckdb


# Update this if your cached path differs — copy it from your
# hf_hub_download output.
DEFAULT_PARQUET_PATH = (
    r"C:\Users\pauls\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI"
    r"\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"
)


def make_chunk_id(text: str) -> str:
    """Stable id for a passage based on its content, so the same
    passage text always maps to the same chunk_id even across rows."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--n-rows", type=int, default=5000,
                         help="Number of query rows to sample from the parquet.")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    con = duckdb.connect()

    # Pull only the English-relevant columns, sampled deterministically.
    query = f"""
        SELECT
            query_id,
            Eng_Query,
            Eng_Answer,
            query_type,
            passages.English_passages   AS eng_passages,
            passages.is_selected        AS is_selected
        FROM read_parquet('{args.parquet_path}')
        USING SAMPLE {args.n_rows} (reservoir, {args.seed})
    """
    rows = con.execute(query).fetchall()
    col_names = [d[0] for d in con.description]

    corpus = {}          # chunk_id -> passage text
    eval_queries = []     # list of dicts

    skipped_no_gold = 0
    skipped_no_query = 0

    for row in rows:
        r = dict(zip(col_names, row))

        eng_query = (r["Eng_Query"] or "").strip()
        eng_answer = (r["Eng_Answer"] or "").strip()
        passages = r["eng_passages"] or []
        is_selected = r["is_selected"] or []

        if not eng_query or not passages:
            skipped_no_query += 1
            continue

        gold_chunk_ids = []
        row_chunk_ids = []

        for passage_text, flag in zip(passages, is_selected):
            if not passage_text or not passage_text.strip():
                continue
            cid = make_chunk_id(passage_text)
            corpus[cid] = passage_text.strip()
            row_chunk_ids.append(cid)
            if flag == 1:
                gold_chunk_ids.append(cid)

        if not gold_chunk_ids:
            # No passage marked as gold for this query (e.g. "No Answer
            # Present." rows) — still useful for guardrail testing later
            # (these are exactly the queries that SHOULD trigger a
            # "not enough information" refusal), so we keep them but
            # tag them explicitly.
            skipped_no_gold += 1

        eval_queries.append({
            "query_id": r["query_id"],
            "query": eng_query,
            "gold_answer": eng_answer if eng_answer != "No Answer Present." else None,
            "gold_chunk_ids": gold_chunk_ids,
            "candidate_chunk_ids": row_chunk_ids,
            "query_type": r["query_type"],
            "has_answer": bool(gold_chunk_ids) and eng_answer != "No Answer Present.",
        })

    # Write corpus.jsonl
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for cid, text in corpus.items():
            f.write(json.dumps({"chunk_id": cid, "text": text}, ensure_ascii=False) + "\n")

    # Write eval_queries.jsonl
    eval_path = os.path.join(args.out_dir, "eval_queries.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for eq in eval_queries:
            f.write(json.dumps(eq, ensure_ascii=False) + "\n")

    print(f"Sampled rows:              {len(rows)}")
    print(f"Skipped (no query/passages): {skipped_no_query}")
    print(f"Rows with no gold passage:  {skipped_no_gold}  (kept, useful for refusal testing)")
    print(f"Unique passages in corpus:  {len(corpus)}")
    print(f"Eval queries written:       {len(eval_queries)}")
    print(f"-> {corpus_path}")
    print(f"-> {eval_path}")


if __name__ == "__main__":
    main()