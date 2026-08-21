'''
Step 2.5 — Ingest MSMARCO-XI, multi-language.

Reads N rows from each requested language's cached parquet file,
extracts BOTH the English fields (Eng_Query/Eng_Answer/English_passages)
AND the native-language fields (query/Answer/Translated_passages),
tags every passage with a `language` code, dedupes into one flat
corpus across all requested languages, and writes:

  data/processed/corpus.jsonl        -> one line per unique passage, tagged with `language`
  data/processed/eval_queries.jsonl  -> one line per query (both English AND native-language
                                          variants of the same underlying query, each tagged)

Run:
  python src/ingest.py --languages eng,hin,ben,guj --n-rows 5000
'''

import argparse
import hashlib
import json
import os

import duckdb

# filename prefix per language, matching your cached snapshot filenames exactly.
# "eng" is not a real MSMARCO-XI file — English content is pulled from the
# Eng_Query/Eng_Answer/English_passages fields present in EVERY language file,
# so we just use hin's file as the English source (any file would do — English
# content is identical/duplicated across all language files).
LANG_FILES = {
    "hin": "hintrain.parquet",
    "ben": "bentrain.parquet",
    "guj": "gujtrain.parquet",
    "tam": "tamtrain.parquet",
    "kan": "kantrain.parquet",
    "mal": "maltrain.parquet",
    "mar": "martrain.parquet",
    "nep": "neptrain.parquet",
    "ori": "oritrain.parquet",
    "pan": "pantrain.parquet",
    "san": "santrain.parquet",
    "urd": "urdtrain.parquet",
    "asm": "asmtrain.parquet",
}

PARQUET_DIR = (
    r"C:\Users\pauls\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI"
    r"\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train"
)


def make_chunk_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def ingest_language(con, lang_code: str, n_rows: int, seed: int,
                     corpus: dict, eval_queries: list, include_english: bool):
    """
    Pulls n_rows from lang_code's parquet file. Always extracts the
    native-language fields (query/Answer/Translated_passages), tagged
    `language=lang_code`. If include_english is True (only done once,
    for the first language processed), ALSO extracts Eng_Query/Eng_Answer/
    English_passages tagged `language=eng`, since English content is
    duplicated identically across every language file.
    """
    path = os.path.join(PARQUET_DIR, LANG_FILES[lang_code])
    query = f"""
        SELECT
            query_id,
            query               AS native_query,
            Answer               AS native_answer,
            Eng_Query,
            Eng_Answer,
            query_type,
            passages.Translated_passages AS native_passages,
            passages.English_passages    AS eng_passages,
            passages.is_selected         AS is_selected
        FROM read_parquet('{path.replace(chr(92), "/")}')
        USING SAMPLE {n_rows} (reservoir, {seed})
    """
    rows = con.execute(query).fetchall()
    col_names = [d[0] for d in con.description]

    n_native = 0
    n_eng = 0

    for row in rows:
        r = dict(zip(col_names, row))
        is_selected = r["is_selected"] or []

        # --- native-language extraction ---
        native_query = (r["native_query"] or "").strip()
        native_answer = (r["native_answer"] or "").strip()
        native_passages = r["native_passages"] or []

        if native_query and native_passages:
            gold_ids, row_ids = [], []
            for text, flag in zip(native_passages, is_selected):
                if not text or not text.strip():
                    continue
                cid = make_chunk_id(text)
                corpus[cid] = {"text": text.strip(), "language": lang_code}
                row_ids.append(cid)
                if flag == 1:
                    gold_ids.append(cid)
            eval_queries.append({
                "query_id": r["query_id"],
                "language": lang_code,
                "query": native_query,
                "gold_answer": native_answer if native_answer not in ("", "No Answer Present.") else None,
                "gold_chunk_ids": gold_ids,
                "candidate_chunk_ids": row_ids,
                "query_type": r["query_type"],
                "has_answer": bool(gold_ids) and native_answer not in ("", "No Answer Present."),
            })
            n_native += 1

        # --- English extraction (only for the first language processed) ---
        if include_english:
            eng_query = (r["Eng_Query"] or "").strip()
            eng_answer = (r["Eng_Answer"] or "").strip()
            eng_passages = r["eng_passages"] or []

            if eng_query and eng_passages:
                gold_ids, row_ids = [], []
                for text, flag in zip(eng_passages, is_selected):
                    if not text or not text.strip():
                        continue
                    cid = make_chunk_id(text)
                    corpus[cid] = {"text": text.strip(), "language": "eng"}
                    row_ids.append(cid)
                    if flag == 1:
                        gold_ids.append(cid)
                eval_queries.append({
                    "query_id": f"{r['query_id']}_eng",
                    "language": "eng",
                    "query": eng_query,
                    "gold_answer": eng_answer if eng_answer not in ("", "No Answer Present.") else None,
                    "gold_chunk_ids": gold_ids,
                    "candidate_chunk_ids": row_ids,
                    "query_type": r["query_type"],
                    "has_answer": bool(gold_ids) and eng_answer not in ("", "No Answer Present."),
                })
                n_eng += 1

    print(f"  [{lang_code}] native queries: {n_native}" + (f", english queries: {n_eng}" if include_english else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hin,ben,guj",
                         help="Comma-separated language codes (hin,ben,guj,tam,...). English is always included automatically.")
    parser.add_argument("--n-rows", type=int, default=5000,
                         help="Rows sampled PER language file.")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lang_codes = [c.strip() for c in args.languages.split(",") if c.strip()]

    for c in lang_codes:
        if c not in LANG_FILES:
            raise ValueError(f"Unknown language code '{c}'. Available: {list(LANG_FILES.keys())}")

    con = duckdb.connect()
    corpus = {}
    eval_queries = []

    print(f"Ingesting languages: {lang_codes} + eng (from {lang_codes[0]}'s file), {args.n_rows} rows each")
    for i, lang_code in enumerate(lang_codes):
        ingest_language(con, lang_code, args.n_rows, args.seed, corpus, eval_queries,
                         include_english=(i == 0))

    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for cid, entry in corpus.items():
            f.write(json.dumps({"chunk_id": cid, "text": entry["text"], "language": entry["language"]}, ensure_ascii=False) + "\n")

    eval_path = os.path.join(args.out_dir, "eval_queries.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for eq in eval_queries:
            f.write(json.dumps(eq, ensure_ascii=False) + "\n")

    from collections import Counter
    lang_counts = Counter(e["language"] for e in corpus.values())

    print(f"\nUnique passages in corpus: {len(corpus)}")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count} passages")
    print(f"Eval queries written: {len(eval_queries)}")
    print(f"-> {corpus_path}")
    print(f"-> {eval_path}")


if __name__ == "__main__":
    main()