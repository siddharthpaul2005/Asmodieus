import sys
import duckdb

sys.stdout.reconfigure(encoding='utf-8')

path = r"C:\Users\pauls\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"

con = duckdb.connect()

print("=== SCHEMA ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchdf())

print("\n=== SAMPLE ROWS ===")
rows = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 3").fetchdf().to_dict(orient="records")
for r in rows:
    print(r)
    print("---")

print("\n=== ROW COUNT ===")
print(con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchdf())