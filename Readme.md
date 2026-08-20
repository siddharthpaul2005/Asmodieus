my disign fo the repo structure 

your-repo/
├── data/
│   ├── raw/                  # cached parquet (gitignored, too big)
│   └── processed/            # our subsampled + deduped corpus (small, can commit)
├── src/
│   ├── ingest.py             # step 2.5: subsample + dedupe + build corpus
│   ├── chunking/             # step 3
│   ├── embedding/            # step 4
│   ├── retrieval/            # step 5
│   ├── answering/            # step 6
│   ├── guardrails/           # step 7
│   ├── harness.py            # step 8
│   └── stt/                  # step 10, empty until teammate's ready
├── benchmarks/
│   └── latency.py            # step 9
├── server/
│   └── main.py                # step 11, FastAPI
├── requirements.txt
├── .gitignore
└── README.md

the whole roadmap
( without akshat's STT and deployment pipelines and stuff)

full build order

The full build order
Inspect the dataset — understand actual fields/splits before designing anything around assumptions
Environment setup — repo structure, dependencies, so the whole team can pull and run the same thing
Chunking module — the multi-strategy indexer (fixed+overlap, semantic, sentence-window, metadata-tagged)
Embedding + vector index — quantized embeddings, in-memory HNSW, build the index from chunks
Retrieval logic — hybrid dense+sparse, RRF fusion, tuned for speed
Extractive answering — the fast-path span extraction that actually hits the ms budget
Guardrails — off-topic detection, grounding check, refuse-path logic
Harness — wrap all of the above in the state machine: timeouts, retries, structured I/O, fallback paths
Latency instrumentation — timers on every stage, percentile computation, test-query harness to generate P50/P70/P100
STT plug-in slot — thin interface ready to receive whichever provider your team picks; wire it in when ready
Server + local run — FastAPI wrapping the harness, test end-to-end on localhost
Deployment — akshat said render ( his part)
README + submission writeup — the honest latency-scoping explanation, architecture diagram, guardrail demo notes


current stithi

step by step developement er plan 
Step	Status
1. Dataset inspection	(yes)
2. Repo + env setup	(yes)
3. Chunking module	(not started yet)
4. Embedding + vector index	(not started yet)
5. Retrieval (hybrid + RRF)	(not started yet)
6. Extractive answering	(not started yet)
7. Guardrails	(not started yet)
8. Harness (state machine)	(not started yet)
9. Latency instrumentation	(not started yet)
10. STT plug-in	(not started yet)
11. FastAPI server, local run	(not started yet)
12. Render deployment	(not started yet)
13. README + writeup	(not started yet)