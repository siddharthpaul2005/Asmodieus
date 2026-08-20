# Asmodieus Architecture & Implementation Roadmap

## Repository Structure

```text
Asmodieus/
├── data/
│   ├── raw/                  # Cached parquet (gitignored, raw data)
│   └── processed/            # Subsampled + deduped corpus (committed)
├── src/
│   ├── ingest.py             # Data ingestion, subsampling & deduplication
│   ├── chunking/             # Multi-strategy text chunking modules
│   ├── embedding/            # Quantized embedding generation & HNSW index
│   ├── retrieval/            # Hybrid dense/sparse retrieval & RRF fusion
│   ├── answering/            # Fast-path extractive span extraction
│   ├── guardrails/           # Off-topic detection & grounding validation
│   ├── harness.py            # Core orchestration state machine & fallbacks
│   └── stt/                  # Speech-to-text interface module
├── benchmarks/
│   └── latency.py            # Stage-by-stage latency tracking (P50/P70/P100)
├── server/
│   └── main.py               # FastAPI application entry point
├── requirements.txt
├── .gitignore
└── README.md

Build Order
Inspect Dataset — Analyze actual fields and schema splits before designing system assumptions.

Environment Setup — Establish project structure and dependency specs for unified team execution.

Chunking Module — Implement multi-strategy indexer (fixed + overlap, semantic, sentence-window, metadata-tagged).

Embedding & Vector Index — Set up quantized embeddings and in-memory HNSW vector index construction.

Retrieval Logic — Implement hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF) optimized for speed.

Extractive Answering — Build fast-path span extraction to meet sub-millisecond response budgets.

Guardrails — Implement off-topic detection, grounding validation, and query refusal execution paths.

Harness State Machine — Wrap modules with timeout management, structured I/O, retry logic, and fallback branches.

Latency Instrumentation — Add stage timers, percentile tracking (P50/P70/P100), and automated benchmarking query suits.

STT Plug-in Slot — Expose a thin interface ready for Speech-To-Text integration.

Server & Local Run — Wrap harness in a FastAPI web server for local testing.

Deployment Pipeline — Deploy service to Render platform.

Documentation & Submission — Finalize README, architecture flowcharts, latency breakdown, and guardrail notes.

Step,Module / Milestone,Status
1,Dataset Inspection,Completed
2,Repo & Environment Setup,Completed
3,Chunking Module,Pending
4,Embedding & Vector Index,Pending
5,Retrieval (Hybrid + RRF),Pending
6,Extractive Answering,Pending
7,Guardrails,Pending
8,Harness (State Machine),Pending
9,Latency Instrumentation,Pending
10,STT Plug-in Interface,Pending
11,FastAPI Server (Local Run),Pending
12,Render Deployment,Pending
13,README & Submission Writeup,Pending