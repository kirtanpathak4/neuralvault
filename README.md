# NeuralVault — Personal RAG Knowledge Engine

> Turn everything you study into a queryable second brain. 100% local, zero cloud cost.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B%20via%20Groq-purple?style=flat-square)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS%200.4.3-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![CI](https://github.com/kirtanpathak/ai-learning-memory/actions/workflows/eval.yml/badge.svg)

---

## What Is NeuralVault?

NeuralVault is a locally-running RAG (Retrieval Augmented Generation) application that ingests your study material — PDFs, blog posts, articles, raw notes — and lets you ask questions about it in plain English. It answers using **only your own material**, never hallucinated knowledge.

**It is not a chatbot.** If your notes do not contain the answer, it responds with `INSUFFICIENT_EVIDENCE` instead of making something up. This is the core design principle that separates it from generic LLM wrappers.

Think of it as a private second brain: feed it what you learn, and it remembers and explains everything back to you on demand.

---

## Why NeuralVault Is Different

Most RAG projects on GitHub are "chat with your PDF" demos. NeuralVault goes further:

| Feature | Generic RAG | NeuralVault |
|---|---|---|
| LLM cost | OpenAI (paid) | Groq / Llama 3.3 70B (free) |
| Embeddings | OpenAI API | MiniLM local model (no API cost) |
| Vector DB | Pinecone (cloud) | ChromaDB (local, persistent) |
| Search strategy | Vector-only | Hybrid: vector + BM25 keyword |
| Reranking | None | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Topic filtering | No | Yes — filter answers by study topic |
| Deduplication | No | MD5 hash prevents duplicate chunks |
| Hallucination guard | Basic | `INSUFFICIENT_EVIDENCE` marker enforced by prompt and tested in CI |
| Evaluation | None | RAGAS 4-metric + 38-question tiered golden dataset |
| CI/CD quality gate | None | GitHub Actions auto-fails if quality drops below threshold |

---

## Architecture

### Ingestion Pipeline

```
+----------------------------------------------------------------+
|                      INGESTION PIPELINE                        |
|                                                                |
|  PDF / URL / Raw Text                                          |
|        |                                                       |
|        v                                                       |
|  +--------------+   +-------------------+   +--------------+  |
|  |    Loader    |-->|   Text Splitter   |-->|   Embedder   |  |
|  |              |   |                   |   |              |  |
|  |  PyPDF /     |   |  500 chars /      |   | MiniLM-L6v2  |  |
|  |  WebLoader / |   |  50 char overlap  |   |  384-dim vec |  |
|  |  Raw string  |   +-------------------+   +------+-------+  |
|  +--------------+                                  |          |
|                                          +---------+--------+ |
|                                    +-----v-----+   +-------v+ |
|                                    | ChromaDB  |   |  BM25  | |
|                                    |  Vectors  |   | corpus | |
|                                    |  + text   |   | (JSON) | |
|                                    |  + meta   |   +--------+ |
|                                    +-----------+              |
+----------------------------------------------------------------+
```

### Query Pipeline

```
+----------------------------------------------------------------+
|                       QUERY PIPELINE                           |
|                                                                |
|  User Question + optional topic filter                         |
|        |                                                       |
|        +---- Stage 1A: Vector Search (ChromaDB, top 10)        |
|        +---- Stage 1B: BM25 Keyword Search (top 10)            |
|                   |                                            |
|                   v                                            |
|           Stage 2: Merge + Deduplicate (~15-20 candidates)     |
|                   |                                            |
|                   v                                            |
|           Stage 3: Cross-Encoder Reranker                      |
|           Scores every [query, chunk] pair together            |
|           --> Selects top 5 by relevance score                 |
|                   |                                            |
|                   v                                            |
|           Groq / Llama 3.3 70B                                 |
|           Generates grounded answer from top 5 chunks          |
|                   |                                            |
|                   v                                            |
|           Answer with source citations                         |
|           OR INSUFFICIENT_EVIDENCE if context is missing       |
+----------------------------------------------------------------+
```

---

## RAGAS Evaluation Results

NeuralVault is evaluated against a **38-question tiered golden dataset** on every push to `main`.

### Tier 1 — RAGAS Metrics (23 factual + cross-chunk questions)

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | **0.871** | 0.7 | ✅ PASS |
| Answer Relevancy | **0.767** | 0.7 | ✅ PASS |
| Context Precision | **0.974** | 0.7 | ✅ PASS |
| Context Recall | **0.893** | 0.7 | ✅ PASS |

### Tier 2 — Behavioral Detection (15 questions)

| Type | Questions | Result | Gates Build? |
|---|---|---|---|
| Insufficient evidence | 10 | 9/10 (90%) | **Yes** — must be ≥ 70% |
| Adversarial / false-premise | 5 | 2 refused + 3 corrected false premise | No (informational) |

> **Adversarial note:** The LLM correctly *corrects* false premises using retrieved notes
> (e.g., "NeuralVault does NOT use GPT-4 — it uses Llama 3.3 70B via Groq").
> This is better behavior than refusing. These questions are tracked but do not gate the build.

### Golden Dataset Breakdown

| Type | Count | Purpose |
|---|---|---|
| Factual | 15 | Direct questions answerable from notes |
| Insufficient | 10 | Off-topic questions (Docker, SQL, React) — must trigger `INSUFFICIENT_EVIDENCE` |
| Cross-chunk | 8 | Synthesis questions requiring multiple notes combined |
| Adversarial | 5 | False-premise questions about NeuralVault itself |
| **Total** | **38** | |

---

## Project Structure

```
ai-learning-memory/
|
+-- agents/                     # Pipeline agents (business logic)
|   +-- __init__.py
|   +-- ingest.py               # PDF / URL / text --> chunks --> ChromaDB + BM25
|   +-- retrieve.py             # Hybrid search + cross-encoder reranking
|   +-- answer.py               # Prompt building + Groq LLM generation
|
+-- core/                       # Shared infrastructure
|   +-- __init__.py
|   +-- embedder.py             # Local MiniLM embeddings (text --> 384-dim vectors)
|   +-- vectorstore.py          # ChromaDB setup + BM25 storage layer
|
+-- ui/
|   +-- app.py                  # Streamlit web interface
|
+-- config/
|   +-- prompts.yaml            # All prompts, model settings, INSUFFICIENT_EVIDENCE marker
|
+-- eval/
|   +-- golden_dataset.json     # 38 tiered Q&A pairs (factual/insufficient/cross_chunk/adversarial)
|   +-- eval.py                 # Dual scoring: RAGAS metrics + refusal detection
|   +-- seed_knowledge.py       # Seeds the knowledge base for CI and fresh clones
|   +-- eval_results.json       # Latest scores (auto-generated by eval.py)
|
+-- data/
|   +-- transformers.pdf        # Sample study material (PDF ingestion example)
|
+-- chroma_db/                  # Auto-created on first ingestion (gitignored)
|   +-- chroma.sqlite3          # ChromaDB vectors + metadata
|   +-- bm25_corpus.json        # BM25 keyword index
|
+-- .github/
|   +-- workflows/
|       +-- eval.yml            # CI: auto-runs eval on every push to main
|
+-- .env                        # Your GROQ_API_KEY (never committed)
+-- .gitignore
+-- requirements.txt
+-- README.md
```

---

## Getting Started

### Prerequisites

- Python **3.10 or higher**
- A free [Groq API key](https://console.groq.com) — takes 2 minutes, no credit card required

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/kirtanpathak/ai-learning-memory.git
cd ai-learning-memory
```

---

### Step 2 — Create and activate a virtual environment

```bash
# Create the environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **First-run note:** Two local models download automatically:
> - `all-MiniLM-L6-v2` (~80 MB) — embedding model, runs entirely on CPU
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) — reranker model
>
> These are one-time downloads cached at `~/.cache/torch/sentence_transformers/`.

---

### Step 4 — Create your `.env` file

Create a file named `.env` in the project root (same directory as `README.md`):

```
GROQ_API_KEY=your_groq_key_here
USER_AGENT=neuralvault/1.0
```

Get your free Groq key at [console.groq.com](https://console.groq.com):
> Log in → API Keys → Create API Key → Copy

---

### Step 5 — Launch the app

```bash
streamlit run ui/app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## How to Use

### Ingesting Content

1. Open the **left sidebar** in the Streamlit app
2. Enter a **Topic Label** (e.g., `RAG`, `Transformers`, `Kafka`) — enables filtered searches later
3. Choose a source type:
   - **PDF** — upload any research paper, textbook, or document (multi-page supported)
   - **URL** — paste any public blog post, documentation page, or article URL
   - **Raw Text** — paste notes, summaries, or copied text directly
4. Click **Ingest**

Content is chunked into 500-character pieces with 50-character overlap, embedded, and stored permanently in ChromaDB and the BM25 index. Ingesting the same content twice is safe — MD5 deduplication prevents duplicate chunks.

### Asking Questions

1. Type your question in the **QUERY** section
2. Optionally select a **Topic** to search only within notes tagged with that label
3. Click **Ask NeuralVault**

The answer appears with numbered source chips showing which notes it came from and a similarity score for each chunk.

If the LLM cannot answer from your notes, it returns `INSUFFICIENT_EVIDENCE: [explanation]` instead of guessing.

### Tips

- Use consistent topic labels to enable precise filtering later
- Ingest the same topic from multiple sources to build richer context
- Similarity scores above 0.7 indicate a well-grounded answer
- Ask follow-up questions with increasing specificity for deeper recall

---

## Running the Evaluation

Verify your installation or test changes by running the full evaluation locally:

### Step 1 — Seed the knowledge base

The `chroma_db/` folder is gitignored. Run this once on a fresh clone to load the notes the golden dataset questions refer to:

```bash
python eval/seed_knowledge.py
```

### Step 2 — Run the evaluation

```bash
python eval/eval.py --threshold 0.7
```

Results are printed to the console and saved to `eval/eval_results.json`.

**What the eval does:**
- Loads all 38 questions from `eval/golden_dataset.json`
- Factual + cross-chunk questions → RAGAS pipeline (faithfulness, answer_relevancy, context_precision, context_recall)
- Insufficient + adversarial questions → refusal detection (checks for `INSUFFICIENT_EVIDENCE` marker)
- Exits with code `1` if any RAGAS metric or insufficient detection rate is below `--threshold`

**Expected runtime:** 45–90 minutes (sequential mode to stay within Groq's free-tier rate limits).

---

## CI / CD

Every push to `main` and every pull request triggers an automated evaluation via GitHub Actions.

**Workflow:** `.github/workflows/eval.yml`

**What CI does:**
1. Checks out the repository
2. Sets up Python 3.11
3. Restores pip packages and embedding model from cache (faster on repeat runs)
4. Installs `requirements.txt`
5. Seeds the knowledge base (`python eval/seed_knowledge.py`)
6. Runs `python eval/eval.py --threshold 0.7`
7. Uploads `eval_results.json` as a downloadable artifact

**Build fails if:**
- Any RAGAS metric (faithfulness, answer_relevancy, context_precision, context_recall) drops below **0.7**
- Insufficient evidence detection rate drops below **0.7**

**One-time GitHub setup:**

> Repository → Settings → Secrets and variables → Actions → New repository secret
> Name: `GROQ_API_KEY` | Value: your Groq API key

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Web UI | Streamlit | Fastest Python → web app, no HTML required |
| LLM (generation) | Llama 3.3 70B via Groq | Free tier, fast inference, high quality |
| LLM (RAGAS judge) | Llama 3.1 8B via Groq | Separate Groq quota, sufficient for scoring |
| Embeddings | all-MiniLM-L6-v2 | Runs on CPU, no API cost, 384-dim vectors |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | More accurate than cosine similarity alone |
| Vector DB | ChromaDB | Local, persistent on disk, zero infrastructure |
| Keyword search | BM25 (rank_bm25) | Catches exact terms that vector search misses |
| PDF parsing | PyPDF + LangChain | Clean multi-page PDF extraction |
| URL parsing | WebBaseLoader + BeautifulSoup | Fetches and parses any public webpage |
| Text splitting | RecursiveCharacterTextSplitter | Respects sentence boundaries |
| Evaluation | RAGAS 0.4.3 | Industry-standard RAG evaluation framework |
| Prompt config | YAML (config/prompts.yaml) | Prompts and model settings versioned separately from code |
| Environment | python-dotenv | API keys out of code, loaded from .env |

---

## Requirements

```
bs4
chromadb
datasets
groq
langchain-community
langchain-text-splitters
openai
pypdf
pyyaml
ragas==0.4.3
rank_bm25
sentence-transformers
streamlit
python-dotenv
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---|---|
| RAG (Retrieval Augmented Generation) | Full pipeline: agents/ingest.py, retrieve.py, answer.py |
| Vector Embeddings | core/embedder.py — MiniLM converts text to 384-dim semantic vectors |
| Hybrid Search | agents/retrieve.py — vector cosine + BM25 keyword combined |
| Cross-Encoder Reranking | agents/retrieve.py — ms-marco scores [query, chunk] pairs together |
| Cosine Similarity | ChromaDB nearest-neighbor search on embedded query vectors |
| Hallucination Prevention | INSUFFICIENT_EVIDENCE marker enforced by system prompt, verified in CI |
| RAGAS Evaluation | eval/eval.py — 4 metrics across 23 questions |
| Behavioral Testing | 4-tier golden dataset: factual, insufficient, cross-chunk, adversarial |
| CI/CD Quality Gates | .github/workflows/eval.yml — auto-fails if quality drops |
| Prompt Engineering | config/prompts.yaml — structured prompts with context injection |
| Local AI | All embeddings and reranking run on CPU, zero cloud dependency |
| Persistent Storage | ChromaDB + BM25 JSON persist across sessions without a server |

---

## Roadmap

- [ ] Conversation history — multi-turn chat that remembers previous questions
- [ ] Topic overview — dashboard showing all ingested topics and chunk counts
- [ ] Export notes — generate a summary PDF of everything stored on a topic
- [ ] YouTube ingestion — ingest video transcripts via YouTube URL
- [ ] Streaming responses — show LLM output as it generates (Groq streaming API)
- [ ] Delete / update content — remove or re-ingest outdated notes

---

## Author

**Kirtan Pathak**
Data Engineer II @ State of Utah
[LinkedIn](https://linkedin.com/in/kirtanpathak4) · [GitHub](https://github.com/kirtanpathak)

---

## License

MIT License — free to use, modify, and distribute.
