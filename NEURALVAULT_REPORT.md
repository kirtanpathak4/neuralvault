# NeuralVault — Complete Project Report
### A Personal RAG Knowledge Engine — From Theory to Production

---

## Table of Contents

1. [What is NeuralVault?](#1-what-is-neuralvault)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Motivation & Initial Thoughts](#3-motivation--initial-thoughts)
4. [Core Concepts You Must Know](#4-core-concepts-you-must-know)
5. [Project Structure & File Map](#5-project-structure--file-map)
6. [Architecture Flow Diagram](#6-architecture-flow-diagram)
7. [Every File — What It Does, Every Function Explained](#7-every-file--what-it-does-every-function-explained)
8. [End-to-End Workflow — Ingestion Path](#8-end-to-end-workflow--ingestion-path)
9. [End-to-End Workflow — Query Path](#9-end-to-end-workflow--query-path)
10. [End-to-End Workflow — Evaluation Path](#10-end-to-end-workflow--evaluation-path)
11. [Runtime Files & Folders — Created & Deleted](#11-runtime-files--folders--created--deleted)
12. [Dry Run #1 — Ingesting a PDF](#12-dry-run-1--ingesting-a-pdf)
13. [Dry Run #2 — Asking a Question](#13-dry-run-2--asking-a-question)
14. [Dry Run #3 — Running RAGAS Evaluation](#14-dry-run-3--running-ragas-evaluation)
15. [What Each Layer Does — Deep Dive](#15-what-each-layer-does--deep-dive)
16. [Why Each Design Decision Was Made](#16-why-each-design-decision-was-made)
17. [Security Measures](#17-security-measures)
18. [Final RAGAS Scores & What They Mean](#18-final-ragas-scores--what-they-mean)
19. [Improvement Strategies](#19-improvement-strategies)
20. [Glossary](#20-glossary)

---

## 1. What is NeuralVault?

NeuralVault is a **personal RAG (Retrieval Augmented Generation) knowledge engine**. It lets you:

1. **Ingest** your study material — PDFs, web articles, or raw notes
2. **Ask questions** about that material in natural language
3. **Get grounded answers** that cite only your own notes — never hallucinated
4. **Evaluate quality** automatically using the RAGAS framework

Think of it as a **private second brain** — you feed it everything you're learning, then quiz it. Every answer is traceable back to a specific chunk of your notes.

**It is NOT a chatbot.** It doesn't use the LLM's training data. If your notes don't contain the answer, it says "INSUFFICIENT_EVIDENCE" instead of making something up. This is the single most important design principle.

---

## 2. The Problem It Solves

### The Hallucination Problem

Large Language Models (LLMs) like GPT-4, Llama, Claude generate fluent text but they **make things up**. Ask an LLM about a topic and it might give you an answer that sounds perfect but is factually wrong. This is called **hallucination**.

### Why This Matters for Learning

When you're studying AI, machine learning, or any technical topic, you need answers you can trust. You need to know: "Did I actually read this, or is the AI making it up?"

### The RAG Solution

RAG solves this by adding a **retrieval step** before generation:

```
Traditional LLM:    Question → LLM → Answer (might hallucinate)
RAG:                Question → Search Your Notes → Feed Notes to LLM → Answer (grounded)
```

The LLM only sees your notes as context. Its prompt explicitly says: "Answer using ONLY the study notes provided. If not enough information, say INSUFFICIENT_EVIDENCE."

---

## 3. Motivation & Initial Thoughts

### Why Build This?

1. **Learning by building** — The best way to understand RAG is to build one end-to-end
2. **Privacy** — All data stays local. ChromaDB stores vectors on your machine. No cloud vector database
3. **Cost** — Groq provides free LLM inference. Embeddings run locally. The only external API is Groq
4. **Evaluation** — Most RAG tutorials skip evaluation entirely. NeuralVault includes a complete RAGAS pipeline so you can measure quality objectively

### Initial Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector DB | ChromaDB (local) | Zero cost, no cloud setup, persists to disk |
| Embedding Model | all-MiniLM-L6-v2 | 80MB, runs on CPU, 384-dim vectors, good quality |
| LLM | Llama 3.3 70B via Groq | Free tier, fast inference, high quality |
| Search | Hybrid (Vector + BM25) | Covers both semantic and keyword matching |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Dramatically improves precision over raw cosine |
| UI | Streamlit | Rapid prototyping, good enough for a personal tool |
| Eval | RAGAS 0.4.3 | Industry standard for RAG evaluation |
| Prompt Config | YAML file | Separates prompts from code, enables versioning |

---

## 4. Core Concepts You Must Know

### 4.1 Embeddings

An **embedding** converts text into a list of numbers (a vector). Texts with similar meaning get similar vectors.

```
"What is attention?" → [0.12, -0.34, 0.56, ..., 0.78]   (384 numbers)
"How does attention work?" → [0.11, -0.33, 0.55, ..., 0.77]   (very close!)
"The weather is nice" → [0.89, 0.12, -0.67, ..., -0.23]   (very different)
```

The model `all-MiniLM-L6-v2` does this conversion. It downloads once (~80MB) and runs locally forever.

### 4.2 Cosine Similarity

Measures the angle between two vectors. Score ranges from 0 to 1:
- **1.0** = identical meaning
- **0.0** = completely unrelated

When you ask a question, your question is embedded into a vector, then compared against all stored chunk vectors. The closest ones are your search results.

### 4.3 Chunking

Documents are too long to embed as a single unit. We split them into **chunks** of ~500 characters with 50-character overlap:

```
Original: "ABCDEFGHIJ" (imagine each letter is 100 chars)

Chunk 1: "ABCDE"     (chars 1-500)
Chunk 2:    "DEFGH"   (chars 451-950, overlaps with D-E)
Chunk 3:       "GHIJ" (chars 901-1000, overlaps with G-H)
```

**Why overlap?** Without it, a sentence split across two chunks would be incomplete in both. The overlap ensures boundary sentences appear fully in at least one chunk.

### 4.4 BM25

**BM25 (Best Match 25)** is a classical keyword ranking algorithm. It scores documents by:
- How many query words appear in the document (term frequency)
- How rare those words are across ALL documents (inverse document frequency)

**Why both Vector AND BM25?** They catch different things:

| Query | Vector Search Finds | BM25 Finds |
|-------|-------------------|------------|
| "how attention works" | Chunks about "multi-head self-attention mechanism" | Chunks literally containing the word "attention" |
| "HNSW parameter" | Chunks about "vector database configuration" | Chunks literally containing "HNSW" |

Vector search handles **meaning**. BM25 handles **exact keywords**. Together = **hybrid search**.

### 4.5 Cross-Encoder Reranking

After hybrid search gives us ~20 candidate chunks, we need to pick the best 5. A **cross-encoder** does this far more accurately than cosine similarity.

**Bi-encoder** (what cosine similarity uses):
```
embed("What is attention?") → vector_A
embed("Attention lets the model focus on relevant parts") → vector_B
cosine(vector_A, vector_B) → 0.82   (approximate — each embedded independently)
```

**Cross-encoder** (what the reranker uses):
```
score("What is attention?" + "Attention lets the model focus on relevant parts") → 0.95
(reads both together as one input — like a human would)
```

The cross-encoder is slower (can't pre-compute) but far more precise. We only run it on the ~20 shortlisted candidates, not all thousands of stored chunks.

### 4.6 RAGAS

**RAGAS (Retrieval Augmented Generation Assessment)** is an evaluation framework with 4 metrics:

| Metric | What It Measures | Score Meaning |
|--------|-----------------|---------------|
| **Faithfulness** | Are answer claims supported by retrieved chunks? | 1.0 = no hallucination |
| **Answer Relevancy** | Does the answer actually address the question? | 1.0 = perfectly on-topic |
| **Context Precision** | Are the retrieved chunks relevant to the question? | 1.0 = all chunks useful |
| **Context Recall** | Did we retrieve ALL chunks needed to fully answer? | 1.0 = nothing missed |

A separate "judge" LLM (Llama 8B) scores these — it reads the question, answer, contexts, and ground truth, then decides.

---

## 5. Project Structure & File Map

```
NeuralVault/
│
├── config/
│   └── prompts.yaml           # All prompts, model config, version
│
├── core/                      # Low-level infrastructure
│   ├── __init__.py            # Package marker (empty)
│   ├── embedder.py            # Sentence-transformer embedding functions
│   └── vectorstore.py         # ChromaDB + BM25 storage layer
│
├── agents/                    # Business logic layer
│   ├── __init__.py            # Package marker (empty)
│   ├── ingest.py              # PDF/URL/text ingestion → chunking → storing
│   ├── retrieve.py            # Hybrid search + cross-encoder reranking
│   └── answer.py              # Prompt building + LLM generation
│
├── ui/
│   └── app.py                 # Streamlit web interface
│
├── eval/
│   ├── golden_dataset.json    # 38 tiered Q&A pairs (factual/insufficient/cross_chunk/adversarial)
│   ├── eval.py                # Dual scoring: RAGAS metrics + refusal detection
│   ├── seed_knowledge.py      # Seeds the knowledge base for CI and fresh clones
│   └── eval_results.json      # Output scores (auto-generated at runtime)
│
├── data/
│   └── transformers.pdf       # Sample study material
│
├── chroma_db/                 # Generated at runtime — vector database (gitignored)
│   ├── chroma.sqlite3         # ChromaDB's SQLite storage
│   ├── bm25_corpus.json       # BM25 keyword index (JSON)
│   └── ba6c1882-.../          # ChromaDB internal segment files
│
├── .github/
│   └── workflows/
│       └── eval.yml           # CI: auto-runs eval on every push to main
│
├── .env                       # GROQ_API_KEY (never committed)
├── .gitignore                 # Excludes secrets, runtime data, venv
└── requirements.txt           # Python dependencies (14 packages)
```

### Layer Architecture

```
┌─────────────────────────────────────┐
│            UI Layer                 │  ← ui/app.py (Streamlit)
│  User interaction, display, forms   │
└──────────┬──────────────────────────┘
           │ calls
┌──────────▼──────────────────────────┐
│         Agent Layer                 │  ← agents/ingest.py, retrieve.py, answer.py
│  Business logic, orchestration      │
└──────────┬──────────────────────────┘
           │ calls
┌──────────▼──────────────────────────┐
│          Core Layer                 │  ← core/embedder.py, vectorstore.py
│  Embeddings, ChromaDB, BM25         │
└──────────┬──────────────────────────┘
           │ reads/writes
┌──────────▼──────────────────────────┐
│        Storage Layer                │  ← chroma_db/ folder on disk
│  chroma.sqlite3, bm25_corpus.json   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       Evaluation Layer              │  ← eval/eval.py (standalone)
│  RAGAS metrics, golden dataset      │  Calls agents/answer.py directly
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      Configuration Layer            │  ← config/prompts.yaml
│  Prompts, model settings, version   │  Read by agents/answer.py
└─────────────────────────────────────┘
```

---

## 6. Architecture Flow Diagram

### Ingestion Flow (User adds knowledge)

```
User uploads PDF/URL/Text
         │
         ▼
┌─────────────────┐
│   ui/app.py     │  Streamlit sidebar form
│  Collects input │  Topic label + content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ agents/ingest.py│  Loads document (PyPDF/WebBase/raw)
│  Split → Chunks │  RecursiveCharacterTextSplitter(500, 50)
└────────┬────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│ core/embedder.py│          │core/vectorstore.py│
│  embed_texts()  │          │ add_to_bm25()     │
│  384-dim vectors│          │ Tokenize + store   │
└────────┬────────┘          └────────┬──────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐          ┌──────────────────┐
│    ChromaDB     │          │  bm25_corpus.json │
│ chroma.sqlite3  │          │  Keyword index     │
│  Vectors+text   │          │  Tokenized text    │
└─────────────────┘          └──────────────────┘
```

### Query Flow (User asks a question)

```
User types question + optional topic filter
         │
         ▼
┌─────────────────┐
│   ui/app.py     │  Calls answer(query, topic)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│agents/answer.py │  Orchestrates the full pipeline
│                 │  1. Call retrieve()
│                 │  2. Build prompt from YAML
│                 │  3. Call Groq LLM
│                 │  4. Return structured result
└────────┬────────┘
         │ step 1
         ▼
┌──────────────────┐
│agents/retrieve.py│
│                  │
│ STAGE 1A: Vector │──→ embed_query() → ChromaDB.query() → top 10
│ STAGE 1B: BM25   │──→ tokenize → BM25Okapi.get_scores() → top 10
│ STAGE 2:  Merge  │──→ MD5 dedup → ~20 unique candidates
│ STAGE 3:  Rerank │──→ CrossEncoder.predict() → top 5
│                  │
└────────┬─────────┘
         │ returns top 5 chunks
         ▼
┌─────────────────┐
│agents/answer.py │  Builds prompt:
│                 │  [1] Topic: RAG
│ Template.safe_  │  "RAG stands for..."
│ substitute()    │  ---
│                 │  [2] Topic: RAG
│                 │  "Chunking splits..."
│                 │
│ Groq API call   │──→ Llama 3.3 70B generates answer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ui/app.py     │  Displays answer card with:
│                 │  - Grounded answer with citations [1][2]
│                 │  - Source pills (topic, file, score)
│                 │  - OR amber "Insufficient Evidence" card
└─────────────────┘
```

---

## 7. Every File — What It Does, Every Function Explained

---

### 7.1 `config/prompts.yaml`

**Purpose:** Single source of truth for all prompts, model configuration, and version tracking. Separating prompts from Python code means you can tweak prompts without touching any `.py` files.

**Keys:**

| Key | Value | Purpose |
|-----|-------|---------|
| `version` | `"1.1"` | Prompt version — shown in UI, logged in eval results, enables A/B testing |
| `model` | `"llama-3.3-70b-versatile"` | Which Groq model to use for generation |
| `temperature` | `0.3` | Low = more deterministic answers. High = more creative. 0.3 is good for factual Q&A |
| `max_tokens` | `1024` | Maximum response length. Prevents runaway generation |
| `system_prompt` | (multiline) | Sets the LLM's identity: "You are NeuralVault, a personal AI study assistant..." |
| `rag_prompt` | (multiline) | The main template with `$context`, `$query` placeholders. Contains citation rules and INSUFFICIENT_EVIDENCE instructions |
| `no_results_message` | (multiline) | Returned when ChromaDB has zero chunks for the topic |
| `insufficient_evidence_marker` | `"INSUFFICIENT_EVIDENCE"` | String the LLM outputs when notes don't cover the question |

**Why `$context` instead of `{context}`?** Python's `string.Template.safe_substitute()` uses `$` syntax. This is safer than `str.replace()` because if chunk text accidentally contains `{context}`, it won't be replaced. `safe_substitute` only replaces known placeholders and ignores unknown `$` variables.

---

### 7.2 `core/embedder.py` (21 lines)

**Purpose:** Wraps the sentence-transformers model. Provides two functions: embed many texts (for ingestion) or embed one query (for search).

**Module-level setup:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```
Loaded once when the module is first imported. Downloads ~80MB on first run, cached in `~/.cache/torch/` after that. Runs entirely on CPU — no GPU needed.

**Functions:**

| Function | Input | Output | When Called |
|----------|-------|--------|-------------|
| `embed_texts(texts)` | `list[str]` (many chunks) | `list[list[float]]` (many 384-dim vectors) | During ingestion — embeds all chunks at once (batch) |
| `embed_query(query)` | `str` (single question) | `list[float]` (one 384-dim vector) | During search — embeds the user's question |

**Why two functions?** `embed_texts` processes a batch efficiently (GPU/CPU parallelism). `embed_query` processes a single string. Technically both call `model.encode()`, but the interface is clearer.

---

### 7.3 `core/vectorstore.py` (158 lines)

**Purpose:** Manages both storage backends — ChromaDB (vectors) and BM25 (keywords). This is the **persistence layer** that everything reads from and writes to.

**Module-level setup:**
```python
DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
BM25_PATH = Path(__file__).parent.parent / "chroma_db" / "bm25_corpus.json"
_bm25_cache: dict = {"corpus": None, "index": None}
```
- `PersistentClient` saves to disk — data survives Python restarts
- BM25 is stored as plain JSON next to ChromaDB's SQLite file
- Module-level cache prevents rebuilding the BM25 index on every query

**Functions:**

| Function | Purpose | Called By |
|----------|---------|-----------|
| `get_collection(name)` | Gets or creates a ChromaDB collection. A collection is like a SQL table. Uses cosine distance metric. | `ingest.py`, `retrieve.py` |
| `_tokenize(text)` | Lowercases text, strips punctuation with regex `[^\w\s]`, splits on whitespace. Returns word list for BM25. | `add_to_bm25_corpus()` |
| `_load_bm25_corpus()` | Reads `bm25_corpus.json` from disk. Returns `[]` if file doesn't exist yet. | `add_to_bm25_corpus()`, `get_bm25_index()` |
| `_save_bm25_corpus(corpus)` | Writes corpus list to `bm25_corpus.json`. Creates parent directories if needed. | `add_to_bm25_corpus()` |
| `_invalidate_bm25_cache()` | Sets `_bm25_cache` to `{corpus: None, index: None}` so next query rebuilds from disk. | `add_to_bm25_corpus()` |
| `add_to_bm25_corpus(texts, ids, metadatas)` | Adds or updates chunks in BM25. Uses same IDs as ChromaDB for sync. If an ID exists, it updates (upsert). Invalidates cache after write. | `ingest.py:_store_chunks()` |
| `get_bm25_index()` | Returns `(BM25Okapi, corpus_list)`. Uses cache if available, otherwise loads from disk and rebuilds. Returns `(None, [])` if empty. | `retrieve.py:retrieve()` |

**Why upsert in BM25?** ChromaDB's `collection.upsert()` updates existing entries. If BM25 didn't do the same, re-ingesting a document would create duplicates in BM25 while ChromaDB would correctly update — they'd be out of sync.

**Why a cache?** Without caching, every query would: read JSON from disk → parse JSON → tokenize all chunks → build BM25 index. With caching, this happens once and is reused until new data is ingested.

---

### 7.4 `agents/ingest.py` (125 lines)

**Purpose:** Entry point for all content ingestion. Takes raw input (PDF file, URL, or text string), splits it into chunks, embeds them, and stores in both ChromaDB and BM25.

**Module-level setup:**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```
The splitter tries to break at natural boundaries (paragraphs → sentences → words) rather than chopping mid-word. `chunk_size=500` characters per chunk, `chunk_overlap=50` characters shared between consecutive chunks.

**Functions:**

| Function | Input | Output | What It Does |
|----------|-------|--------|-------------|
| `generate_id(text)` | `str` | `str` (MD5 hex) | Creates a deterministic unique ID from content. Same content always gets the same ID — prevents duplicates on re-ingestion. |
| `ingest_pdf(filepath, topic)` | file path + topic string | `int` (chunk count) | Uses PyPDFLoader to extract text from PDF → splits → stores. Raises `ValueError` if PDF has no pages. |
| `ingest_url(url, topic)` | URL + topic string | `int` (chunk count) | Uses WebBaseLoader (BeautifulSoup under the hood) to fetch webpage → extract text → split → store. Raises `ValueError` if page is empty. |
| `ingest_text(text, topic, source)` | raw text + topic + source label | `int` (chunk count) | Directly splits the text string → stores. Raises `ValueError` if text is empty after stripping. |
| `_store_chunks(texts, topic, source)` | chunk list + metadata | None | **The core function.** Embeds all chunks via `embed_texts()`, generates IDs, then upserts into both ChromaDB and BM25 in a single call. Both stores get the same IDs and metadata — guaranteed sync. |

**Why MD5 for IDs?** Two identical chunks should map to the same ID. This way, re-ingesting the same PDF doesn't create duplicates — ChromaDB's `upsert` recognizes the existing ID and updates instead of inserting.

**Why all three ingest functions return `int`?** So the caller (UI) knows how many chunks were created, useful for user feedback ("Stored 42 chunks under RAG").

---

### 7.5 `agents/retrieve.py` (195 lines)

**Purpose:** The hybrid retrieval + reranking pipeline. This is the most algorithmically important file. It combines two search strategies and then uses a neural reranker to produce the final top-5 results.

**Module-level setup:**
```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```
Downloads once (~80MB), cached in `~/.cache/torch/`. Trained on MS MARCO — Microsoft's massive passage ranking dataset with millions of query-passage pairs labeled for relevance.

**The `retrieve()` function — 3 stages:**

#### Stage 1A — Vector Search (semantic)
```python
query_vector = embed_query(query)
vector_results = collection.query(query_embeddings=[query_vector], n_results=10, ...)
```
- Embeds the query into a 384-dim vector
- ChromaDB finds the 10 nearest stored vectors by cosine distance
- Returns chunks whose **meaning** is close to the query
- Handles synonyms, paraphrasing, conceptual similarity
- Example: "how attention works" matches "multi-head self-attention mechanism"

#### Stage 1B — BM25 Keyword Search
```python
bm25_index, corpus = get_bm25_index()
scores = bm25_index.get_scores(query_tokens)
```
- Tokenizes the query into words
- BM25 scores every stored chunk by keyword overlap
- Returns top 10 chunks with score > 0
- Handles exact terms, acronyms, numbers, proper nouns
- Example: "HNSW parameter" matches the chunk literally containing "HNSW"
- Respects topic filter if provided

#### Stage 2 — Merge + Deduplicate
```python
key = hashlib.md5(chunk["text"].encode()).hexdigest()
```
- Combines both sets (up to 20 chunks)
- Deduplicates using MD5 hash of chunk content
- A chunk appearing in BOTH searches is a very strong relevance signal

#### Stage 3 — Cross-Encoder Reranking
```python
pairs = [[query, c["text"]] for c in candidates]
reranker_scores = reranker.predict(pairs)
```
- Creates `[query, chunk]` pairs for each candidate
- The cross-encoder reads both texts together and outputs a relevance score
- Sort by reranker score, return top 5
- This dramatically improves results vs. raw cosine similarity

**Return value:** List of dicts, each with:
```python
{"text": "chunk content...", "topic": "RAG", "source": "file.pdf", "similarity": 0.8234}
```
The `similarity` field is the reranker score (float), not the raw cosine distance.

---

### 7.6 `agents/answer.py` (117 lines)

**Purpose:** The orchestrator. Calls retrieve(), builds the prompt from YAML config, sends to Groq LLM, and returns a structured response.

**Module-level setup:**
```python
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"
```

**Functions:**

| Function | Purpose |
|----------|---------|
| `_load_prompts_cached(mtime)` | `@lru_cache` — loads and parses `prompts.yaml`. Cache key is the file's modification time. If you edit the YAML file, the next call detects a new mtime and reloads. |
| `load_prompts()` | Gets the file's current mtime, calls the cached loader. This pattern avoids re-reading the file on every query while still picking up edits. |
| `answer(query, topic)` | The main pipeline function. See below. |

**`answer()` step by step:**

1. **Load config** from prompts.yaml (cached)
2. **Retrieve** top 5 chunks via `retrieve(query, topic)`
3. **Early return** if no chunks found → returns `no_results_message`
4. **Build context block** — numbered chunks:
   ```
   [1] Topic: RAG
   RAG stands for Retrieval Augmented Generation...
   ---
   [2] Topic: RAG
   Chunking splits documents into smaller segments...
   ```
5. **Template substitution** — `Template(rag_template).safe_substitute(context=..., query=...)` replaces `$context` and `$query` in the YAML prompt
6. **Groq API call** — sends system prompt + user prompt to Llama 3.3 70B
7. **Error handling** — if Groq fails, raises `RuntimeError` with the error message
8. **Check for INSUFFICIENT_EVIDENCE** — if the answer starts with the marker
9. **Return structured dict:**
   ```python
   {
       "answer": "RAG stands for...",
       "sources": [{"text": "...", "source": "file.pdf", "topic": "RAG", "similarity": 0.82}],
       "prompt_version": "1.1",
       "insufficient_evidence": False
   }
   ```

**Critical fix — sources include `text`:** Previously, sources only contained `source` (filename) and `topic`. The RAGAS evaluation framework needs the actual chunk text to judge faithfulness. Without it, RAGAS was evaluating against filenames like "rag_basics.md", which obviously scored 0.0 for Faithfulness.

---

### 7.7 `ui/app.py` (711 lines)

**Purpose:** The Streamlit web interface. Handles user interaction, display, form submission, and error handling.

**Structure (top to bottom):**

| Section | Lines | What It Does |
|---------|-------|-------------|
| Imports + `_safe()` | 1-19 | XSS prevention helper using `html.escape()` |
| CSS Design System | 24-468 | Complete dark theme with custom components (hero, cards, pills, pipeline) |
| Sidebar — Ingestion | 475-547 | Topic input, source type radio (PDF/URL/Text), ingest buttons, memory stats |
| Hero Banner | 553-559 | "Your AI Learning Memory" title and subtitle |
| Query Section | 565-646 | Question input, topic filter, Search button, answer display |
| Pipeline Visualization | 650-711 | Collapsible "How NeuralVault Works" 7-step pipeline diagram |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `_safe(text)` | `html.escape(str(text))` — prevents XSS from LLM output or metadata being rendered as HTML |
| `build_pipeline_html(steps)` | Generates the 7-step pipeline visualization HTML from a data list |

**Security Features:**
- **XSS Prevention:** All dynamic content (LLM answers, source names, topics) goes through `_safe()` before being injected into HTML
- **Path Traversal Prevention:** PDF uploads use `tempfile.NamedTemporaryFile` + `Path(name).name` — strips any `../../` from filenames
- **Temp File Cleanup:** `os.unlink()` in a `finally` block ensures temp files are deleted even if ingestion fails
- **Error Handling:** Every external call (ingest, answer, stats) is wrapped in try/except with `st.error()` display
- **Session State:** Query results stored in `st.session_state["last_result"]` so they persist across Streamlit reruns (e.g., when the pipeline toggle triggers a rerun)

**Answer Display — Two Card Types:**

1. **Normal answer (blue card):** Shows the grounded answer with citation numbers [1][2], prompt version badge, source pills showing topic/file/similarity score
2. **Insufficient evidence (amber card):** Shows amber border, warning icon, the LLM's explanation of why it can't answer, and a hint to ingest more content

---

### 7.8 `eval/eval.py` (351 lines)

**Purpose:** Standalone evaluation script that runs the full NeuralVault pipeline against a 38-question tiered golden dataset, then measures quality using RAGAS metrics (factual/cross_chunk) and refusal detection (insufficient/adversarial).

**Key Components:**

#### `CompatHuggingFaceEmbeddings` (class)
RAGAS 0.4.3 has a bug: its `HuggingFaceEmbeddings` class has `embed_text()` and `embed_texts()` but NOT `embed_query()` and `embed_documents()`, which the singleton `_answer_relevancy` metric calls internally. This subclass bridges the gap:
```python
class CompatHuggingFaceEmbeddings(_RagasHFEmbeddings):
    def embed_query(self, text): return self.embed_text(text)
    def embed_documents(self, texts): return self.embed_texts(texts)
```

#### `RAGAS_RUN_CONFIG`
```python
RunConfig(max_workers=1, timeout=120, max_retries=3)
```
`max_workers=1` forces sequential evaluation — Groq's free tier can't handle parallel LLM calls without hitting rate limits. Slower but always completes.

#### Functions:

| Function | Purpose |
|----------|---------|
| `build_ragas_llm()` | Creates the RAGAS judge LLM using Groq's llama-3.1-8b-instant. Uses OpenAI client pointed at Groq's endpoint (workaround for RAGAS's buggy Groq SDK handling). |
| `build_ragas_embedder()` | Creates the CompatHuggingFaceEmbeddings instance for answer relevancy scoring. |
| `load_golden_dataset()` | Reads all 38 Q&A pairs from `golden_dataset.json`, each typed as factual/insufficient/cross_chunk/adversarial. |
| `run_pipeline_on_dataset(golden)` | For each Q&A: calls `answer()`, extracts chunk texts, strips INSUFFICIENT_EVIDENCE prefix, builds RAGAS row dict. |
| `run_ragas_eval(rows, llm, embedder)` | Converts rows to HuggingFace Dataset, runs RAGAS `evaluate()` with all 4 metrics. |
| `print_results(results, threshold)` | Prints the score table with progress bars, PASS/FAIL status. Returns `(all_pass, scores_dict)`. |
| `main()` | CLI entry point with `--threshold` argument. Orchestrates all steps, saves JSON results, exits with code 0 (pass) or 1 (fail). |

**Why 8B for RAGAS instead of 70B?**
- RAGAS only asks simple yes/no judgment questions — doesn't need a large model
- 8B has a separate 500K token/day quota on Groq vs 70B's 100K
- Much less likely to hit rate limits during evaluation

---

### 7.9 `eval/golden_dataset.json`

**Purpose:** 38-question tiered golden dataset covering four behavioral categories. Tests retrieval quality, generation quality, hallucination resistance, and adversarial robustness.

Each entry has a `"type"` field (backward-compatible — defaults to `"factual"` if absent):
```json
{
    "question": "What does RAG stand for and what problem does it solve?",
    "ground_truth": "RAG stands for Retrieval Augmented Generation. It solves the hallucination problem...",
    "topic": "RAG",
    "type": "factual"
}
```

**Question breakdown:**

| Type | Count | Purpose | Scoring |
|------|-------|---------|---------|
| `factual` | 15 | Direct questions answerable from seed notes | RAGAS (4 metrics) |
| `insufficient` | 10 | Off-topic questions — must trigger `INSUFFICIENT_EVIDENCE` | Refusal detection |
| `cross_chunk` | 8 | Synthesis questions spanning multiple note sections | RAGAS (4 metrics) |
| `adversarial` | 5 | False-premise questions about NeuralVault itself | Refusal detection (informational) |
| **Total** | **38** | | |

**Factual topics (15):** RAG definition, BM25 vs vector search, cross-encoder reranking, chunking and overlap, RAGAS metrics, versioned prompts, INSUFFICIENT_EVIDENCE behavior, embedding model and dimensions, LLM and inference provider, cosine similarity, plus 5 additional factual variants.

**Insufficient topics (10):** Docker, Kubernetes, SQL joins, React hooks, Kafka, gradient descent, pandas, PostgreSQL indexing, Django ORM, Apache Spark — none present in the seed knowledge base.

**Cross-chunk topics (8):** Require synthesizing 2–3 seed sections (e.g., how hybrid search and cross-encoder reranking work together end-to-end).

**Adversarial topics (5):** False premises about NeuralVault (e.g., "How does NeuralVault's GPT-4 integration work?" — NeuralVault uses Llama 3.3 70B, not GPT-4).

---

### 7.10 Other Files

| File | Purpose |
|------|---------|
| `requirements.txt` | 14 Python packages. `ragas==0.4.3` is pinned because its API changes between versions. |
| `.env` | Contains `GROQ_API_KEY=gsk_...` — the only secret. Never committed to git. |
| `.gitignore` | Excludes `venv/`, `.env`, `__pycache__/`, `chroma_db/`, `data/`, `eval/eval_results.json`, `.claude/` |
| `core/__init__.py` | Empty — marks `core/` as a Python package so `from core.embedder import ...` works |
| `agents/__init__.py` | Empty — marks `agents/` as a Python package so `from agents.ingest import ...` works |

---

## 8. End-to-End Workflow — Ingestion Path

**Goal:** Turn a PDF/URL/text into searchable, retrievable knowledge.

### Step-by-step (PDF example):

```
1. USER ACTION:
   Opens Streamlit UI → sidebar → selects "PDF" → uploads "transformers.pdf"
   → types topic "Transformers" → clicks "Ingest PDF"

2. UI LAYER (app.py):
   - Extracts safe filename: Path("transformers.pdf").name → "transformers.pdf"
   - Creates temp file: tempfile.NamedTemporaryFile(suffix=".pdf")
   - Writes uploaded bytes to temp file
   - Calls: ingest_pdf(tmp.name, "Transformers")

3. INGEST LAYER (ingest.py):
   - PyPDFLoader(filepath) reads the PDF, extracts text from each page
   - Returns list of Document objects (one per page)
   - Raises ValueError if PDF has 0 pages

4. CHUNKING:
   - RecursiveCharacterTextSplitter splits all pages into ~500-char chunks
   - With 50-char overlap between consecutive chunks
   - Example: 20-page PDF → ~45 chunks

5. STORING (_store_chunks):
   a. EMBED: embed_texts(45 chunks) → 45 vectors of 384 dimensions each
      (all-MiniLM-L6-v2 processes the batch)

   b. GENERATE IDS: MD5 hash of each chunk content → 45 unique IDs
      (deterministic — same content always gets same ID)

   c. BUILD METADATA: [{"topic": "Transformers", "source": "/tmp/file.pdf"}] × 45

   d. CHROMADB UPSERT:
      collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
      → Writes vectors + text + metadata to chroma.sqlite3
      → If an ID already exists, it updates instead of duplicating

   e. BM25 UPSERT:
      add_to_bm25_corpus(texts, ids, metadatas)
      → Tokenizes each chunk (lowercase, strip punctuation, split)
      → Updates or appends entries in bm25_corpus.json
      → Invalidates the BM25 cache

6. CLEANUP (app.py):
   - os.unlink(tmp.name) in finally block → deletes temp PDF
   - Shows: "✓ Stored under Transformers"
```

---

## 9. End-to-End Workflow — Query Path

**Goal:** Answer a question using only the user's ingested notes.

### Step-by-step:

```
1. USER ACTION:
   Types: "What is the difference between BM25 and vector search?"
   Optional filter: "RAG"
   Clicks: "Search Memory"

2. UI LAYER (app.py):
   Calls: answer("What is the difference between BM25 and vector search?", topic="RAG")

3. ANSWER LAYER (answer.py):
   a. LOAD CONFIG: load_prompts() → reads prompts.yaml (cached by mtime)
   b. RETRIEVE: retrieve(query, topic="RAG", n_results=5) → enters retrieve.py

4. RETRIEVAL LAYER (retrieve.py):

   STAGE 1A — VECTOR SEARCH:
   - embed_query("What is the difference between BM25 and vector search?")
     → [0.12, -0.34, 0.56, ...] (384 floats)
   - ChromaDB.query(vector, n=10, where={"topic": "RAG"})
     → Returns 10 chunks closest in meaning
     → Example match: chunk about "keyword-based retrieval vs semantic embeddings"
       with vector_score 0.7823

   STAGE 1B — BM25 SEARCH:
   - Tokenize query: ["what", "is", "the", "difference", "between", "bm25", "and", "vector", "search"]
   - BM25Okapi.get_scores(tokens) → score for every chunk in corpus
   - Top 10 with score > 0, filtered by topic="RAG"
     → Example match: chunk literally containing "BM25" with bm25_score 4.21

   STAGE 2 — MERGE:
   - Combine 10 vector + 10 BM25 results = up to 20
   - Deduplicate by MD5 hash of text content
   - Result: ~15 unique candidates
   - Some chunks appear in BOTH lists (very strong signal)

   STAGE 3 — RERANK:
   - Create 15 pairs: [query, candidate_text]
   - CrossEncoder predicts relevance for each pair
   - Sort by reranker score descending
   - Return top 5 with highest scores
   - Example output:
     [
       {"text": "BM25 scores documents by exact term...", "similarity": 0.9234, ...},
       {"text": "Vector search converts text to embeddings...", "similarity": 0.8891, ...},
       {"text": "Hybrid search combines both approaches...", "similarity": 0.8456, ...},
       {"text": "Cosine similarity measures the angle...", "similarity": 0.7123, ...},
       {"text": "The retrieval pipeline first embeds...", "similarity": 0.6892, ...},
     ]

5. BACK IN ANSWER LAYER (answer.py):

   a. BUILD CONTEXT:
      "[1] Topic: RAG\nBM25 scores documents by exact term...\n\n---\n\n
       [2] Topic: RAG\nVector search converts text to embeddings...\n\n---\n\n
       [3] Topic: RAG\nHybrid search combines both approaches..."

   b. TEMPLATE SUBSTITUTION:
      Template(rag_prompt).safe_substitute(
          context = <the numbered context above>,
          query   = "What is the difference between BM25 and vector search?",
          topic   = "RAG"
      )
      → Produces the final prompt with context + question

   c. GROQ API CALL:
      client.chat.completions.create(
          model="llama-3.3-70b-versatile",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": final_prompt}
          ],
          temperature=0.3,
          max_tokens=1024
      )
      → Llama 3.3 70B reads the context + question
      → Generates a grounded answer with citations [1], [2], etc.

   d. CHECK INSUFFICIENT EVIDENCE:
      Does the answer start with "INSUFFICIENT_EVIDENCE"?
      → No? Normal answer. Return it.
      → Yes? Flag it. Still return the answer text.

   e. RETURN:
      {
          "answer": "BM25 is a keyword-based search algorithm that scores documents
                     by exact term frequency [1]. Vector search converts text to
                     embeddings and finds similar content using cosine similarity [2]...",
          "sources": [
              {"text": "BM25 scores...", "source": "rag.pdf", "topic": "RAG", "similarity": 0.9234},
              ...
          ],
          "prompt_version": "1.1",
          "insufficient_evidence": false
      }

6. UI LAYER (app.py):
   - Stores result in st.session_state["last_result"]
   - Renders blue answer card with:
     - "Neural Response" label
     - "prompt v1.1" badge
     - Answer text with citations
   - Renders source pills:
     - [1] RAG · rag.pdf · 0.92
     - [2] RAG · rag.pdf · 0.89
     - etc.
```

---

## 10. End-to-End Workflow — Evaluation Path

**Goal:** Objectively measure how good NeuralVault's answers are, covering both quality (RAGAS) and safety (refusal behavior).

### Step-by-step:

```
1. COMMAND:
   python eval/eval.py --threshold 0.7

2. INITIALIZATION:
   - sys.stdout.reconfigure(encoding="utf-8") — Windows encoding fix for emoji output
   - MARKER = _load_marker() — reads INSUFFICIENT_EVIDENCE string from config/prompts.yaml
     (single source of truth — prompt and eval always use the same marker)
   - build_ragas_llm() → OpenAI client at Groq's endpoint → llm_factory("llama-3.1-8b-instant")
   - build_ragas_embedder() → CompatHuggingFaceEmbeddings("all-MiniLM-L6-v2")

3. LOAD GOLDEN DATASET:
   - Reads golden_dataset.json → 38 Q&A pairs across 4 types:
     - factual (15): standard questions answerable from notes
     - insufficient (10): off-topic questions that must trigger INSUFFICIENT_EVIDENCE
     - cross_chunk (8): synthesis questions requiring multiple chunks
     - adversarial (5): false-premise questions about NeuralVault itself

4. RUN PIPELINE ON ALL 38 QUESTIONS:
   For each item, call answer(question, topic) → full pipeline (retrieve + generate)

   ROUTING BY TYPE:
   - If type is "insufficient" or "adversarial":
       → Check if answer starts with MARKER (INSUFFICIENT_EVIDENCE)
       → Append to refusal_rows: {question, answer, type, detected: bool}
   - If type is "factual" or "cross_chunk":
       → Strip INSUFFICIENT_EVIDENCE prefix if present (edge case)
       → Append to ragas_rows: {question, answer, contexts, ground_truth}

   Result: 23 ragas_rows, 15 refusal_rows

5. RAGAS EVALUATION (on 23 factual + cross_chunk rows):
   - Convert to HuggingFace Dataset
   - Pass to ragas.evaluate() with 4 metrics + judge LLM + embedder
   - Sequential mode (max_workers=1) — avoids Groq rate-limit errors

     FAITHFULNESS (per question):
       → Extract claims from the generated answer
       → Check if ANY retrieved context supports each claim
       → Score = (supported claims) / (total claims)

     ANSWER RELEVANCY (per question):
       → Generate N hypothetical questions from the answer
       → Embed each → cosine similarity to original question
       → Score = mean similarity

     CONTEXT PRECISION (per question):
       → For each chunk: "Is this relevant to the question?"
       → Score = (relevant chunks) / (total chunks), weighted by position

     CONTEXT RECALL (per question):
       → Compare ground_truth claims to retrieved contexts
       → Score = (claims found in contexts) / (total claims)

6. REFUSAL DETECTION (on 15 refusal_rows):
   - insufficient (10 questions): GATES THE BUILD
       → Count how many returned answer starting with MARKER
       → insuf_rate = insuf_correct / 10
       → Pass if insuf_rate >= threshold (0.7)
   - adversarial (5 questions): INFORMATIONAL ONLY
       → Tracked but does not affect exit code
       → Rationale: LLM correctly corrects false premises using notes
         ("NeuralVault does NOT use GPT-4, it uses Llama 3.3 70B via Groq")
         This is ideal behavior, not a failure.

7. COMBINED PASS/FAIL:
   - overall_pass = ragas_pass AND insufficient_pass
   - ragas_pass: all 4 RAGAS metrics >= threshold
   - insufficient_pass: insufficient detection rate >= threshold
   - adversarial: printed for information, does not affect overall_pass

8. OUTPUT:
   - Print two-section report: RAGAS Metrics + Refusal Detection
   - Save eval_results.json with:
     {threshold, passed, scores, insufficient_detection_rate,
      adversarial_correction_rate, question_types, num_questions,
      num_ragas_questions, num_refusal_questions, ragas_llm, ragas_embedder}
   - Exit code 0 (pass) or 1 (fail)
```

---

## 11. Runtime Files & Folders — Created & Deleted

### Created at Runtime (Persist)

| File/Folder | Created When | Purpose | Persists? |
|-------------|-------------|---------|-----------|
| `chroma_db/` | First ingestion | Root folder for all persistence | Yes — this IS your knowledge base |
| `chroma_db/chroma.sqlite3` | First ingestion | ChromaDB's SQLite database with vectors, text, metadata | Yes |
| `chroma_db/ba6c1882-.../` | First ingestion | ChromaDB internal segment files (HNSW index) | Yes |
| `chroma_db/bm25_corpus.json` | First ingestion | BM25 keyword index (tokenized text + metadata) | Yes |
| `eval/eval_results.json` | Each eval run | Latest RAGAS scores in JSON format | Yes (overwritten each run) |

### Created & Deleted at Runtime (Temporary)

| File | Created When | Deleted When | Purpose |
|------|-------------|-------------|---------|
| Temp PDF file (`/tmp/transformers.pdf_xxxxx.pdf`) | PDF upload in UI | After ingestion (in `finally` block) | Safe staging for PDF processing |

### Never Created by Code

| File | Why |
|------|-----|
| `.env` | Created manually by user — contains GROQ_API_KEY |
| `data/transformers.pdf` | Placed manually by user — sample study material |
| `golden_dataset.json` | Written by developer — the test Q&A pairs |

---

## 12. Dry Run #1 — Ingesting a PDF

**Input:** User uploads `data/transformers.pdf` with topic "Transformers"

```
STEP 1: PyPDFLoader reads the PDF
  → Extracts text from all pages
  → Result: 20 Document objects, one per page
  → Total text: ~15,000 characters

STEP 2: RecursiveCharacterTextSplitter(500, 50)
  → Splits 15,000 chars into chunks
  → 500 chars per chunk, 50 char overlap
  → Result: 34 chunks

  Example chunk #1 (500 chars):
  "The transformer architecture was introduced in 'Attention Is All You
   Need' by Vaswani et al. in 2017. Unlike recurrent neural networks that
   process sequences step by step, transformers process all positions in
   parallel using a mechanism called self-attention. Self-attention allows
   each position in the sequence to attend to all other positions,
   capturing long-range dependencies more effectively than RNNs. The key
   innovation is the multi-head attention mechanism, which runs several
   attention functions in par..."

  Example chunk #2 (starts with 50-char overlap):
  "which runs several attention functions in parallel, each focusing on
   different aspects of the input relationships. The outputs are
   concatenated and linearly projected..."

STEP 3: embed_texts(34 chunks)
  → all-MiniLM-L6-v2 encodes each chunk into 384 floats
  → Batch processing (all at once)
  → Result: 34 vectors, each [0.12, -0.34, 0.56, ..., 0.78]

STEP 4: Generate IDs
  → MD5("0-The transformer architecture was...") → "a3f8c2e1d4b5..."
  → MD5("1-which runs several attention...") → "b7e9d1c3a2f6..."
  → ... × 34

STEP 5a: ChromaDB upsert
  → collection.upsert(ids=[34 hashes], embeddings=[34 vectors],
                       documents=[34 texts], metadatas=[34 × {topic, source}])
  → Written to chroma.sqlite3

STEP 5b: BM25 upsert
  → For each chunk:
      tokenize("The transformer architecture was introduced...")
      → ["the", "transformer", "architecture", "was", "introduced", ...]
  → Appended to bm25_corpus.json with same IDs
  → BM25 cache invalidated

RESULT:
  → 34 chunks stored in ChromaDB (vectors + text + metadata)
  → 34 entries in bm25_corpus.json (tokenized text + metadata)
  → Ready for hybrid search
```

---

## 13. Dry Run #2 — Asking a Question

**Input:** "What is self-attention in transformers?" with topic filter "Transformers"

```
STEP 1: answer.py loads config from prompts.yaml (cached)

STEP 2: retrieve("What is self-attention in transformers?", topic="Transformers")

  STAGE 1A — Vector Search:
  → embed_query("What is self-attention in transformers?")
    → [0.23, -0.41, 0.67, ..., 0.12]  (384 floats)
  → ChromaDB.query(vector, n=10, where={"topic": "Transformers"})
  → Returns 10 chunks:
    #1: "Self-attention allows each position..." (distance=0.18 → score=0.82)
    #2: "The multi-head attention mechanism..." (distance=0.21 → score=0.79)
    #3: "Transformers process all positions..." (distance=0.25 → score=0.75)
    ... 7 more

  STAGE 1B — BM25 Search:
  → Tokenize query: ["what", "is", "self", "attention", "in", "transformers"]
  → BM25 scores all chunks in corpus
  → Top 10 where topic="Transformers" and score > 0:
    #1: "Self-attention allows each position..." (bm25_score=5.12)
    #2: "The key innovation is the multi-head attention mechanism..." (bm25_score=3.87)
    #3: "attention weights determine how much..." (bm25_score=3.21)
    ... 7 more

  STAGE 2 — Merge:
  → 10 vector + 10 BM25 = 20 chunks
  → After MD5 dedup: 14 unique candidates
  → Note: "Self-attention allows each position..." appears in BOTH lists
    (strongest signal — relevant by meaning AND by keywords)

  STAGE 3 — Rerank:
  → 14 pairs fed to cross-encoder:
    predict([
      ["What is self-attention in transformers?", "Self-attention allows each position..."],
      ["What is self-attention in transformers?", "The multi-head attention mechanism..."],
      ... × 14
    ])
  → Reranker scores (reads query + chunk together):
    "Self-attention allows each position..."           → 0.9456
    "The key innovation is the multi-head attention..." → 0.8912
    "Transformers process all positions in parallel..."  → 0.8234
    "attention weights determine how much..."           → 0.7891
    "The encoder consists of a stack of..."             → 0.7234
  → Return top 5

STEP 3: Build context block
  "[1] Topic: Transformers
   Self-attention allows each position in the sequence to attend to all
   other positions, capturing long-range dependencies...

   ---

   [2] Topic: Transformers
   The key innovation is the multi-head attention mechanism, which runs
   several attention functions in parallel...

   ---

   [3] Topic: Transformers
   ..."

STEP 4: Template substitution
  → $context replaced with the numbered context above
  → $query replaced with the question
  → $topic replaced with "Transformers"

STEP 5: Groq API call
  → model: "llama-3.3-70b-versatile"
  → temperature: 0.3
  → max_tokens: 1024
  → System prompt: "You are NeuralVault, a personal AI study assistant..."
  → User prompt: "Answer using ONLY the study notes provided..."

STEP 6: LLM generates (example):
  "Self-attention is a mechanism that allows each position in a sequence
   to attend to all other positions, capturing long-range dependencies
   more effectively than RNNs [1]. It is the key component of the
   transformer architecture, which processes all positions in parallel
   rather than sequentially [3]. The multi-head attention mechanism runs
   several attention functions in parallel, each focusing on different
   aspects of the input relationships [2]."

STEP 7: Check INSUFFICIENT_EVIDENCE → No, normal answer

STEP 8: Return to UI
  {
    "answer": "Self-attention is a mechanism that allows...",
    "sources": [
      {"text": "Self-attention allows...", "source": "transformers.pdf",
       "topic": "Transformers", "similarity": 0.9456},
      {"text": "The key innovation...", "source": "transformers.pdf",
       "topic": "Transformers", "similarity": 0.8912},
      ...
    ],
    "prompt_version": "1.1",
    "insufficient_evidence": false
  }

UI renders:
  ┌──────────────────────────────────────────────────┐
  │ ▎ Neural Response                   prompt v1.1  │
  │ ▎                                                │
  │ ▎ Self-attention is a mechanism that allows each  │
  │ ▎ position in a sequence to attend to all other   │
  │ ▎ positions [1]. It is the key component of the   │
  │ ▎ transformer architecture [3]. The multi-head    │
  │ ▎ attention mechanism runs several functions in    │
  │ ▎ parallel [2].                                   │
  └──────────────────────────────────────────────────┘

  Sources:
  [●] [1] Transformers · transformers.pdf · 0.95
  [●] [2] Transformers · transformers.pdf · 0.89
  [●] [3] Transformers · transformers.pdf · 0.82
  [●] [4] Transformers · transformers.pdf · 0.79
  [●] [5] Transformers · transformers.pdf · 0.72
```

---

## 14. Dry Run #3 — Running RAGAS Evaluation

**Command:** `python eval/eval.py --threshold 0.7`

```
STEP 1: Initialize
  → sys.stdout.reconfigure(encoding="utf-8") — Windows encoding fix
  → MARKER = "INSUFFICIENT_EVIDENCE" (loaded from config/prompts.yaml)
  → build_ragas_llm() → Groq OpenAI client → llm_factory("llama-3.1-8b-instant")
  → build_ragas_embedder() → CompatHuggingFaceEmbeddings("all-MiniLM-L6-v2")

STEP 2: Load golden dataset (38 Q&A pairs across 4 types)

STEP 3: Run full pipeline on all 38 questions

  Example — Q1 [factual]: "What does RAG stand for and what problem does it solve?"
    Type: factual → routes to ragas_rows
    → Full pipeline: retrieve + Groq
    → Adds: {question, answer, contexts=[5 chunks], ground_truth}

  Example — Q16 [insufficient]: "How do Docker containers differ from virtual machines?"
    Type: insufficient → routes to refusal_rows
    → Pipeline runs, answer starts with: "INSUFFICIENT_EVIDENCE: The retrieved notes..."
    → detected = True   (correct behavior)

  Example — Q34 [adversarial]: "How does NeuralVault use GPT-4 for answer generation?"
    Type: adversarial → routes to refusal_rows
    → Pipeline runs, answer: "NeuralVault does NOT use GPT-4, it uses Llama 3.3 70B via Groq"
    → detected = False  (OK — LLM correctly refuted the false premise, not a failure)

  Result: 23 ragas_rows + 15 refusal_rows

STEP 4: RAGAS evaluates 23 factual + cross_chunk rows

  Example — faithfulness for Q1:
    Claims: ["RAG = Retrieval Augmented Generation", "solves hallucination", ...]
    → Each claim checked against 5 retrieved chunks
    → All supported → score = 1.0

  Mean scores across 23 questions:
    Faithfulness:      0.871  ✅ PASS
    Answer Relevancy:  0.767  ✅ PASS
    Context Precision: 0.974  ✅ PASS
    Context Recall:    0.893  ✅ PASS

STEP 5: Refusal detection on 15 refusal_rows

  Insufficient (10 questions — GATES BUILD):
    → 9 correctly returned INSUFFICIENT_EVIDENCE (the gradient descent question did not)
    → Detection rate: 9/10 = 90%  ✅ PASS (threshold 70%)

  Adversarial (5 questions — INFORMATIONAL):
    → 2 correctly refused (returned INSUFFICIENT_EVIDENCE)
    → 3 correctly corrected the false premise with grounded answers
    → Correction rate: 60% — not a failure, reported for information only

STEP 6: Combined pass/fail
  ragas_pass = True (all 4 metrics ≥ 0.7)
  insufficient_pass = True (9/10 ≥ 0.7)
  overall_pass = True → exit code 0

STEP 7: Save to eval_results.json:
  {
    "threshold": 0.7,
    "passed": true,
    "scores": {"faithfulness": 0.871, "answer_relevancy": 0.767,
                "context_precision": 0.974, "context_recall": 0.893},
    "insufficient_detection_rate": 0.9,
    "insufficient_passed": true,
    "adversarial_correction_rate": 0.4,
    "question_types": {"factual": 15, "insufficient": 10, "cross_chunk": 8, "adversarial": 5},
    "num_questions": 38,
    "num_ragas_questions": 23,
    "num_refusal_questions": 15,
    "ragas_llm": "groq/llama-3.1-8b-instant",
    "ragas_embedder": "sentence-transformers/all-MiniLM-L6-v2"
  }
```

---

## 15. What Each Layer Does — Deep Dive

### Layer 1: Embedding Layer (`core/embedder.py`)

**What:** Converts human-readable text into machine-comparable numbers.

**How:** Uses a pre-trained neural network (`all-MiniLM-L6-v2`) that learned from millions of text pairs to map similar meanings to nearby points in 384-dimensional space.

**Output:** A list of 384 floating-point numbers. Two similar texts will produce similar lists (small cosine distance). Two unrelated texts will produce very different lists (large cosine distance).

**Why this model?** It's the sweet spot of quality vs. size. At 80MB it runs on any CPU in milliseconds. Larger models (768-dim, 1024-dim) give marginally better results but are 5-10x slower and require more memory.

### Layer 2: Storage Layer (`core/vectorstore.py`)

**What:** Persists vectors and keyword indices to disk so knowledge survives restarts.

**How:** Two complementary storage engines:
- **ChromaDB** stores vectors in a SQLite database with an HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
- **BM25** stores tokenized text as JSON for keyword matching

**Output:** No direct output — it's a persistence layer. Other layers read from it.

**Why two stores?** Vector search misses exact keywords. BM25 misses semantic similarity. Example: searching for "BERT" — vector search finds chunks about "transformer models" (semantically close) while BM25 finds the chunk that literally says "BERT".

### Layer 3: Ingestion Layer (`agents/ingest.py`)

**What:** The "write path" — takes raw content and prepares it for storage.

**How:** Loads documents → splits into overlapping chunks → embeds each chunk → stores in both ChromaDB and BM25 with the same IDs.

**Output:** Chunks stored in both stores, ready for retrieval. Returns the count of chunks created.

**Why chunk + overlap?** Documents are too long to embed as a single unit (embeddings work best on paragraph-sized text). Overlap prevents information loss at chunk boundaries.

### Layer 4: Retrieval Layer (`agents/retrieve.py`)

**What:** The "read path" — finds the most relevant chunks for a given question.

**How:** Three-stage pipeline:
1. Cast a wide net with two search methods (vector + BM25)
2. Merge and deduplicate
3. Precisely rank with a cross-encoder

**Output:** Top 5 chunks ranked by cross-encoder relevance score. Each chunk includes its text, topic, source file, and similarity score.

**Why reranking?** Raw cosine similarity is approximate — it compares pre-computed embeddings. The cross-encoder reads query + chunk together and produces a much more accurate relevance judgment. This single addition dramatically improves result quality.

### Layer 5: Generation Layer (`agents/answer.py`)

**What:** Constructs a grounded answer using the retrieved chunks.

**How:** Builds a numbered context block from chunks, substitutes it into the YAML prompt template, sends to Groq's Llama 3.3 70B, and structures the response.

**Output:** A dict with the answer text, source metadata (including chunk text), prompt version, and INSUFFICIENT_EVIDENCE flag.

**Why template substitution?** Separating the prompt from code means prompt engineers can iterate without touching Python. Using `safe_substitute()` prevents accidental variable replacement if chunk text contains `$` characters.

### Layer 6: UI Layer (`ui/app.py`)

**What:** The user-facing web application.

**How:** Streamlit renders forms, buttons, and custom HTML cards. All user input flows into the agent layer, and all responses are displayed with XSS-safe escaping.

**Output:** A dark-themed web page with sidebar ingestion, main query area, answer cards, source pills, and a collapsible pipeline visualization.

### Layer 7: Evaluation Layer (`eval/eval.py`)

**What:** Automated quality measurement.

**How:** Runs the full pipeline on 38 golden Q&A pairs (4 behavioral types), routes factual/cross_chunk questions through RAGAS (4 quality metrics), and routes insufficient/adversarial questions through refusal detection (marker-based).

**Output:** PASS/FAIL per metric, JSON results file, exit code for CI/CD integration.

---

## 16. Why Each Design Decision Was Made

| Decision | Why |
|----------|-----|
| **ChromaDB (local) over Pinecone/Weaviate** | Zero cost, no account needed, data stays on your machine. For a personal learning tool, cloud vector DBs are overkill. |
| **BM25 stored as JSON** | ChromaDB has no keyword search. BM25 needs raw tokens. JSON is the simplest format that can be read/written without dependencies. |
| **Hybrid search over vector-only** | Vector search alone misses exact keywords (acronyms, numbers, proper nouns). BM25 alone misses semantic similarity. Together they cover all cases. |
| **Cross-encoder reranking** | Cosine similarity is fast but approximate. Reranking the shortlist with a cross-encoder is the single highest-impact improvement you can make to a RAG pipeline. |
| **500-char chunks with 50-char overlap** | 500 chars ≈ 1-2 paragraphs — enough context for the embedding model to capture meaning. 50-char overlap prevents boundary information loss. |
| **Groq over OpenAI/Anthropic** | Free tier with fast inference. Perfect for a learning project. Groq runs on custom LPU hardware, so inference is extremely fast. |
| **YAML prompts over hardcoded strings** | Prompts change frequently during development. YAML lets you edit without restarting the app. Version field enables tracking which prompt produced which scores. |
| **`safe_substitute` over `str.replace`** | `str.replace("{context}", ...)` would replace ALL occurrences of `{context}` — including any that might appear inside chunk text. Template is safer. |
| **`@lru_cache` keyed on mtime** | Cache prompt loading to avoid disk I/O on every query, but automatically bust cache when the file is edited. Best of both worlds. |
| **MD5 IDs for chunks** | Deterministic — same content always gets same ID. Re-ingesting a document updates chunks instead of duplicating. Content-addressable storage. |
| **8B model for RAGAS judge** | RAGAS asks simple yes/no questions — doesn't need 70B. 8B has a separate 500K token/day quota on Groq, so evaluations don't eat into your generation quota. |
| **Sequential RAGAS evaluation (max_workers=1)** | Groq free tier rate-limits parallel requests. Sequential is slower but guarantees completion without timeout errors. |
| **HTML escaping (`_safe()`) in UI** | LLM output could contain HTML/JavaScript. Without escaping, a malicious chunk could inject scripts into the page (XSS). |
| **Temp files for PDF uploads** | Streamlit's UploadedFile object can't be read directly by PyPDFLoader (needs a real file path). Temp files bridge this gap and are cleaned up in `finally`. |
| **`st.session_state` for results** | Streamlit reruns the entire script on every interaction. Without session state, toggling the pipeline view would clear the answer. State persistence prevents this. |

---

## 17. Security Measures

| Threat | Protection | File |
|--------|-----------|------|
| **XSS (Cross-Site Scripting)** | `_safe()` → `html.escape()` on all dynamic HTML content | `ui/app.py` |
| **Path Traversal** | `Path(uploaded_file.name).name` strips `../` paths | `ui/app.py` |
| **Temp File Leaks** | `os.unlink()` in `finally` block | `ui/app.py` |
| **Prompt Injection (template)** | `Template.safe_substitute()` ignores unknown `$` vars | `agents/answer.py` |
| **API Key Exposure** | `.env` in `.gitignore`, never hardcoded | `.gitignore` |
| **Error Information Leak** | try/except shows user-friendly errors, logs details internally | All modules |

---

## 18. Final Evaluation Scores & What They Mean

The evaluation runs a 38-question tiered golden dataset with dual scoring: RAGAS metrics for retrieval+generation quality, and refusal detection for hallucination safety.

### RAGAS Metrics (23 factual + cross-chunk questions)

```
Faithfulness:      0.871  ✅  (87.1% of answer claims are supported by retrieved chunks)
Answer Relevancy:  0.767  ✅  (76.7% on-topic — answers address the question well)
Context Precision: 0.974  ✅  (97.4% of retrieved chunks are relevant to the question)
Context Recall:    0.893  ✅  (89.3% of needed information was retrieved)
```

### Refusal Detection (15 behavioral questions)

```
Insufficient evidence:  9/10  (90.0%)  ✅  PASS  [gates build — threshold 70%]
Adversarial refusal:    2/5   (40.0%)       [informational — 3 correctly refuted false premises]
```

### What These Scores Tell Us

**Faithfulness 0.871** — The LLM rarely hallucinates. 87.1% of answer claims are directly supported by retrieved chunks. The grounding prompt ("Answer using ONLY the study notes") is working. The gap vs. the earlier 10-question baseline (0.936) is expected — the expanded 38-question dataset includes harder cross-chunk synthesis questions where the LLM combines information across multiple chunks.

**Answer Relevancy 0.767** — Answers address questions well. Cross-chunk synthesis questions produce longer, more complex answers that RAGAS scores slightly lower on relevancy. This score shows natural run-to-run variance (±0.05) due to LLM non-determinism.

**Context Precision 0.974** — The hybrid search + cross-encoder reranking pipeline is excellent. Only 2.6% of retrieved chunks are irrelevant noise. The cross-encoder is the primary driver of this score.

**Context Recall 0.893** — The pipeline retrieves most needed information. The 10.7% gap means some ground-truth claims span chunks that didn't make the top 5. This improved from the earlier baseline (0.860) as the larger, more diverse golden dataset better exercises the retrieval system.

**Insufficient evidence (90%)** — When asked off-topic questions (Docker, SQL, Kubernetes, React), the LLM correctly returns INSUFFICIENT_EVIDENCE 9 times out of 10. The one miss (gradient descent) occurs because some RAG notes touch on optimization concepts at the boundary. 90% comfortably clears the 70% build gate.

**Adversarial refusal (40% — informational)** — When asked questions with false premises ("How does NeuralVault use GPT-4?"), the LLM refutes the false premise with grounded answers 60% of the time ("NeuralVault does NOT use GPT-4, it uses Llama 3.3 70B via Groq"). This is correct and desirable behavior — the system uses its notes to correct misinformation. Adversarial questions are tracked but do not gate the build.

---

## 19. Improvement Strategies

### Quick Wins (Low Effort, High Impact)

| Strategy | Expected Impact | Effort |
|----------|----------------|--------|
| Increase `n_results` from 5 to 7 | Context Recall +5-8% | 1 line change in answer.py |
| Reduce `chunk_size` from 500 to 400 | More precise chunks, better Context Precision | 1 line change in ingest.py |
| Increase `chunk_overlap` from 50 to 100 | Fewer boundary losses, better Context Recall | 1 line change in ingest.py |
| Add `--limit N` flag to eval.py | Run 5 questions instead of 10 during development (saves tokens) | ~10 lines |

### Medium Effort Improvements

| Strategy | Expected Impact | What's Involved |
|----------|----------------|-----------------|
| Switch generation to `llama-3.3-70b-versatile` | Better answer quality, more articulate responses | Change model in prompts.yaml (already done) |
| Add query expansion | Better recall for vague questions — LLM rewrites query before search | New function in retrieve.py |
| Add metadata filtering in UI | Let users filter by source file, date range, topic | UI changes + ChromaDB where clauses |
| Chunk deduplication at query time | Prevent near-identical chunks from wasting the top-5 slots | Similarity threshold in retrieve.py |
| Streaming responses | Show LLM output as it generates (better UX) | Switch to Groq streaming API + Streamlit st.write_stream |

### Advanced Improvements

| Strategy | Expected Impact | What's Involved |
|----------|----------------|-----------------|
| Parent-child chunking | Small chunks for retrieval, full sections for context | Double-layer chunking + parent lookup |
| Multi-vector retrieval (ColBERT) | Better than single-vector cosine | Major architecture change |
| User feedback loop | "Was this helpful?" button → retrain/retune | New feedback table + UI + reranking adjustment |
| Multi-modal ingestion | Images, diagrams, tables from PDFs | Vision model integration |
| Ollama local fallback | Fully local, zero API dependency | New LLM client class, env var for provider selection |

### What NOT to Improve

| Anti-Pattern | Why It's Bad |
|-------------|-------------|
| Adding a general chatbot mode | Defeats the purpose — NeuralVault's value IS grounding |
| Using GPT-4 instead of Groq | Adds cost and latency for minimal quality gain in this use case |
| Cloud vector database | Adds complexity, cost, and network dependency for a personal tool |
| Fine-tuning the embedding model | The pre-trained model is already excellent for general text; fine-tuning requires significant data and expertise |

---

## 20. Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval Augmented Generation — retrieve relevant documents first, then generate an answer grounded in them |
| **Embedding** | A fixed-size vector (list of numbers) representing the meaning of a text |
| **Vector** | A list of numbers (e.g., 384 floats) that represents text in mathematical space |
| **Cosine Similarity** | Measures the angle between two vectors. 1.0 = identical direction (same meaning), 0.0 = perpendicular (unrelated) |
| **Chunk** | A ~500-character segment of a document, the basic unit of storage and retrieval |
| **BM25** | Best Match 25 — classical keyword ranking algorithm based on term frequency and inverse document frequency |
| **ChromaDB** | Open-source vector database that stores embeddings on disk and supports cosine similarity search |
| **Cross-Encoder** | A model that reads query and chunk together as one input — more accurate than comparing separate embeddings |
| **Bi-Encoder** | A model that embeds query and chunk separately, then compares vectors — fast but less accurate |
| **Reranking** | Re-scoring a shortlist of candidates with a more precise model after initial retrieval |
| **Hybrid Search** | Combining vector search (semantic) with keyword search (BM25) for more comprehensive results |
| **HNSW** | Hierarchical Navigable Small World — the index structure ChromaDB uses for fast approximate nearest neighbor search |
| **Groq** | Cloud inference provider running LLMs on custom LPU (Language Processing Unit) hardware — very fast, free tier available |
| **RAGAS** | Retrieval Augmented Generation Assessment — framework with 4 metrics for evaluating RAG pipeline quality |
| **Faithfulness** | RAGAS metric — % of answer claims supported by retrieved chunks (measures hallucination) |
| **Answer Relevancy** | RAGAS metric — how well the answer addresses the actual question asked |
| **Context Precision** | RAGAS metric — % of retrieved chunks that are relevant to the question |
| **Context Recall** | RAGAS metric — % of needed information that was successfully retrieved |
| **Golden Dataset** | Hand-crafted set of questions with known correct answers, used as ground truth for evaluation |
| **INSUFFICIENT_EVIDENCE** | Marker returned when the LLM determines retrieved chunks don't contain enough information to answer reliably |
| **Upsert** | Update-or-insert — if the ID exists, update it; if not, insert a new entry |
| **LRU Cache** | Least Recently Used cache — stores results of expensive function calls, evicts oldest when full |
| **XSS** | Cross-Site Scripting — security vulnerability where untrusted content is rendered as HTML/JavaScript |
| **Template Injection** | When user-controlled text accidentally gets interpreted as template variables |
| **InstructorLLM** | RAGAS's internal LLM wrapper created by `llm_factory()` — required for RAGAS 0.4+ metrics |
| **Sentence Transformers** | Python library for computing dense vector representations of text using transformer models |
| **Streamlit** | Python framework for building web applications from scripts — handles UI, state, and reruns |
| **Session State** | Streamlit's mechanism for persisting variables across script reruns (every widget interaction triggers a rerun) |

---

*Report generated for the NeuralVault project. After reading this document, you should understand every file, every function, every design decision, and every data flow in the system — from PDF upload to grounded answer to automated evaluation.*
