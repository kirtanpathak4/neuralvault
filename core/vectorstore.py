import re
import json
import logging
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Absolute path — always resolves to project_root/chroma_db
# regardless of where Python is called from.
# Path(__file__) = this file (core/vectorstore.py)
# .parent        = core/
# .parent        = project root
# / "chroma_db"  = project_root/chroma_db/
# ─────────────────────────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent.parent / "chroma_db")

# PersistentClient saves your data to disk — survives restarts
client = chromadb.PersistentClient(path=DB_PATH)


def get_collection(name: str = "learning_memory"):
    """
    A collection = a table in SQL terms.
    get_or_create = safe to call anytime, won't duplicate.
    cosine = similarity metric best suited for text/meaning comparison.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────────────────────────
# BM25 — keyword search index
#
# BM25 (Best Match 25) is a classical keyword ranking algorithm.
# It scores chunks by how many query words appear in them,
# weighted by how rare those words are across ALL chunks.
#
# Why store separately from ChromaDB?
# ChromaDB handles vectors only — it has no keyword search.
# BM25 needs raw tokenized text. So we maintain both in sync.
#
# How it persists: saved as JSON next to chroma_db/ on disk,
# loaded fresh each query. Same persistence guarantee as ChromaDB.
#
# Example of what BM25 catches that vector search misses:
#   Query: "HNSW index parameter"
#   Vector search: returns chunks about "vector database config"
#                  (semantically close but may miss exact term)
#   BM25:          returns chunk literally containing "HNSW"
#                  because that exact word is in it
# ─────────────────────────────────────────────────────────────────
BM25_PATH = Path(__file__).parent.parent / "chroma_db" / "bm25_corpus.json"

# Module-level cache so BM25 isn't rebuilt from disk on every query (#23)
_bm25_cache: dict = {"corpus": None, "index": None}


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25: lowercase, strip punctuation, split on whitespace.
    Removes noise characters that hurt keyword matching accuracy (#12).
    """
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return cleaned.split()


def _load_bm25_corpus() -> list:
    """Load saved BM25 corpus from disk. Returns [] if not found."""
    if BM25_PATH.exists():
        with open(BM25_PATH, "r") as f:
            return json.load(f)
    return []


def _save_bm25_corpus(corpus: list):
    """Persist BM25 corpus to disk so it survives restarts."""
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "w") as f:
        json.dump(corpus, f)


def _invalidate_bm25_cache():
    """Clear cached BM25 index so next query rebuilds from disk."""
    _bm25_cache["corpus"] = None
    _bm25_cache["index"] = None


def add_to_bm25_corpus(texts: list, ids: list, metadatas: list):
    """
    Add or update chunks in the BM25 corpus.
    Called by ingest.py every time _store_chunks() runs.

    ChromaDB uses upsert (update-or-insert), so BM25 must do the same
    to stay in sync (#10). If an ID already exists, its text and
    metadata are updated rather than skipped.

    texts     = list of raw chunk strings
    ids       = same IDs used in ChromaDB (ensures dedup consistency)
    metadatas = same metadata dicts used in ChromaDB
    """
    corpus = _load_bm25_corpus()
    existing = {entry["id"]: i for i, entry in enumerate(corpus)}

    added, updated = 0, 0
    for text, chunk_id, meta in zip(texts, ids, metadatas):
        entry = {
            "id": chunk_id,
            "text": text,
            "tokens": _tokenize(text),
            "metadata": meta,
        }
        if chunk_id in existing:
            corpus[existing[chunk_id]] = entry
            updated += 1
        else:
            corpus.append(entry)
            added += 1

    _save_bm25_corpus(corpus)
    _invalidate_bm25_cache()
    log.info("BM25 corpus updated: +%d new, %d updated (%d total)",
             added, updated, len(corpus))


def get_bm25_index():
    """
    Build a live BM25Okapi index from the saved corpus.
    Called at query time by retrieve.py.

    Uses a module-level cache (#23) so repeated queries don't
    reload from disk and re-tokenize every time. Cache is
    invalidated whenever add_to_bm25_corpus() writes new data.

    Returns (BM25Okapi, corpus_list) so the caller can map
    result indices back to original chunk text and metadata.
    Returns (None, []) if corpus is empty.
    """
    if _bm25_cache["index"] is not None:
        return _bm25_cache["index"], _bm25_cache["corpus"]

    corpus = _load_bm25_corpus()
    if not corpus:
        return None, []

    tokenized = [entry["tokens"] for entry in corpus]
    index = BM25Okapi(tokenized)

    _bm25_cache["corpus"] = corpus
    _bm25_cache["index"] = index

    return index, corpus
