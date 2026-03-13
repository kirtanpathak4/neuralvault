import hashlib
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.vectorstore import get_collection, add_to_bm25_corpus
from core.embedder import embed_texts

log = logging.getLogger(__name__)

# -----------------------------------------------------------
# Text Splitter
# chunk_size=500   → each chunk is ~500 characters
# chunk_overlap=50 → last 50 chars of chunk N appear at start of N+1
# -----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def generate_id(text: str) -> str:
    """
    MD5 hash of content = unique ID per chunk.
    Prevents duplicates if you ingest the same document twice.
    """
    return hashlib.md5(text.encode()).hexdigest()


def ingest_pdf(filepath: str, topic: str) -> int:
    """
    Reads a PDF, splits into chunks, stores in ChromaDB + BM25.
    topic gets stored as metadata for topic-filtered queries later.

    Raises:
        FileNotFoundError: if the PDF file doesn't exist
        ValueError: if the PDF is empty or can't be parsed
    """
    log.info("Loading PDF: %s", filepath)
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    if not docs:
        raise ValueError(f"PDF produced no pages: {filepath}")

    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    log.info("Split into %d chunks", len(texts))
    _store_chunks(texts, topic, source=filepath)
    return len(texts)


def ingest_url(url: str, topic: str) -> int:
    """
    Fetches a webpage, splits, stores in ChromaDB + BM25.
    Great for AI papers, blog posts, documentation pages.

    Raises:
        ConnectionError: if the URL can't be fetched
        ValueError: if the page has no extractable content
    """
    log.info("Loading URL: %s", url)
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs:
        raise ValueError(f"URL produced no content: {url}")

    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    log.info("Split into %d chunks", len(texts))
    _store_chunks(texts, topic, source=url)
    return len(texts)


def ingest_text(text: str, topic: str, source: str = "manual") -> int:
    """
    Ingest raw text directly — notes, summaries, anything.

    Raises:
        ValueError: if the text is empty after stripping
    """
    if not text.strip():
        raise ValueError("Cannot ingest empty text")

    log.info("Ingesting raw text for topic: %s", topic)
    chunks = splitter.split_text(text)
    log.info("Split into %d chunks", len(chunks))
    _store_chunks(chunks, topic, source=source)
    return len(chunks)


def _store_chunks(texts: list, topic: str, source: str):
    """
    Stores chunks in BOTH ChromaDB (vectors) and BM25 (keywords).

    Before Phase 1: only ChromaDB
    After Phase 1:  ChromaDB + BM25 updated together in one call
    Both use the same IDs and metadatas so they stay in sync.
    """
    collection = get_collection()

    embeddings = embed_texts(texts)
    ids = [generate_id(f"{i}-{t}") for i, t in enumerate(texts)]
    metadatas = [{"topic": topic, "source": source} for _ in texts]

    # ── Store in ChromaDB (vectors + text + metadata) ──────────
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    # ── Store in BM25 (tokenized text for keyword search) ──────
    # Same ids and metadatas as ChromaDB — the two stores are
    # always in sync because they're updated in the same function.
    add_to_bm25_corpus(texts, ids, metadatas)

    log.info("Stored %d chunks under topic: '%s'", len(texts), topic)
