import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.vectorstore import get_collection, get_bm25_index
from core.embedder import embed_query
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
#
# BEFORE Phase 1 (bi-encoder only):
#   embed(query) → vector
#   embed(chunk) → vector
#   cosine(query_vec, chunk_vec) → score
#   Problem: each is embedded independently — approximate
#
# AFTER Phase 1 (cross-encoder reranker):
#   encode(query + chunk together as one input) → relevance score
#   The model READS both at the same time — like a human would
#   Much slower per chunk but far more precise
#
# We use the cross-encoder ONLY on the shortlist (≤20 chunks)
# after hybrid search — not on all thousands of stored chunks.
# This gives speed from bi-encoder + precision from cross-encoder.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
# Trained on MS MARCO (Microsoft's massive passage ranking dataset)
# ~80MB, downloads once, runs on CPU, cached automatically.
# ─────────────────────────────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve(query: str, topic: str = None, n_results: int = 5) -> list[dict]:
    """
    Hybrid retrieval pipeline — 3 stages:

    Stage 1 — SEARCH (wide net, fast):
        Vector search:  embed query → cosine search → top 10 semantic matches
        BM25 search:    tokenize query → keyword score → top 10 keyword matches

    Stage 2 — MERGE:
        Combine both sets, deduplicate → up to 20 unique candidates

    Stage 3 — RERANK (narrow, precise):
        Cross-encoder scores each candidate against the query together
        Sort by reranker score → return top n_results

    Args:
        query:     user's question
        topic:     optional topic filter — only search within that topic
        n_results: how many chunks to return after reranking (default 5)

    Returns:
        list of dicts with text, topic, source, similarity (reranker score)
    """
    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    where_filter = {"topic": topic} if topic else None

    # ──────────────────────────────────────────────────────────
    # STAGE 1A — Vector search (semantic)
    # Finds chunks whose MEANING is close to the query.
    # Handles synonyms, paraphrasing, conceptual similarity.
    # Example: "how attention works" finds "multi-head attention
    #          mechanism" even without matching words.
    # ──────────────────────────────────────────────────────────
    query_vector = embed_query(query)
    vector_k = min(10, total_chunks)

    vector_results = collection.query(
        query_embeddings=[query_vector],
        n_results=vector_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    vector_chunks = []
    if vector_results["documents"] and vector_results["documents"][0]:
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        ):
            vector_chunks.append({
                "text":         doc,
                "topic":        meta.get("topic", ""),
                "source":       meta.get("source", ""),
                "vector_score": round(1 - dist, 4)
            })

    # ──────────────────────────────────────────────────────────
    # STAGE 1B — BM25 keyword search
    # Finds chunks that contain the EXACT words from the query.
    # Handles specific terms, acronyms, numbers, proper nouns.
    # Example: "HNSW parameter" → finds the chunk literally
    #          containing "HNSW" regardless of its meaning.
    # ──────────────────────────────────────────────────────────
    bm25_chunks = []
    bm25_index, corpus = get_bm25_index()

    if bm25_index is not None:
        query_tokens = query.lower().split()
        scores = bm25_index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        count = 0
        for idx in ranked_indices:
            if count >= 10:
                break
            entry = corpus[idx]
            if topic and entry["metadata"].get("topic") != topic:
                continue
            if scores[idx] > 0:
                bm25_chunks.append({
                    "text":       entry["text"],
                    "topic":      entry["metadata"].get("topic", ""),
                    "source":     entry["metadata"].get("source", ""),
                    "bm25_score": round(float(scores[idx]), 4)
                })
                count += 1

    # ──────────────────────────────────────────────────────────
    # STAGE 2 — Merge + Deduplicate
    # Combine both sets. Use content hash as dedup key (#11).
    # First-80-chars matching was fragile — hash is exact.
    # A chunk appearing in BOTH searches is a very strong signal —
    # it's semantically relevant AND keyword-relevant.
    # ──────────────────────────────────────────────────────────
    seen = set()
    candidates = []

    for chunk in vector_chunks + bm25_chunks:
        key = hashlib.md5(chunk["text"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            candidates.append(chunk)

    if not candidates:
        return []

    log.info("Hybrid: %d vector + %d BM25 -> %d unique candidates",
             len(vector_chunks), len(bm25_chunks), len(candidates))

    # ──────────────────────────────────────────────────────────
    # STAGE 3 — Cross-encoder reranking
    # Feed each [query, chunk] pair to the cross-encoder together.
    # It outputs a relevance score for each pair — not a vector
    # comparison but an actual joint reading of both texts.
    #
    # Why this dramatically improves results:
    # Vector score:  how close are the embeddings? (approximate)
    # Reranker score: how relevant is THIS chunk to THIS question?
    #                 (precise — reads them together like a human)
    # ──────────────────────────────────────────────────────────
    pairs = [[query, c["text"]] for c in candidates]
    reranker_scores = reranker.predict(pairs)

    for i, chunk in enumerate(candidates):
        chunk["reranker_score"] = float(reranker_scores[i])

    ranked = sorted(candidates,
                    key=lambda x: x["reranker_score"],
                    reverse=True)
    top = ranked[:n_results]

    log.info("Reranker selected top %d from %d candidates", len(top), len(candidates))

    # ──────────────────────────────────────────────────────────
    # Return similarity as float (#9) — not a formatted string.
    # Callers that need display formatting can format it themselves.
    # ──────────────────────────────────────────────────────────
    return [
        {
            "text":       c["text"],
            "topic":      c["topic"],
            "source":     c["source"],
            "similarity": round(c["reranker_score"], 4),
        }
        for c in top
    ]
