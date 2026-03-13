"""
NeuralVault — Seed Knowledge for CI Evaluation
================================================
Ingests the reference study notes that the golden dataset questions
are based on.  Run this BEFORE eval.py so ChromaDB + BM25 have
chunks to retrieve from.

Why this exists:
  chroma_db/ and data/ are gitignored — they contain runtime data
  and potentially large PDFs.  In CI (GitHub Actions), the repo
  starts fresh with no vector store.  This script injects the
  minimum knowledge needed for the golden-dataset questions to be
  answerable.

Usage:
  python eval/seed_knowledge.py          # run once before eval
  python eval/eval.py --threshold 0.7    # then evaluate

The text below covers every concept tested by golden_dataset.json.
Written as study notes — the same format a real user would paste
into NeuralVault via the UI.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.ingest import ingest_text

# ── Study notes covering all concepts tested by the 38-question golden dataset ──

RAG_NOTES = """
RAG — Retrieval Augmented Generation
=====================================

RAG stands for Retrieval Augmented Generation. It solves the hallucination
problem by grounding LLM answers in retrieved documents rather than relying
on the model's training data alone.  The core idea: before the LLM generates
an answer, a retrieval step finds the most relevant documents from a
knowledge base and passes them as context.  The LLM then answers using ONLY
that context, dramatically reducing the chance of fabricating facts.

Hybrid Search — BM25 vs Vector Search
--------------------------------------

BM25 (Best Match 25) is a keyword-based search algorithm that scores
documents by exact term frequency and inverse document frequency (how rare
a word is across all documents). It excels at matching exact terms, acronyms,
numbers, and proper nouns.

Vector search converts text into embeddings (dense numerical vectors) using
a model like all-MiniLM-L6-v2 and finds semantically similar content using
cosine similarity. It handles synonyms, paraphrasing, and conceptual
similarity — things BM25 misses entirely.

Combining both is called hybrid search. BM25 catches exact keywords while
vector search catches meaning. Together they provide much better retrieval
than either alone.

Cross-Encoder Reranking
------------------------

A cross-encoder reranker reads the query and candidate chunk together as a
single input and outputs a direct relevance score.  This is fundamentally
different from a bi-encoder, which embeds the query and chunk separately
into independent vectors and then compares them with cosine similarity.

Cross-encoders are more accurate because they see both inputs simultaneously
— like a human would judge relevance — rather than comparing isolated
embeddings.  The tradeoff is speed: cross-encoders can't pre-compute
embeddings, so they're only used on a shortlist of ~20 candidates after
the initial fast retrieval.

Chunking and Overlap
---------------------

Chunking in a RAG pipeline splits documents into smaller segments (typically
300-500 characters) for storage and retrieval. This is necessary because
embedding models work best on paragraph-sized text, and retrieval is more
precise with smaller units.

Overlap between chunks is critical. When chunks are split at fixed
boundaries, sentences at the edge get cut in half. Overlap ensures that
the last N characters of chunk K are repeated at the start of chunk K+1,
so no context is lost at the boundary. NeuralVault uses chunk_size=500
with chunk_overlap=50.

RAGAS Evaluation Metrics
--------------------------

RAGAS (Retrieval Augmented Generation Assessment) provides four metrics:

1. Faithfulness — measures whether claims in the generated answer are
   supported by the retrieved context chunks. A faithfulness of 1.0 means
   zero hallucination — every claim traces back to a chunk.

2. Answer Relevancy — measures whether the answer actually addresses the
   question asked. Uses semantic similarity between the question and
   hypothetical questions generated from the answer.

3. Context Precision — measures whether the retrieved chunks are relevant
   to the question. High precision means no irrelevant noise in the
   retrieved context.

4. Context Recall — measures whether ALL the information needed to fully
   answer the question was retrieved. High recall means nothing was missed.

Versioned Prompts
------------------

Versioned prompts in a production RAG system treat prompt configuration as
code — tracking changes, enabling rollback, and allowing A/B testing without
modifying Python files.  They separate configuration from logic and make it
possible to correlate prompt versions with evaluation scores.  NeuralVault
stores prompts in config/prompts.yaml with a version field.

INSUFFICIENT_EVIDENCE
----------------------

INSUFFICIENT_EVIDENCE is a marker returned by the LLM when the retrieved
chunks do not contain enough information to answer the question reliably.
It prevents hallucination by making the system explicitly refuse rather
than generating a plausible but unsupported answer.  This is a deliberate
design choice — a system that says "I don't know" is more trustworthy than
one that makes up answers.

Embedding Model and Vectors
-----------------------------

NeuralVault uses all-MiniLM-L6-v2 from the sentence-transformers library.
It encodes text into 384-dimensional vectors representing semantic meaning.
The model is approximately 80MB, downloads once, and runs entirely on CPU
— no GPU required.  Each vector is a list of 384 floating-point numbers.

LLM and Inference Provider
----------------------------

NeuralVault uses Llama 3.3 70B as the LLM for answer generation.  The
inference provider is Groq, which runs the model on custom LPU (Language
Processing Unit) hardware providing extremely fast inference.  Groq offers
a free tier suitable for personal projects.

Cosine Similarity
------------------

Cosine similarity measures the angle between two vectors.  A score of 1.0
means the vectors point in the same direction (identical meaning) and 0.0
means they are perpendicular (completely unrelated).  In retrieval, the
user's query is embedded into a vector, and cosine similarity is computed
against all stored chunk vectors to find the most semantically relevant
chunks.  ChromaDB uses cosine as its distance metric by default in
NeuralVault.
"""


def main():
    print("Seeding NeuralVault knowledge base for evaluation...")
    count = ingest_text(RAG_NOTES.strip(), topic="RAG", source="eval_seed_notes")
    print(f"  Ingested {count} chunks under topic 'RAG'")
    print("  Ready for eval.py\n")


if __name__ == "__main__":
    main()
