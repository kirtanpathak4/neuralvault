from sentence_transformers import SentenceTransformer

# This model downloads once (~80MB), then runs locally forever — free
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts list of text chunks → list of vectors.
    Each vector = 384 numbers representing the meaning of that chunk.
    ChromaDB stores these and finds similar ones at search time.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    """
    Same as above but for a single question when you search.
    Your question also becomes a vector, then ChromaDB finds
    the stored chunks closest to it in meaning.
    """
    return model.encode([query]).tolist()[0]