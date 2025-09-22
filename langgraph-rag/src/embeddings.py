from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=2)
def get_embedder(name:str):
    return SentenceTransformer(name)

def embed_texts(texts, name="sentence-transformers/all-MiniLM-L6-v2"):
    model = get_embedder(name)
    return model.encode(texts, normalize_embeddings=True)