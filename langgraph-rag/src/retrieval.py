from rank_bm25 import BM25Okapi
import numpy as np
from .embeddings import embed_texts

class HybridRetriever:
    def __init__(self, docs, embed_model):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        # BM25
        self.tokens = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        # Dense
        embs = embed_texts(self.texts, name=embed_model)
        self.embs = np.array(embs, dtype=np.float32)
        self.embs = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True)+1e-9)

    def query(self, q, k=8, alpha=0.55):
        qemb = embed_texts([q])[0]
        qemb = qemb / (np.linalg.norm(qemb)+1e-9)
        dense_scores = self.embs @ qemb
        bm25_scores = self.bm25.get_scores(q.split())
        d = (dense_scores - dense_scores.min()) / (dense_scores.ptp()+1e-9)
        b = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp()+1e-9)
        mix = alpha*d + (1-alpha)*b
        idx = np.argsort(-mix)[:k]
        return [self.docs[i] | {"score": float(mix[i])} for i in idx]