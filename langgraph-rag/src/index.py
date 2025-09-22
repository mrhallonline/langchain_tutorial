import faiss, numpy as np
from typing import List, Dict

class FaissIndex:
    def __init__(self, dim:int):
        self.index = faiss.IndexFlatIP(dim)
        self._meta = []

    def add(self, vectors:np.ndarray, metas:List[Dict]):
        assert vectors.shape[0] == len(metas)
        self.index.add(vectors.astype(np.float32))
        self._meta.extend(metas)

    def search(self, qvec:np.ndarray, k:int=5):
        D, I = self.index.search(qvec.astype(np.float32), k)
        return D[0], [self._meta[i] for i in I[0]]