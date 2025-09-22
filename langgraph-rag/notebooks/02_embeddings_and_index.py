# %% [markdown]
# # 02 — Embeddings & Index

# %%
import time, numpy as np
from src.corpus import load_corpus
from src.chunkers import sentence_merge_chunk
from src.embeddings import embed_texts
from src.index import FaissIndex

docs = load_corpus()
chunked=[]
for d in docs:
    for i,c in enumerate(sentence_merge_chunk(d["text"])):
        chunked.append({"id":f"{d['id']}-{i}", "text":c["text"], "title":d["title"]})

texts = [c["text"] for c in chunked]
t0 = time.time()
embs = embed_texts(texts)
dim = embs.shape[1]
idx = FaissIndex(dim)
idx.add(np.array(embs), chunked)
print("Build ms:", int((time.time()-t0)*1000), "docs:", len(chunked))