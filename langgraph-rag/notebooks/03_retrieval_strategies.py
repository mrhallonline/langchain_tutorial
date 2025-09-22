# %% [markdown]
# # 03 — Retrieval Strategies

# %%
from src.retrieval import HybridRetriever
from src.corpus import load_corpus
from src.chunkers import sentence_merge_chunk

docs = load_corpus()
chunked=[]
for d in docs:
    for i,c in enumerate(sentence_merge_chunk(d["text"])):
        chunked.append({"id":f"{d['id']}-{i}","text":c["text"],"title":d["title"]})

retriever = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
for q in ["What is RAG?","Why hybrid retrieval?"]:
    hits = retriever.query(q, k=5)
    print(q, "→", [h["id"] for h in hits])

# %% [markdown]
# **You should now be able to…**
# - Run hybrid retrieval and inspect top-k.
#
# **Challenges**
# 1. Implement pure BM25 baseline and compare ids.
# 2. Add a cosine reranker over top-20 dense hits.
# 3. Vary alpha in hybrid and observe changes.