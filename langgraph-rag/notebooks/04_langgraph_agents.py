# %% [markdown]
# # 04 — LangGraph Agents

# %%
from src.corpus import load_corpus
from src.chunkers import sentence_merge_chunk
from src.retrieval import HybridRetriever
from src.agents import build_graph

docs = load_corpus()
chunked=[]
for d in docs:
    for i,c in enumerate(sentence_merge_chunk(d["text"])):
        chunked.append({"id":f"{d['id']}-{i}","text":c["text"],"title":d["title"]})

retriever = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
graph = build_graph(retriever)

out = graph.invoke({"query":"What does LangGraph add?"})
print(out["answer"])
print("Critique:", out.get("critique"))