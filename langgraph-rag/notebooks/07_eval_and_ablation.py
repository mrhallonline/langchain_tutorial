# %% [markdown]
# # 07 — Eval & Ablation

# %%
from pathlib import Path
from src.evals import load_eval, run_micro_eval
from src.corpus import load_corpus
from src.chunkers import sentence_merge_chunk
from src.retrieval import HybridRetriever
from src.agents import build_graph

eval_items = load_eval(Path("data/labeled_eval.jsonl"))
docs = load_corpus()
chunked=[]
for d in docs:
    for i,c in enumerate(sentence_merge_chunk(d["text"])):
        chunked.append({"id":f"{d['id']}-{i}","text":c["text"],"title":d["title"]})

retriever = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
graph = build_graph(retriever)
report = run_micro_eval(graph, retriever, eval_items)
print(report)