from src.corpus import ensure_minimal_corpus, load_corpus
from src.chunkers import sentence_merge_chunk
from src.retrieval import HybridRetriever

def test_retriever_basic():
    ensure_minimal_corpus()
    docs = load_corpus()
    chunked=[]
    for d in docs:
        for i,c in enumerate(sentence_merge_chunk(d["text"])):
            chunked.append({"id":f"{d['id']}-{i}","text":c["text"]})
    r = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
    hits = r.query("What is RAG?", k=5)
    assert len(hits) > 0
    assert "text" in hits[0]