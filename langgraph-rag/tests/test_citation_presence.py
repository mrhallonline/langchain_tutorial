from src.agents import build_graph
from src.corpus import ensure_minimal_corpus, load_corpus
from src.chunkers import sentence_merge_chunk
from src.retrieval import HybridRetriever

def test_answer_includes_citations(monkeypatch):
    ensure_minimal_corpus()
    docs = load_corpus()
    chunked=[]
    for d in docs:
        for i,c in enumerate(sentence_merge_chunk(d["text"])):
            chunked.append({"id":f"{d['id']}-{i}","text":c["text"]})
    retriever = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
    graph = build_graph(retriever)

    # Monkeypatch LLM with a dummy that echoes with no citations; our attributor should add them.
    from langchain.chat_models import init_chat_model
    def fake_llm():
        class F: 
            def invoke(self, msgs): 
                class M: content = "RAG uses retrieval with generation."
                return M()
        return F()
    import src.agents as agents
    agents._llm = fake_llm

    out = graph.invoke({"query":"Define RAG."})
    assert "[#" in out["answer"]