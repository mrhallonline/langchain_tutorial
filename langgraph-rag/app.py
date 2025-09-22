import streamlit as st
from src.corpus import ensure_minimal_corpus, load_corpus
from src.chunkers import sentence_merge_chunk
from src.retrieval import HybridRetriever
from src.agents import build_graph

st.set_page_config(page_title="LangGraph RAG", layout="wide")
st.title("🧭 Multi-Agent RAG with LangGraph (Demo)")

ensure_minimal_corpus()
docs = load_corpus()

chunked = []
for d in docs:
    for i, c in enumerate(sentence_merge_chunk(d["text"])):
        chunked.append({"id": f"{d['id']}-{i}", "text": c["text"], "title": d["title"]})

retriever = HybridRetriever(chunked, embed_model="sentence-transformers/all-MiniLM-L6-v2")
graph = build_graph(retriever)

q = st.text_input("Ask a question:", "What is RAG and why use it?")
if st.button("Run"):
    with st.spinner("Thinking..."):
        out = graph.invoke({"query": q})
    st.subheader("Answer")
    st.write(out["answer"])
    with st.expander("Retrieved chunks"):
        for r in out["retrieved"][:6]:
            st.markdown(f"**{r['id']}** — {r.get('title','')}")
            st.write(r["text"])
    with st.expander("Trace / Critique"):
        st.write(out.get("critique","(Critic disabled)"))