from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .retrieval import HybridRetriever
from .attribution import force_inline_citations, supported_by_sources
from .config import USE_CRITIC, TEMPERATURE, MAX_TOKENS, LLM_PROVIDER, LLM_MODEL

# Minimal LLM adapter via langchain's init_chat_model
from langchain.chat_models import init_chat_model

class RAGState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved: List[dict]
    answer: str
    citations: List[str]
    critique: Optional[str]

def _llm():
    return init_chat_model(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        model_provider=LLM_PROVIDER
    )

def node_query_rewriter(state:RAGState)->RAGState:
    llm = _llm()
    msg = llm.invoke([
        SystemMessage(content="Rewrite the user query for retrieval. Keep core intent; add missing key terms; ≤ 25 tokens."),
        HumanMessage(content=state["query"])
    ])
    state["rewritten_query"] = msg.content.strip()
    return state

def node_retriever(state:RAGState, retriever:HybridRetriever)->RAGState:
    rq = state.get("rewritten_query", state["query"])
    docs = retriever.query(rq, k=8)
    state["retrieved"] = docs
    return state

def node_answerer(state:RAGState)->RAGState:
    llm = _llm()
    from .utils import select_top_snippets
    snippets = select_top_snippets(state["retrieved"])
    prompt = [
        SystemMessage(content="You are a grounded assistant. Answer ONLY using the provided sources. If insufficient, say 'I don’t have enough information.' Include inline citations like [#docid]."),
        HumanMessage(content=f"Sources:\n" + "\n---\n".join(snippets)),
        HumanMessage(content=f"Question: {state['query']}")
    ]
    out = llm.invoke(prompt).content
    state["answer"] = force_inline_citations(out, state["retrieved"])
    state["citations"] = [d.get("id","?") for d in state["retrieved"][:2]]
    return state

def node_critic(state:RAGState)->RAGState:
    if not USE_CRITIC: return state
    supported = supported_by_sources(state["answer"], [d["text"] for d in state["retrieved"]])
    if not supported:
        state["critique"] = "Answer may not be fully supported; consider re-query or narrower scope."
    else:
        state["critique"] = "Supported by sources."
    return state

def build_graph(retriever:HybridRetriever):
    g = StateGraph(RAGState)
    g.add_node("rewrite", node_query_rewriter)
    g.add_node("retrieve", lambda s: node_retriever(s, retriever))
    g.add_node("answer", node_answerer)
    g.add_node("critic", node_critic)

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "critic")
    g.add_edge("critic", END)
    return g.compile()