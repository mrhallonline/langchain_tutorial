# Multi-Agent, RAG-Grounded Chatbot with LangGraph
A graduate-level, hands-on tutorial (model-agnostic, laptop-friendly).

## Why LangGraph (vs chains)?
Use LangGraph when you need: (1) explicit state (documents, citations, control signals),
(2) branching/loops (retries, re-queries, critiques), and (3) multiple competent roles
(rewriter, retriever, attributor, answerer, critic).

### Failure modes you’ll address
- Shallow retrieval → hybrid search + rerank + relevance feedback
- Hallucination → Attributor + Critic/Verifier that checks claim support
- Latency creep → cheap defaults, toggles for reranker/critic, caching
- Evaluation blindness → micro-evals: faithfulness, citation coverage

## Quickstart
```bash
conda env create -f environment.yml
conda activate langgraph-rag
pip install -r requirements.txt

# Add your API key(s) to .env (copy .env.example → .env)
pytest -q
streamlit run app.py
```

## Architectural variants
| Variant | Summary | Complexity | Latency | Quality | When to use |
|---|---|---:|---:|---:|---|
| Fixed multi-agent (this tutorial) | Rewrite→Retrieve→Answer→Critic | Medium | Medium | High | Most classroom/PD demos |
| Single-agent “planner” | One agent plans & calls tools | Low | Low | Medium | Prototyping |
| Multi-stage w/ reranker | Add cross-encoder rerank | Med-High | High | Higher | Noisy corpora |

## Troubleshooting
- Windows CUDA conflicts → run CPU-only; embeddings are fast enough.
- Token limits → reduce snippets via `select_top_snippets`.
- Rate limits → set `temperature=0.2`, `max_tokens=700`, batch evals.

## Roadmap
- DSPy or structured outputs (JSON-schema)
- NLI-style verifier / self-consistency
- FastAPI backend; Streamlit → web frontends