def variant_fixed_multi_agent(retriever):
    from .agents import build_graph
    return build_graph(retriever)

def variant_single_agent_planner(retriever):
    from .agents import build_graph
    return build_graph(retriever)  # pedagogical shortcut