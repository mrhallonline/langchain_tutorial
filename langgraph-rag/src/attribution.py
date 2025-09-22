def force_inline_citations(answer:str, sources:list):
    if "[#" in answer or not sources:
        return answer
    cites = " ".join(f"[#{s.get('id','?')}]" if isinstance(s, dict) else "[#?]" for s in sources[:2])
    return f"{answer}\n\nSources: {cites}"

def supported_by_sources(claim:str, snippets:list)->bool:
    text = " ".join(snippets).lower()
    terms = [t for t in claim.lower().split() if len(t)>4]
    hits = sum(1 for t in terms if t in text)
    return hits >= max(2, len(terms)//3)