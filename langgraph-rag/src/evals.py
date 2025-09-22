import json
from pathlib import Path
from typing import Dict

def load_eval(path:Path)->list:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def faithfulness_check(answer:str, must_cite:bool=True)->bool:
    if must_cite and "[#" not in answer:
        return False
    return True

def run_micro_eval(graph, retriever, eval_items:list)->Dict:
    passed = 0
    results = []
    for item in eval_items:
        state = graph.invoke({"query": item["query"]})
        ok = faithfulness_check(state["answer"], item.get("must_cite",True))
        results.append({"query":item["query"], "passed":ok, "answer":state["answer"][:200]})
        if ok: passed += 1
    return {"passed": passed, "total": len(eval_items), "results": results}