import json
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "data"

def load_corpus(path=DATA/"sample_corpus.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def ensure_minimal_corpus():
    path = DATA/"sample_corpus.jsonl"
    DATA.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Created in packaging step; leave as-is.
        pass
    return path