from typing import List, Dict
import re

def fixed_chunk(text:str, size:int=400, overlap:int=50)->List[Dict]:
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append({"text":chunk})
        i += max(1, size - overlap)
    return chunks

def paragraph_chunk(text:str)->List[Dict]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [{"text":p} for p in paras]

def sentence_merge_chunk(text:str, max_len:int=450)->List[Dict]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out, buf = [], ""
    for s in sents:
        if len(buf)+len(s) <= max_len:
            buf = f"{buf} {s}".strip()
        else:
            if buf: out.append({"text":buf})
            buf = s
    if buf: out.append({"text":buf})
    return out