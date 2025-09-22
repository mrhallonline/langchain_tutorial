def select_top_snippets(docs, max_chars=1400):
    buf, out = 0, []
    for d in docs:
        t = d["text"]
        if buf + len(t) <= max_chars:
            out.append(t); buf += len(t)
        else:
            break
    return out