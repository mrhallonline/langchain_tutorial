from src.chunkers import fixed_chunk, sentence_merge_chunk

def test_fixed_chunk_overlap():
    text = "a"*1000
    chunks = fixed_chunk(text, size=200, overlap=50)
    assert len(chunks) > 3
    assert all(len(c["text"])<=200 for c in chunks)

def test_sentence_merge():
    text = "A. B. C. D. E."
    chunks = sentence_merge_chunk(text, max_len=5)
    assert len(chunks) >= 3