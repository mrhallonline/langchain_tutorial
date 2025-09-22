# %% [markdown]
# # 01 — Corpus & Chunking

# %%
from src.corpus import ensure_minimal_corpus, load_corpus
from src.chunkers import fixed_chunk, paragraph_chunk, sentence_merge_chunk
ensure_minimal_corpus()
docs = load_corpus()
sample = docs[0]["text"]

chunks_fixed = fixed_chunk(sample, size=200, overlap=50)
chunks_para = paragraph_chunk(sample)
chunks_sent = sentence_merge_chunk(sample, max_len=400)
print("counts:", len(chunks_fixed), len(chunks_para), len(chunks_sent))

lengths = [len(c["text"]) for d in docs for c in sentence_merge_chunk(d["text"])]
print("len stats (min/avg/max):", min(lengths), sum(lengths)//len(lengths), max(lengths))

# %% [markdown]
# **Trade-offs**: Fixed vs Paragraph vs Sentence-merge.
# **Default**: sentence-merge with max_len≈400.
#
# **You should now be able to…**
# - Apply different chunkers and read size stats.
#
# **Challenges**
# 1. Add a title/metadata field to chunks.
# 2. Plot a histogram of chunk lengths.
# 3. Avoid fragments under 50 chars.