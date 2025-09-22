from dotenv import load_dotenv
import os
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
USE_CRITIC   = os.getenv("USE_CRITIC", "true").lower() == "true"
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "700"))