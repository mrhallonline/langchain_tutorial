# %% [markdown]
# # 00 — Setup & Sanity Checks
# Create/activate env, install requirements, configure .env, verify devices.

# %%
import os, platform
print("Python:", platform.python_version())
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch not available:", e)

# %%
from dotenv import load_dotenv; load_dotenv()
print("LLM_PROVIDER:", os.getenv("LLM_PROVIDER"))
print("EMBED_MODEL:", os.getenv("EMBED_MODEL"))

# %% [markdown]
# **You should now be able to…**
# - Activate environment and import packages.
# - Load API keys from `.env`.
# - Confirm device availability.
#
# **Challenges**
# 1. Switch to a different embedding model and re-run.
# 2. Try CPU-only vs GPU and time the difference.
# 3. Add a 'mock LLM' mode if no API keys are set.