LM models helper

This package provides a small loader for language models used by DIRT. It supports:

- Local Hugging Face models (via `transformers`) — `provider='hf'`
- OpenAI API wrapper (optional) — `provider='openai'`

Usage examples:

```py
from lm_models import get_model
m = get_model("gpt2", provider="hf")
print(m.generate("Hello world"))
```

Notes:
- This is a lightweight compatibility layer. Install `transformers` or `openai` as needed.
- Do not store API keys in source. Use environment variables.
