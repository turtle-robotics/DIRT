"""Simple loader for LM backends used in DIRT.

Design:
- `get_model(name, provider='hf'|'openai')` returns an object with `generate(prompt, **kwargs)`.
- Uses runtime guards so importing the package doesn't require heavy deps.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
import os
import logging

log = logging.getLogger(__name__)


class ModelInterface:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class LocalHFModel(ModelInterface):
    def __init__(self, name: str):
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("transformers required for LocalHFModel") from exc
        # small text-generation pipeline; user can replace with a streaming/LLM client
        self._pipe = pipeline("text-generation", model=name)

    def generate(self, prompt: str, max_length: int = 128, **kwargs) -> str:
        out = self._pipe(prompt, max_length=max_length, do_sample=False)
        return out[0]["generated_text"]


class OpenAIModel(ModelInterface):
    def __init__(self, name: str):
        try:
            import openai
        except Exception as exc:
            raise RuntimeError("openai package required for OpenAIModel") from exc
        self._client = openai
        self._name = name

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self._client.api_key = api_key
        resp = self._client.Completion.create(engine=self._name, prompt=prompt, max_tokens=max_tokens)
        return resp.choices[0].text


def get_model(name: str = "gpt2", provider: str = "hf") -> ModelInterface:
    """Return a model backend instance.

    provider: 'hf' for Hugging Face local models, 'openai' for OpenAI API.
    """
    if provider == "hf":
        return LocalHFModel(name)
    elif provider == "openai":
        return OpenAIModel(name)
    else:
        raise ValueError("Unknown provider")
