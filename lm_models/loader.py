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
    """Interface for a text-generation backend: ``generate(prompt) -> str``."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Return generated text for ``prompt``. Backends accept extra keyword options.
        """
        raise NotImplementedError


class LocalHFModel(ModelInterface):
    """Runs a Hugging Face model locally through ``transformers.pipeline``."""

    def __init__(self, name: str):
        """Load the model (downloads it on first use).

        Args:
            name: Hugging Face model id, for example ``"gpt2"``.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
        """
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("transformers required for LocalHFModel") from exc
        # small text-generation pipeline; user can replace with a streaming/LLM client
        self._pipe = pipeline("text-generation", model=name)

    def generate(self, prompt: str, max_length: int = 128, **kwargs) -> str:
        """Generate text greedily (no sampling, so output is repeatable).

        Args:
            prompt: Input text.
            max_length: Maximum total length in tokens, prompt included.

        Returns:
            The prompt followed by the generated continuation.
        """
        out = self._pipe(prompt, max_length=max_length, do_sample=False)
        return out[0]["generated_text"]


class OpenAIModel(ModelInterface):
    """Calls the OpenAI completions API.

    Uses the pre-1.0 ``openai`` package interface (``openai.Completion``), which
    newer versions of the package removed. Pin ``openai<1`` or update this class
    before using it. Reads the key from the ``OPENAI_API_KEY`` environment
    variable; never put keys in the code.
    """

    def __init__(self, name: str):
        """Store the model name.

        Args:
            name: OpenAI model/engine name.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        try:
            import openai
        except Exception as exc:
            raise RuntimeError("openai package required for OpenAIModel") from exc
        self._client = openai
        self._name = name

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> str:
        """Send one completion request.

        Raises:
            RuntimeError: If ``OPENAI_API_KEY`` is not set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self._client.api_key = api_key
        resp = self._client.Completion.create(engine=self._name, prompt=prompt, max_tokens=max_tokens)
        return resp.choices[0].text


def get_model(name: str = "gpt2", provider: str = "hf") -> ModelInterface:
    """Return a model backend instance.

    Args:
        name: Model name or id for the chosen provider.
        provider: ``"hf"`` for a local Hugging Face model, ``"openai"`` for the
            OpenAI API.

    Raises:
        ValueError: If ``provider`` is not recognised.
    """
    if provider == "hf":
        return LocalHFModel(name)
    elif provider == "openai":
        return OpenAIModel(name)
    else:
        raise ValueError("Unknown provider")
