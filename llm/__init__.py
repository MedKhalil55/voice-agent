"""LLM package.

Public API:
- generate_ai_response: single-turn response generation via ChatOllama.
- warmup_llm: preload the LLM model.
"""

from .agent import generate_ai_response, warmup_llm

__all__ = [
    "generate_ai_response",
    "warmup_llm",
]
