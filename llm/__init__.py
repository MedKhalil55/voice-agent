"""LLM package.

Public API:
- generate_ai_response: single-turn response generation via ChatOllama.
"""

from .agent import generate_ai_response

__all__ = ["generate_ai_response"]
