"""Provider abstraction. LangChain's own interfaces are the abstraction —
swapping providers is a config value, not a code change."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

LLM = BaseChatModel
EmbeddingsModel = Embeddings

__all__ = ["LLM", "EmbeddingsModel"]
