"""Build the configured LLM and embeddings from settings.

Defaults: Groq Llama for generation, local HuggingFace sentence-transformers
for embeddings, wrapped in a filesystem-backed cache so re-ingestion never
recomputes an unchanged chunk's vector.
"""

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

from core.config import Settings, get_settings
from rag.providers.base import EmbeddingsModel, LLM


def get_llm(settings: Settings | None = None, model: str | None = None) -> LLM:
    settings = settings or get_settings()
    return init_chat_model(
        model=model or settings.primary_model,
        model_provider=settings.llm_provider,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=settings.llm_max_retries,
    )


def _build_base_embeddings(settings: Settings) -> EmbeddingsModel:
    """The un-cached embedding model. Isolated so tests can swap it."""
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider}")


def get_embeddings(settings: Settings | None = None) -> EmbeddingsModel:
    settings = settings or get_settings()
    base = _build_base_embeddings(settings)
    Path(settings.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(settings.embedding_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        base, store, namespace=settings.embedding_model, key_encoder="sha256"
    )
