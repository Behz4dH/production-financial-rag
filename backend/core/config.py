"""
Centralized configuration.
Uses pydantic-settings for validated environment variables. Nothing else in
the codebase should read environment variables directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM / embeddings ---
    groq_api_key: str
    llm_provider: str = "groq"
    primary_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "llama-3.3-70b-versatile"
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Vector store / data ---
    vector_store: str = "chroma"
    chroma_dir: str = "data/chroma"
    data_dir: str = "data/docs"
    embedding_cache_dir: str = "data/embeddings_cache"
    doc_metadata_path: str = "data/doc_metadata.json"
    docstore_path: str = "data/docstore.jsonl"
    collection_name: str = "financial_reports"
    metadata_extract_pages: int = 3
    metadata_extract_chars: int = 5000
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    retrieval_mode: str = "hybrid"
    top_k: int = 20
    top_n: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    refusal_score_threshold: float = 0.3

    # --- Security / reliability / limits ---
    enable_llm_guard: bool = False
    llm_max_retries: int = 3       # LangChain/Groq SDK's own transport-level retries
    request_max_retries: int = 3   # our with_retry() wrapper around a whole /chat call
    agentic_max_retries: int = 3   # agentic loop: retries of retrieval after a weak-relevance grade
    # Token budget for the assembled generation prompt (question + retrieved
    # excerpts) — NOT a limit on the raw request; ChatRequest.message is
    # separately capped at 2000 characters (~500 tokens) for input hygiene.
    max_context_tokens: int = 8000
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300

    # --- Observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "financial-rag"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()
