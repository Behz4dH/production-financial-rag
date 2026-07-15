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
    # Generation needs a mid-size model or better: small models refuse
    # verbatim answers under the strict matching procedure. The fallback sits
    # in a separate provider quota bucket on purpose.
    primary_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    fallback_model: str = "openai/gpt-oss-120b"
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # The rerank score gates refusal; small models emit degenerate all-zero
    # score batches, so this must never drop below mid-size.
    reranker_llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    rerank_snippet_chars: int = 500  # table-row values can sit hundreds of chars in
    rerank_batch_size: int = 20      # per-call size stays under provider TPM caps

    # --- Vector store / data ---
    vector_store: str = "chroma"
    chroma_dir: str = "data/chroma"
    data_dir: str = "data/docs"
    embedding_cache_dir: str = "data/embeddings_cache"
    doc_metadata_path: str = "data/doc_metadata.json"
    docstore_path: str = "data/docstore.jsonl"
    eval_results_path: str = "data/eval_results.json"
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
    llm_max_retries: int = 3       # SDK transport-level retries
    request_max_retries: int = 3   # with_retry() around a whole /chat call
    retry_base_delay: float = 0.5  # seconds; first backoff step
    # Budget for question + retrieved excerpts only; system prompt and tool
    # schema add ~1000 tokens, and the whole request must clear the provider's
    # per-minute token cap (a single oversized request is an unretryable 413).
    max_context_tokens: int = 4500
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
