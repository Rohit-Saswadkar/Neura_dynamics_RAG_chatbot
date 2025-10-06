import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a .env if present
load_dotenv()


@dataclass
class AppConfig:
    # LLM
    google_api_key: Optional[str]
    gemini_model: str

    # Embeddings
    embedding_model: str
    google_embedding_model: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_collection: str

    # Weather / OWM
    openweather_api_key: Optional[str]
    openweather_base_url: str
    weather_units: str
    weather_cache_ttl_seconds: int
    http_timeout_seconds: float
    http_max_retries: int

    # LangSmith
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    langchain_tracing_v2: bool

    # LLM tuning (defaults)
    llm_temperature: float = 0.2
    llm_max_output_tokens: Optional[int] = None

    # Weather LLM usage toggles (defaults)
    enable_weather_slot_llm: bool = False
    enable_weather_clarification_llm: bool = False
    enable_weather_summary_llm: bool = False

    # Router and RAG tuning (defaults)
    enable_llm_routing: bool = False
    rag_rewrite_with_history: bool = False
    rag_head_chars: int = 600
    rag_max_context_chars: int = 4000
    # Optional features
    enable_history_node: bool = False


def load_config() -> AppConfig:
    return AppConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        gemini_model=os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.0-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_output_tokens=(
            int(os.getenv("LLM_MAX_OUTPUT_TOKENS")) if os.getenv("LLM_MAX_OUTPUT_TOKENS") else None
        ),
        embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        google_embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "neura_docs"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        openweather_base_url=os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5"),
        weather_units=os.getenv("WEATHER_UNITS", "metric"),
        weather_cache_ttl_seconds=int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "300")),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
        http_max_retries=int(os.getenv("HTTP_MAX_RETRIES", "3")),
        enable_weather_slot_llm=os.getenv("WEATHER_SLOT_LLM", "false").lower() == "true",
        enable_weather_clarification_llm=os.getenv("WEATHER_CLARIFICATION_LLM", "false").lower() == "true",
        enable_weather_summary_llm=os.getenv("WEATHER_SUMMARY_LLM", "false").lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "NeuraDynamicsAssignment"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
        enable_llm_routing=os.getenv("ENABLE_LLM_ROUTING", "false").lower() == "true",
        rag_rewrite_with_history=os.getenv("RAG_REWRITE_WITH_HISTORY", "false").lower() == "true",
        rag_head_chars=int(os.getenv("RAG_HEAD_CHARS", "600")),
        rag_max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
        enable_history_node=os.getenv("ENABLE_HISTORY_NODE", "false").lower() == "true",
    )

