from typing import Optional
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Prefer new package, fall back to community for compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except Exception:  # pragma: no cover - fallback for older envs
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings as _HuggingFaceEmbeddings,
    )

from .config import AppConfig


def get_llm(
    config: AppConfig,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    temp = config.llm_temperature if temperature is None else temperature
    mot = config.llm_max_output_tokens if max_output_tokens is None else max_output_tokens
    return ChatGoogleGenerativeAI(
        model=config.gemini_model,
        google_api_key=config.google_api_key,
        temperature=temp,
        max_output_tokens=mot,
    )


def get_embeddings(config: AppConfig, provider: str = "sentence"):
    provider_lower = (provider or "").strip().lower()
    if provider_lower in ("google", "gemini", "google-genai"):
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model=config.google_embedding_model,
            google_api_key=config.google_api_key,
        )
    # Force CPU device to avoid PyTorch "meta tensor" issues on some Windows setups
    try:
        model_name = config.embedding_model
        normalize = os.getenv("EMBEDDING_NORMALIZE", "auto").lower()
        # Auto-enable normalization for BGE models unless explicitly disabled
        should_normalize = (
            normalize == "true" or (normalize == "auto" and ("bge" in (model_name or "").lower()))
        )
        encode_kwargs = {"normalize_embeddings": True} if should_normalize else {}
        return _HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs=encode_kwargs,
        )
    except Exception as e:
        # Graceful fallback to Google embeddings when available
        if config.google_api_key:
            return GoogleGenerativeAIEmbeddings(
                model=config.google_embedding_model,
                google_api_key=config.google_api_key,
            )
        raise

