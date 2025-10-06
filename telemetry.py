import os
from typing import Optional, Dict, Any

from .config import AppConfig


def enable_langsmith(config: AppConfig, run_tags: Optional[list] = None, run_metadata: Optional[Dict[str, Any]] = None) -> None:
    if config.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    if config.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.langchain_tracing_v2 else "false"
    if run_tags:
        os.environ["LANGCHAIN_RUN_TAGS"] = ",".join(run_tags)
    if run_metadata:
        # LangChain reads JSON-like metadata only when passed per-run; we keep this env for reference
        os.environ["LANGCHAIN_RUN_METADATA"] = str(run_metadata)

