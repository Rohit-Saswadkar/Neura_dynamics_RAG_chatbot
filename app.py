import os
import logging
from typing import List

import streamlit as st

from src.config import load_config
from src.llm import get_llm, get_embeddings
from src.vectorstore import create_qdrant_client, get_vectorstore, get_chroma_vectorstore
from src.ingest import ingest_pdfs_into_qdrant
from src.graph import build_agent_graph
from src.memory import AgentMemory
from src.telemetry import enable_langsmith


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
    if "embeddings_provider" not in st.session_state:
        st.session_state.embeddings_provider = "sentence"


def main():
    st.set_page_config(page_title="Neura Dynamics - AI Agent", layout="wide")
    init_state()

    # Configure terminal logging (idempotent across Streamlit reruns)
    def _configure_logging():
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger = logging.getLogger("neura")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    logger = _configure_logging()

    cfg = load_config()
    enable_langsmith(cfg, run_tags=["neura-dynamics"], run_metadata={"ui": "streamlit"})

    def evaluate_and_log(config, question: str, result: dict) -> None:
        """LLM-based quick evaluation; logs a LangSmith evaluation run when available.

        Best-effort; silently no-ops if LangSmith or the model isn't configured.
        """
        try:
            from langsmith import Client  # type: ignore
            import re
            from datetime import datetime, timezone

            llm = get_llm(config)
            answer = (result or {}).get("answer") or ""
            tool = (result or {}).get("tool") or "unknown"
            prompt = (
                "You are an evaluator. Score the assistant's answer on a 1-5 scale for correctness, "
                "groundedness, and relevance to the user's question. Provide a short reason.\n"
                "Return strictly in the format: 'SCORE: <int 1-5> | REASON: <one sentence>'.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Tool: {tool}\n"
            )
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            m = re.search(r"SCORE\s*:\s*(\d+)\s*\|\s*REASON\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                score = int(m.group(1))
                reason = m.group(2).strip()
            else:
                score = 3
                reason = "Defaulted: could not parse model evaluation output."

            client = Client()
            now = datetime.now(timezone.utc)
            client.create_run(
                name="auto_eval_answer_quality",
                inputs={"question": question, "answer": answer, "tool": tool},
                outputs={"score": score, "reason": reason},
                run_type="evaluation",
                start_time=now,
                end_time=now,
                tags=["neura-dynamics", "auto-eval"],
            )
        except Exception:
            # Ignore any evaluation/logging errors to avoid impacting UX
            pass

    with st.sidebar:
        st.header("Settings")
        st.caption("Provide keys in .env or here (session only)")
        google_key = st.text_input("Google API Key", value=cfg.google_api_key or "", type="password")
        owm_key = st.text_input("OpenWeatherMap API Key", value=cfg.openweather_api_key or "", type="password")
        langsmith_key = st.text_input("LangSmith API Key", value=cfg.langsmith_api_key or "", type="password")
        qdrant_url = st.text_input("Qdrant URL", value=cfg.qdrant_url)
        st.session_state.embeddings_provider = st.selectbox("Embedding provider", ["sentence", "google"], index=0)
        show_chunks = st.toggle("Show retrieved chunks", value=False)
        rag_top_k = st.slider("RAG: number of chunks (k)", min_value=4, max_value=20, value=8, step=1)

        os.environ["GOOGLE_API_KEY"] = google_key or ""
        os.environ["OPENWEATHER_API_KEY"] = owm_key or ""
        os.environ["LANGSMITH_API_KEY"] = langsmith_key or ""
        os.environ["QDRANT_URL"] = qdrant_url or cfg.qdrant_url

        # Reload config after user updates environment values
        cfg = load_config()
        enable_langsmith(cfg, run_tags=["neura-dynamics"], run_metadata={"ui": "streamlit"})

        st.divider()
        st.subheader("Add documents")
        uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Ingest PDFs"):
            if not uploaded:
                st.warning("Please upload at least one PDF.")
            else:
                try:
                    embeddings = get_embeddings(cfg, provider=st.session_state.embeddings_provider)
                    qclient = create_qdrant_client(cfg)
                    vstore = get_vectorstore(qclient, cfg.qdrant_collection, embeddings)
                    paths: List[str] = []
                    os.makedirs("data", exist_ok=True)
                    for f in uploaded:
                        path = os.path.join("data", f.name)
                        with open(path, "wb") as out:
                            out.write(f.read())
                        paths.append(path)
                    # Replace existing collection with only these PDFs
                    count = ingest_pdfs_into_qdrant(cfg, vstore, embeddings, paths, replace=True)
                    st.session_state.vectorstore = vstore
                    # Rebuild graph to ensure it uses the latest vectorstore
                    st.session_state.graph = build_agent_graph(cfg, st.session_state.vectorstore, embeddings, rag_top_k=rag_top_k, memory=st.session_state.memory)
                    st.success(f"Ingested {count} chunks into Qdrant.")
                except Exception as e:
                    st.error("Ingestion failed. See error details below.")
                    st.exception(e)
                    with st.expander("How to start Qdrant (Docker)"):
                        st.code("docker run -p 6333:6333 -v ${PWD}\\.qdrant:/qdrant/storage qdrant/qdrant:latest", language="bash")

        # Optional: ingest any PDFs already present in ./data without uploading
        if st.button("Ingest PDFs from data/ folder"):
            try:
                data_dir = "data"
                os.makedirs(data_dir, exist_ok=True)
                paths: List[str] = [
                    os.path.join(data_dir, p)
                    for p in os.listdir(data_dir)
                    if p.lower().endswith(".pdf")
                ]
                if not paths:
                    st.info("No PDFs found in ./data. Upload files or place PDFs in the data folder.")
                else:
                    embeddings = get_embeddings(cfg, provider=st.session_state.embeddings_provider)
                    qclient = create_qdrant_client(cfg)
                    vstore = get_vectorstore(qclient, cfg.qdrant_collection, embeddings)
                    # Replace existing collection with only these PDFs
                    count = ingest_pdfs_into_qdrant(cfg, vstore, embeddings, paths, replace=True)
                    st.session_state.vectorstore = vstore
                    # Rebuild graph to ensure it uses the latest vectorstore
                    st.session_state.graph = build_agent_graph(cfg, st.session_state.vectorstore, embeddings, rag_top_k=rag_top_k, memory=st.session_state.memory)
                    st.success(f"Ingested {count} chunks from data/ into Qdrant.")
            except Exception as e:
                st.error("Ingestion failed. See error details below.")
                st.exception(e)
                with st.expander("How to start Qdrant (Docker)"):
                    st.code("docker run -p 6333:6333 -v ${PWD}\\.qdrant:/qdrant/storage qdrant/qdrant:latest", language="bash")

        st.divider()
        st.subheader("Diagnostics")
        if st.button("Run diagnostics"):
            details = {}
            try:
                # Embedding dimension probe
                emb = get_embeddings(cfg, provider=st.session_state.embeddings_provider)
                vec = emb.embed_query("dimension_probing_string")
                details["embedding_provider"] = st.session_state.embeddings_provider
                details["embedding_dimension"] = len(vec)
            except Exception as e:
                details["embedding_error"] = str(e)
            try:
                qclient = create_qdrant_client(cfg)
                collections = qclient.get_collections()
                details["qdrant_collections"] = [c.name for c in collections.collections]
                try:
                    info = qclient.get_collection(cfg.qdrant_collection)
                    # points_count may differ by version; include full info repr
                    details["collection_info"] = {
                        "name": cfg.qdrant_collection,
                        "raw": repr(info),
                    }
                except Exception as inner:
                    details["collection_info_error"] = str(inner)
            except Exception as e:
                details["qdrant_error"] = str(e)
            st.json(details)

    st.title("Agentic Chat")

    # Ensure vectorstore and graph exist
    embeddings = get_embeddings(cfg, provider=st.session_state.embeddings_provider)
    vstore = None
    try:
        qclient = create_qdrant_client(cfg)
        vstore = get_vectorstore(qclient, cfg.qdrant_collection, embeddings)
        st.session_state.vectorstore = vstore
        st.caption("Vectorstore: Qdrant")
    except Exception:
        # Fallback to local Chroma so the app works without Docker
        vstore = get_chroma_vectorstore(cfg.qdrant_collection, embeddings)
        st.session_state.vectorstore = vstore
        st.warning("Qdrant is not reachable. Using local Chroma fallback (persistent in ./.chroma).")

    graph = build_agent_graph(cfg, vstore, embeddings, rag_top_k=rag_top_k, memory=st.session_state.memory)
    st.session_state.graph = graph

    try:
        has_docs = bool(vstore and vstore.similarity_search("test", k=1))
    except Exception:
        has_docs = False
    if not has_docs:
        st.info("No documents found or Qdrant offline. You can still ask questions; add docs when ready.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            tool_tag = (m or {}).get("tool")
            if tool_tag == "weather":
                weather = (m or {}).get("weather") or {}
                st.markdown("Status badge: Weather API")
                st.markdown(m.get("content") or "")
                st.caption(
                    f"Location: {weather.get('location')} | Units: {weather.get('units')} | Timestamp: {weather.get('timestamp')}"
                )
            elif tool_tag in {"pdf_rag", "rag"}:
                st.markdown("Status badge: PDF RAG")
                st.markdown(m.get("content") or "")
                sources = (m or {}).get("sources") or []
                if sources:
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.write(f"- {s['source']} | score={s['score']:.3f} | page={s.get('page')}")
            else:
                st.markdown(m.get("content") or "")

    user_input = st.chat_input("Ask about weather or your PDFs...")
    if user_input:
        logger.info("User message: %s", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("Status: routing...")
            try:
                # Pass only the last 2 messages to keep context minimal
                history = st.session_state.messages[-2:]
                logger.debug("Invoking agent with chat_history_len=%s", len(history))
                result = st.session_state.graph.invoke({"question": user_input, "chat_history": history})
            except Exception as e:
                status_placeholder.markdown("Status: error")
                st.error(f"Agent error: {e}")
                result = {"tool": "error", "answer": "Sorry, something went wrong running the agent."}
            tool = result.get("tool")
            logger.info("Agent routed to: %s", tool)
            assistant_markdown = ""
            if tool == "weather":
                status_placeholder.markdown("Status badge: Weather API")
                weather = result.get("weather", {})
                assistant_markdown = (
                    "Status badge: Weather API\n\n"
                    f"**Answer**: {result.get('answer')}\n\n"
                    f"Location: {weather.get('location')} | Units: {weather.get('units')} | Timestamp: {weather.get('timestamp')}"
                )
                st.markdown(assistant_markdown)
                logger.debug("Weather context: location=%s units=%s ts=%s", weather.get("location"), weather.get("units"), weather.get("timestamp"))
            else:
                status_placeholder.markdown("Status badge: PDF RAG")
                sources = result.get("sources", [])
                logger.debug("RAG sources_count=%s", len(sources))
                if sources:
                    src_lines = [f"- {s['source']} | score={s['score']:.3f} | page={s.get('page')}" for s in sources]
                    src_md = "\n".join(src_lines)
                    assistant_markdown = (
                        "Status badge: PDF RAG\n\n"
                        f"**Answer**: {result.get('answer')}\n\n"
                        "**Sources:**\n" + src_md
                    )
                else:
                    assistant_markdown = (
                        "Status badge: PDF RAG\n\n"
                        f"**Answer**: {result.get('answer')}"
                    )
                st.markdown(assistant_markdown)
                if sources and show_chunks:
                    for s in sources:
                        with st.expander("Preview"):
                            st.code(s.get("snippet") or "", language="markdown")
            # Fire-and-forget evaluation (LangSmith-traced if configured)
            evaluate_and_log(cfg, user_input, result)
        # Persist structured assistant message so prior answers re-render with details
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_markdown or result.get("answer"),
            "tool": result.get("tool"),
            "weather": result.get("weather"),
            "sources": result.get("sources"),
        })


if __name__ == "__main__":
    main()

