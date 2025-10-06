from typing import Dict, List, Tuple, Optional
import os

from langchain_core.prompts import PromptTemplate
import logging


ANSWER_PROMPT = PromptTemplate(
    template=(
        "You are a careful, evidence-based assistant. Answer ONLY using the information in the context.\n"
        "- Synthesize across ALL relevant snippets; do not copy verbatim.\n"
        "- If information conflicts or is incomplete, state the uncertainty.\n"
        "- Be concise and directly answer the question.\n"
        "- Include a 'Sources' section listing source and page (if available).\n\n"
        "Question:\n{question}\n\n"
        "Context (multiple snippets):\n{context}\n\n"
        "Answer:"
    ),
    input_variables=["question", "context"],
)


def _format_chat_history_for_rewrite(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 6) -> str:
    """Return a compact, readable transcript from the last N turns.

    Input is expected to be a list of dicts with keys 'role' in {'user','assistant'} and 'content'.
    """
    if not chat_history:
        return ""
    recent = chat_history[-max_turns:]
    lines: List[str] = []
    for m in recent:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def _rewrite_question_with_history(llm, question: str, chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Use the LLM to turn anaphoric follow-ups into a standalone question.

    Best-effort: if the rewrite fails, fall back to the original question.
    """
    try:
        logger = logging.getLogger("neura.rag")
        transcript = _format_chat_history_for_rewrite(chat_history)
        if not transcript:
            return question
        prompt = (
            "You are a helpful assistant that rewrites a user's follow-up into a standalone question.\n"
            "- Incorporate necessary details from the conversation history.\n"
            "- Keep it concise and faithful.\n"
            "Return only the rewritten question, no preamble.\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            f"Latest user message: {question}\n"
        )
        logger.debug("Rewriting question with history (len=%s)", 0 if not chat_history else len(chat_history))
        rewritten = llm.invoke(prompt)
        text = getattr(rewritten, "content", str(rewritten)).strip()
        logger.info("Rewritten question: %s", text)
        return text or question
    except Exception:
        return question


def build_context(snippets: List[Tuple[float, dict, str]], head_chars: int = 1200) -> str:
    lines = []
    for score, meta, content in snippets:
        source = meta.get("source") or meta.get("source_path") or "unknown"
        page = meta.get("page")
        head = (content or "")[:head_chars].replace("\n", " ")
        lines.append(f"[score={score:.3f}] {source} page={page} :: {head}")
    return "\n".join(lines)


def retrieve_with_scores(vectorstore, query: str, k: int = 5) -> List[Tuple[float, dict, str]]:
    logger = logging.getLogger("neura.rag")
    if not vectorstore:
        return []
    try:
        use_mmr = os.getenv("RAG_USE_MMR", "false").lower() == "true"
        if use_mmr and hasattr(vectorstore, "max_marginal_relevance_search_with_relevance_scores"):
            results = vectorstore.max_marginal_relevance_search_with_relevance_scores(query, k=k)
        else:
            results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        logger.exception("Vectorstore retrieval failed")
        return []
    tuples: List[Tuple[float, dict, str]] = []
    for doc, score in results:
        tuples.append((score, doc.metadata or {}, doc.page_content or ""))
    logger.debug("Retrieved %s snippets for query", len(tuples))
    return tuples


def rag_answer(llm, vectorstore, question: str, k: int = 8, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict:
    logger = logging.getLogger("neura.rag")
    target_k = k
    rerank_top_n: Optional[int] = None
    try:
        env_val = os.getenv("RAG_RERANK_TOP_N")
        rerank_top_n = int(env_val) if env_val else None
    except Exception:
        rerank_top_n = None

    # Rewrite follow-ups into standalone question to improve retrieval (enabled by default)
    rewrite_flag = os.getenv("RAG_REWRITE_WITH_HISTORY", "true").lower()
    if rewrite_flag in {"1", "true", "yes", "on"}:
        effective_question = _rewrite_question_with_history(llm, question, chat_history)
    else:
        effective_question = question
    logger.info("RAG effective question: %s", effective_question)

    snippets = retrieve_with_scores(vectorstore, effective_question, k=target_k)
    if not snippets:
        answer = (
            "I don't have any indexed documents yet or the vector database is offline. "
            "Please add PDFs in the sidebar to enable grounded answers."
        )
        return {"answer": answer, "sources": []}
    if rerank_top_n and rerank_top_n > 0 and rerank_top_n < len(snippets):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            model_name = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            ce = CrossEncoder(model_name)
            pairs = [(question, s[2]) for s in snippets]
            scores = ce.predict(pairs)
            ranked = sorted(zip(scores, snippets), key=lambda x: float(x[0]), reverse=True)
            snippets = [sn for _score, sn in ranked[:rerank_top_n]]
        except Exception:
            logger.debug("Cross-encoder rerank unavailable; skipping")
            pass
    head_chars = int(os.getenv("RAG_HEAD_CHARS", os.getenv("RAG_HEAD_CHARS")) or os.getenv("RAG_HEAD_CHARS", "1200"))
    try:
        head_chars = int(os.getenv("RAG_HEAD_CHARS", "1200"))
    except Exception:
        head_chars = 1200
    context = build_context(snippets, head_chars=head_chars)
    try:
        max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
    except Exception:
        max_context_chars = 8000
    if len(context) > max_context_chars:
        acc = []
        total = 0
        for line in context.splitlines():
            if total + len(line) + 1 > max_context_chars:
                break
            acc.append(line)
            total += len(line) + 1
        context = "\n".join(acc)
    prompt_text = ANSWER_PROMPT.format(question=effective_question, context=context)
    logger.debug("Calling LLM with built context (chars=%s)", len(context))
    response = llm.invoke(prompt_text)
    answer = getattr(response, "content", str(response)).strip()
    logger.info("LLM answered with %s chars", len(answer))
    sources = []
    for score, meta, content in snippets:
        sources.append({
            "source": meta.get("source") or meta.get("source_path") or "unknown",
            "page": meta.get("page"),
            "score": score,
            "snippet": (content or "")[:600],
            "metadata": meta,
        })
    return {"answer": answer, "sources": sources}

