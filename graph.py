from typing import Dict, Any, Optional, Literal
import re
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from .config import AppConfig
from .llm import get_llm
from .weather import WeatherClient, llm_to_weather_request
from .rag import rag_answer
from .memory import AgentMemory


def build_agent_graph(config: AppConfig, vectorstore, embeddings, rag_top_k: int = 8, memory: AgentMemory | None = None):
    llm = get_llm(config)
    weather_client = WeatherClient(config)
    mem = memory or AgentMemory()

    class RouterDecision(BaseModel):
        route: Literal["weather", "rag"]
        confidence: float
        reason: Optional[str] = None
        location: Optional[str] = None
        units: Optional[str] = None
        lang: Optional[str] = None
        needs_clarification: Optional[bool] = None
        clarification: Optional[str] = None

    class WeatherSlots(BaseModel):
        """Structured slots for weather queries.

        "when" captures temporal intent as a simple normalized label to guide clarification.
        Supported values: now|today|tomorrow|past|future (others treated as unknown).
        """
        location: Optional[str] = None
        units: Optional[Literal["standard", "metric", "imperial"]] = None
        lang: Optional[str] = None
        when: Optional[Literal["now", "today", "tomorrow", "past", "future", "unknown"]] = None

    logger = logging.getLogger("neura.graph")

    # Removed history meta-intent heuristic to avoid overfitting to specific phrasings.

    def llm_route(text: str, default_units: str = "metric") -> RouterDecision:
        structured = llm.with_structured_output(RouterDecision)
        prompt = (
            "Decide whether the user asks about current weather or about documents.\n"
            "Return fields: route, confidence, reason, location, units, lang, needs_clarification, clarification.\n"
            "- route: 'weather' or 'rag'\n"
            "- confidence: 0.0-1.0 (certainty of your decision)\n"
            "- reason: short rationale for your routing\n"
            "- If weather, extract only the city name in 'location' (no dates); optional 'units'/'lang'.\n"
            "- If city is unclear, set needs_clarification=true and propose a concise clarification question in 'clarification'.\n\n"
            f"User: {text}\n"
        )
        return structured.invoke(prompt)

    def llm_extract_weather_slots(text: str, default_units: str = "metric") -> WeatherSlots:
        structured = llm.with_structured_output(WeatherSlots)
        prompt = (
            "Extract structured weather slots from the user question.\n"
            "- location: city name only (no dates).\n"
            f"- units: one of standard/metric/imperial; default to '{default_units}'.\n"
            "- lang: two-letter code if present.\n"
            "- when: normalize temporal intent into now|today|tomorrow|past|future|unknown.\n\n"
            f"Question: {text}\n"
        )
        return structured.invoke(prompt)

    def cheap_extract_weather_slots(text: str, default_units: str = "metric") -> WeatherSlots:
        t = (text or "")
        location = WeatherClient.parse_location_from_text(t)
        units: Optional[Literal["standard", "metric", "imperial"]] = None
        low = t.lower()
        if any(k in low for k in ["fahrenheit", "f°", "°f"]):
            units = "imperial"
        elif any(k in low for k in ["celsius", "c°", "°c"]):
            units = "metric"
        when: Optional[Literal["now", "today", "tomorrow", "past", "future", "unknown"]] = None
        if re.search(r"\b(tomorrow|next|later)\b", low):
            when = "tomorrow"
        elif re.search(r"\b(yesterday|last)\b", low):
            when = "past"
        elif re.search(r"\b(today|now|currently)\b", low):
            when = "today"
        else:
            when = "unknown"
        return WeatherSlots(location=location, units=units or default_units, when=when)

    def llm_generate_clarification(question: str, slots: WeatherSlots, mem: AgentMemory) -> str:
        present_location = slots.location or mem.last_location
        present_units = slots.units or mem.last_units or config.weather_units
        present_when = slots.when or "unknown"
        sys = (
            "You write ONE short, direct question to get missing info for a weather lookup. "
            "If user seems to ask for forecast or non-current time, explain you can provide current weather only and ask if that's okay. "
            "Prefer concrete city names over vague phrases. Avoid multiple questions; ask the most blocking one."
        )
        context = (
            f"Known so far -> location: {present_location!r}, units: {present_units!r}, when: {present_when!r}.\n"
            f"Original question: {question}"
        )
        prompt = (
            f"{sys}\n\n{context}\n\n"
            "Return only the single question, no preamble."
        )
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    def has_high_confidence_weather_intent(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        # Primary weather terms as whole words; fewer false positives than substring search
        pattern = r"\b(weather|forecast|temperature|temp|humidity|rain|snow|wind|uv|precipitation|storm|sunny|clouds?)\b"
        return re.search(pattern, t) is not None

    def has_high_confidence_doc_intent(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        pattern = r"\b(pdf|document|doc|paper|page|section|figure|table|reference|explain|summarize|according to)\b"
        return re.search(pattern, t) is not None

    def router(state: Dict[str, Any]) -> Dict[str, Any]:
        question_text: str = state.get("question") or ""
        chat_history = state.get("chat_history")
        logger.info("Router received question: %s", question_text)
        # If the previous turn asked for a weather clarification, prefer routing to weather
        if mem.pending_weather_question:
            logger.debug("Pending weather clarification -> forcing weather route")
            return {
                "route": "weather",
                "question": question_text,
                "location": WeatherClient.parse_location_from_text(question_text) or mem.last_location,
                "units": mem.last_units or config.weather_units,
                "lang": mem.last_lang,
                "routing_reason": "follow-up: pending weather clarification",
                "clarification": None,
                "chat_history": chat_history,
            }
        # Step A (rule): high-confidence keywords -> route to weather; extract city via regex/NLP
        if has_high_confidence_weather_intent(question_text):
            city = WeatherClient.parse_location_from_text(question_text)
            logger.debug("Keyword route -> weather (city=%s)", city)
            return {
                "route": "weather",
                "question": question_text,
                "location": city or mem.last_location,
                "units": mem.last_units or config.weather_units,
                "lang": mem.last_lang,
                "routing_reason": "rule: weather keywords",
                # Defer clarification generation to weather node for consistency
                "clarification": None,
                "chat_history": chat_history,
            }
        # Step B (fast heuristic for RAG intent): skip LLM if clear doc intent
        if has_high_confidence_doc_intent(question_text):
            logger.debug("Heuristic route -> rag (doc intent keywords)")
            return {
                "route": "rag",
                "question": question_text,
                "routing_reason": "rule: doc keywords",
                "chat_history": chat_history,
            }
        # Step C (optional): LLM classifier with structured output
        if getattr(config, "enable_llm_routing", False):
            try:
                decision = llm_route(question_text, default_units=config.weather_units)
            except Exception:
                decision = None
            if decision and decision.route == "weather":
                logger.debug("LLM route -> weather (reason=%s)", getattr(decision, "reason", None))
                return {
                    "route": "weather",
                    "question": question_text,
                    # Step C handled downstream: if missing location, weather node asks clarification or uses memory
                    "location": decision.location or mem.last_location,
                    "units": decision.units or mem.last_units or config.weather_units,
                    "lang": decision.lang or mem.last_lang,
                    "routing_reason": decision.reason or "llm classifier",
                    # Defer clarification question to weather node
                    "clarification": None,
                    "chat_history": chat_history,
                }
        # Default route: RAG
        logger.debug("Default route -> rag")
        return {
            "route": "rag",
            "question": question_text,
            "routing_reason": "default: not weather",
            "chat_history": chat_history,
        }

    def do_weather(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        chat_history = state.get("chat_history")
        slots = None
        if getattr(config, "enable_weather_slot_llm", False):
            try:
                slots = llm_extract_weather_slots(question, default_units=config.weather_units)
            except Exception:
                slots = None
        if not slots:
            slots = cheap_extract_weather_slots(question, default_units=config.weather_units)

        # Resolve vague references using chat_history and memory
        location = state.get("location") or slots.location or WeatherClient.parse_location_from_text(question)
        if not location:
            # try last explicit user city in chat_history
            if chat_history:
                for m in reversed(chat_history):
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if role == "user":
                        loc_hist = WeatherClient.parse_location_from_text(content)
                        if loc_hist:
                            location = loc_hist
                            break
        if not location:
            # fallback: parse location from assistant replies (e.g., "in Paris")
            if chat_history:
                for m in reversed(chat_history):
                    content = (m.get("content") or "").strip()
                    loc_hist = WeatherClient.parse_location_from_text(content)
                    if loc_hist:
                        location = loc_hist
                        break
        if not location:
            location = mem.last_location
        units = state.get("units") or slots.units or mem.last_units or config.weather_units
        lang = state.get("lang") or slots.lang or mem.last_lang

        logger.info("Weather node: location=%s units=%s lang=%s", location, units, lang)

        # If location is still unknown, ask one concise clarifying question
        if not location:
            if getattr(config, "enable_weather_clarification_llm", False):
                clar_q = llm_generate_clarification(question, slots, mem)
            else:
                clar_q = "Which city should I check the weather for?"
            mem.set_pending_weather_question(True)
            return {
                "answer": clar_q,
                "tool": "weather",
                "weather": {},
            }

        # If user asked for non-current time, clarify capability
        when_label = (slots.when or "unknown")
        if when_label in {"tomorrow", "future", "past"}:
            if getattr(config, "enable_weather_clarification_llm", False):
                confirm_q = llm_generate_clarification(question, slots, mem)
            else:
                confirm_q = "I can provide current weather only. Would you like the current weather instead?"
            mem.set_pending_weather_question(True)
            return {
                "answer": confirm_q,
                "tool": "weather",
                "weather": {},
            }
        try:
            result = weather_client.fetch_current_weather(location, units=units, lang=lang)
        except Exception as e:
            logger.exception("Weather fetch failed: %s", e)
            # Handle CityNotFoundError with suggestions if available
            try:
                from .weather import CityNotFoundError  # local import to avoid cycles
            except Exception:
                CityNotFoundError = None  # type: ignore
            if CityNotFoundError and isinstance(e, CityNotFoundError):
                candidates = getattr(e, "candidates", [])
                if candidates:
                    options = []
                    for c in candidates[:3]:
                        name = c.get("name")
                        state = c.get("state")
                        country = c.get("country")
                        label = ", ".join([s for s in [name, state, country] if s])
                        options.append(label)
                    msg = "City not found. Did you mean: " + "; ".join(options) + "?"
                else:
                    msg = "City not found. Please provide a valid city name."
                mem.set_pending_weather_question(True)
                return {
                    "answer": msg,
                    "tool": "weather",
                    "weather": {},
                }
            return {
                "answer": f"Weather service error: {e}. Please provide a valid city or try again later.",
                "tool": "weather",
                "weather": {},
            }
        # Once we have complete info and fetched data, clear pending flag
        mem.set_pending_weather_question(False)
        # Summarize result (LLM optional)
        if getattr(config, "enable_weather_summary_llm", False):
            prompt = (
                "Summarize the following OpenWeatherMap data in one paragraph. "
                "Include location, units, and unix timestamp for auditability.\n\n"
                f"Data: {result}\n"
            )
            summary = llm.invoke(prompt).content
        else:
            loc = result.get("location") or location
            u = result.get("units")
            ts = result.get("timestamp")
            main = result.get("weather", {})
            conditions = ", ".join(result.get("conditions") or [])
            wind = result.get("wind", {})
            wind_speed = wind.get("speed")
            # Keep concise, conversational style
            summary = (
                f"It's {main.get('temp')}° (feels {main.get('feels_like')}°) in {loc}. "
                f"Humidity {main.get('humidity')}%, pressure {main.get('pressure')} hPa, "
                f"wind {wind_speed} m/s. Conditions: {conditions or 'n/a'}."
            )
        # Update memory
        mem.update_weather_context(
            location=result.get("location"),
            units=result.get("units"),
            lang=lang,
            timestamp=result.get("timestamp"),
        )
        logger.debug("Weather node succeeded for %s", result.get("location"))
        return {
            "answer": summary,
            "tool": "weather",
            "weather": result,
        }

    def do_rag(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        chat_history = state.get("chat_history")
        logger.info("RAG node: question=%s history_len=%s", question, 0 if not chat_history else len(chat_history))
        try:
            result = rag_answer(llm, vectorstore, question, k=rag_top_k, chat_history=chat_history)
        except Exception:
            logger.exception("RAG failed (vectorstore or LLM error)")
            result = {"answer": "RAG unavailable (vector DB offline). Please add docs and ensure Qdrant is running.", "sources": []}
        logger.debug("RAG node: sources=%s", len(result.get("sources", [])))
        return {
            "answer": result["answer"],
            "tool": "pdf_rag",
            "sources": result.get("sources", []),
        }

    # History node removed to keep behavior general and avoid pattern-specific routing.

    graph = StateGraph(dict)
    graph.add_node("router", RunnableLambda(router).with_config(run_name="router", tags=["agent-router"]))
    graph.add_node("weather", RunnableLambda(do_weather).with_config(run_name="weather_node", tags=["tool:weather"]))
    graph.add_node("rag", RunnableLambda(do_rag).with_config(run_name="rag_node", tags=["tool:rag"]))
    # No history node

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s.get("route"), {"weather": "weather", "rag": "rag"})
    graph.add_edge("weather", END)
    graph.add_edge("rag", END)
    # no history edge

    return graph.compile()
    