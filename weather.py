import time
import re
from typing import Any, Dict, Optional, List
from typing import Literal

from pydantic import BaseModel, Field

import httpx

from .config import AppConfig


class WeatherClient:
    def __init__(self, config: AppConfig):
        self._config = config
        self._base_url = config.openweather_base_url.rstrip("/")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timeout = httpx.Timeout(timeout=config.http_timeout_seconds)

    @staticmethod
    def _cache_key(location: str, units: str, lang: Optional[str]) -> str:
        lang_part = lang or ""
        return f"{location}|{units}|{lang_part}"

    def _get_cached(self, location: str, units: str, lang: Optional[str]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(location, units, lang)
        item = self._cache.get(key)
        if not item:
            return None
        if time.time() > item["expiry_ts"]:
            self._cache.pop(key, None)
            return None
        return item["data"]

    def _set_cache(self, location: str, units: str, lang: Optional[str], data: Dict[str, Any]) -> None:
        key = self._cache_key(location, units, lang)
        self._cache[key] = {
            "expiry_ts": time.time() + self._config.weather_cache_ttl_seconds,
            "data": data,
        }

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        max_attempts = max(1, int(self._config.http_max_retries))
        delay = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError:
                if attempt >= max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 6)

    def _normalize_weather(self, data: Dict[str, Any], fallback_location: str, resolved_units: str, lang: Optional[str]) -> Dict[str, Any]:
        return {
            "location": f"{data.get('name', fallback_location)}, {data.get('sys', {}).get('country', '')}",
            "units": resolved_units,
            "timestamp": data.get("dt"),
            "fetched_at": int(time.time()),
            "weather": {
                "temp": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
            },
            "wind": data.get("wind", {}),
            "clouds": data.get("clouds", {}),
            "conditions": [w.get("description") for w in (data.get("weather") or [])],
            "owm_raw": data,
        }

    def geocode_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        if not self._config.openweather_api_key:
            return []
        url = f"{self._base_url}/../../geo/1.0/direct"  # resolve to api.openweathermap.org/geo/1.0
        params = {"q": query, "limit": limit, "appid": self._config.openweather_api_key}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return []
                return data
        except httpx.HTTPError:
            # Gracefully degrade to no candidates to enable clarification path
            return []

    def fetch_current_weather_by_coords(self, lat: float, lon: float, units: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        resolved_units = units or self._config.weather_units
        params: Dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "appid": self._config.openweather_api_key,
            "units": resolved_units,
        }
        if lang:
            params["lang"] = lang
        data = self._get("weather", params)
        # Best-effort location string
        loc = f"{data.get('name') or ''}, {data.get('sys', {}).get('country', '')}".strip(', ')
        normalized = self._normalize_weather(data, loc or f"{lat},{lon}", resolved_units, lang)
        return normalized

    def fetch_current_weather(self, location: str, units: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        if not self._config.openweather_api_key:
            raise ValueError("OPENWEATHER_API_KEY missing.")
        resolved_units = units or self._config.weather_units
        cached = self._get_cached(location, resolved_units, lang)
        if cached:
            return cached
        params = {
            "q": location,
            "appid": self._config.openweather_api_key,
            "units": resolved_units,
        }
        if lang:
            params["lang"] = lang
        try:
            data = self._get("weather", params)
            normalized = self._normalize_weather(data, location, resolved_units, lang)
            self._set_cache(location, resolved_units, lang, normalized)
            return normalized
        except httpx.HTTPStatusError as e:  # type: ignore[attr-defined]
            status = getattr(e, "response", None).status_code if hasattr(e, "response") and e.response is not None else None
            if status == 404:
                # Try geocoding suggestions
                candidates = self.geocode_search(location, limit=3)
                if not candidates:
                    raise CityNotFoundError(location, [])
                # If exactly one good candidate, try by coordinates immediately
                if len(candidates) == 1:
                    cand = candidates[0]
                    lat = float(cand.get("lat"))
                    lon = float(cand.get("lon"))
                    result = self.fetch_current_weather_by_coords(lat, lon, units=resolved_units, lang=lang)
                    # Cache under the original query as well
                    self._set_cache(location, resolved_units, lang, result)
                    return result
                # Multiple candidates -> let caller clarify
                raise CityNotFoundError(location, candidates)
            raise

    @staticmethod
    def parse_location_from_text(text: str) -> Optional[str]:
        # Primary: phrases like "in Paris", "at New York", "for Berlin"
        match = re.search(r"(?:in|at|for)\s+([A-Za-z\-\s,]{2,})", text, flags=re.IGNORECASE)
        candidate = None
        if match:
            candidate = match.group(1).strip().strip(",")
        else:
            # Fallback: if the entire text looks like a city name (e.g., just "Paris"), accept it
            stripped = (text or "").strip().strip(",")
            # Exclude deictic/pronoun/generic words that are not city names
            generic = {"here", "there", "this", "that", "it", "where", "city", "place"}
            if stripped.lower() in generic:
                return None
            if re.fullmatch(r"[A-Za-z][A-Za-z\-\s,]{1,}$", stripped):
                candidate = stripped
            else:
                return None
        # Split at secondary prepositions/temporal phrases (e.g., "in October", "on Monday")
        temporal_words = {
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "today", "tomorrow", "tonight", "yesterday", "now", "week", "month", "year",
        }
        stop_words = {"in", "on", "by", "this", "next", "coming", "during"}
        tokens = [t for t in re.split(r"\s+", candidate) if t]
        cleaned_tokens = []
        for tok in tokens:
            low = tok.lower().strip(",")
            if low in stop_words or low in temporal_words:
                break
            cleaned_tokens.append(tok)
        cleaned = " ".join(cleaned_tokens).strip().strip(",")
        return cleaned or None

class CityNotFoundError(Exception):
    def __init__(self, query: str, candidates: Optional[List[Dict[str, Any]]] = None):
        super().__init__(query)
        self.query = query
        self.candidates = candidates or []


class WeatherRequest(BaseModel):
    """Pydantic schema for OpenWeather Current Weather API parameters.

    Note: For Current Weather endpoint we use 'q' (city name), optional 'units', optional 'lang'.
    """
    q: str = Field(..., description="City name for OpenWeather 'q' parameter (e.g., 'Pune' or 'Pune,IN')")
    units: Literal["standard", "metric", "imperial"] = Field("metric", description="Units system")
    lang: Optional[str] = Field(None, description="Two-letter language code, e.g., 'en'")


def llm_to_weather_request(llm, question: str, default_units: str = "metric") -> WeatherRequest:
    """Use the LLM to extract a WeatherRequest from natural language via structured output.

    The model should return only the location in 'q' (no dates/temporal phrases),
    and choose units if explicitly asked; otherwise use default_units.
    """
    # Some models support direct structured output via LangChain
    structured_llm = llm.with_structured_output(WeatherRequest)
    prompt = (
        "Extract OpenWeather current weather request parameters from the user question.\n"
        "- q: city name only (optionally include country code), no dates or temporal info.\n"
        f"- units: one of standard/metric/imperial; default to '{default_units}'.\n"
        "- lang: two-letter code if clearly specified; otherwise omit.\n\n"
        f"Question: {question}\n"
    )
    return structured_llm.invoke(prompt)

