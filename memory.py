from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentMemory:
    """Ephemeral multi-turn memory persisted in the UI session.

    This stores the last known user preferences and context that can be reused
    across turns to handle ellipsis like "and what about tomorrow?".
    """

    last_location: Optional[str] = None
    last_units: Optional[str] = None
    last_lang: Optional[str] = None
    last_weather_timestamp: Optional[int] = None
    pending_weather_question: bool = False

    def update_weather_context(
        self,
        location: Optional[str] = None,
        units: Optional[str] = None,
        lang: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        if location:
            self.last_location = location
        if units:
            self.last_units = units
        if lang:
            self.last_lang = lang
        if timestamp is not None:
            self.last_weather_timestamp = timestamp

    def set_pending_weather_question(self, pending: bool) -> None:
        self.pending_weather_question = pending


