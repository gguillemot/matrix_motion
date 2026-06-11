from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class GameState:
    status: str = "SYSTEM ONLINE"
    status_until: float = 0.0
    last_action_ts: float = 0.0
    rain_boost: bool = False


@dataclass(slots=True)
class GameEvent:
    status: str
    rain_boost: bool
    action_taken: bool
    pill: str | None = None
    published: bool = False


class GameEngine:
    def __init__(self, state: GameState | None = None, action_cooldown: float = 2.5, status_duration: float = 2.0) -> None:
        self.state = state or GameState()
        self.action_cooldown = action_cooldown
        self.status_duration = status_duration

    def tick(self, now: float) -> None:
        if self.state.status and now > self.state.status_until:
            self.state.status = ""

    def process_gesture(self, gesture: str, now: float, publish_pill: Callable[[str], bool]) -> GameEvent:
        self.tick(now)

        if gesture == "none":
            return GameEvent(status=self.state.status, rain_boost=self.state.rain_boost, action_taken=False)

        if self.state.last_action_ts and now - self.state.last_action_ts <= self.action_cooldown:
            return GameEvent(status=self.state.status, rain_boost=self.state.rain_boost, action_taken=False)

        pill: str | None = None
        published = False

        if gesture == "thumbs_up":
            pill = "blue"
            published = publish_pill(pill)
            self._set_status("PILULE BLEUE" if published else "THUMBS UP DETECTED", now)
        elif gesture == "ok_sign":
            pill = "red"
            published = publish_pill(pill)
            self._set_status("PILULE ROUGE" if published else "OK SIGN DETECTED", now)
        elif gesture == "open_palm":
            self.state.rain_boost = not self.state.rain_boost
            self._set_status("MATRIX BOOST ON" if self.state.rain_boost else "MATRIX BOOST OFF", now)
        else:
            return GameEvent(status=self.state.status, rain_boost=self.state.rain_boost, action_taken=False)

        self.state.last_action_ts = now
        return GameEvent(
            status=self.state.status,
            rain_boost=self.state.rain_boost,
            action_taken=True,
            pill=pill,
            published=published,
        )

    def _set_status(self, status: str, now: float) -> None:
        self.state.status = status
        self.state.status_until = now + self.status_duration
