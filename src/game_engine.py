from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from src.challenges import Challenge, build_campaign, challenge_matches


IDLE = "IDLE"
IN_ROUND = "IN_ROUND"
VICTORY = "VICTORY"
GAME_OVER = "GAME_OVER"


@dataclass(slots=True)
class GameState:
    phase: str = IDLE
    status: str = "SYSTEM ONLINE"
    status_until: float = 0.0
    last_action_ts: float = 0.0
    round_index: int = 0
    round_started_at: float = 0.0
    round_deadline: float = 0.0
    round_transition_at: float = 0.0
    start_hold_since: float | None = None
    reset_hold_since: float | None = None
    rain_boost: bool = False
    victory_mqtt_sent: bool = False
    game_over_reason: str = ""


@dataclass(slots=True)
class GameEvent:
    status: str
    phase: str
    round_index: int
    round_total: int
    rain_boost: bool
    action_taken: bool
    pill: str | None = None
    published: bool = False
    timer_left: float = 0.0
    challenge_title: str = ""
    challenge_prompt: str = ""
    victory_pill: str = "blue"
    challenge_background_asset: str = ""


@dataclass(slots=True)
class CampaignController:
    state: GameState
    challenges: list[Challenge]
    start_hold_duration: float = 1.0
    reset_hold_duration: float = 1.0
    transition_lock_duration: float = 0.35


class GameEngine:
    def __init__(
        self,
        sequence_length: int | GameState = 5,
        state: GameState | None = None,
        status_duration: float = 2.0,
        victory_pill: str = "auto",
    ) -> None:
        sequence_length_value: int
        if isinstance(sequence_length, GameState) and state is None:
            state = sequence_length
            sequence_length_value = 5
        else:
            sequence_length_value = sequence_length if isinstance(sequence_length, int) else 5

        self.state = state or GameState()
        self.status_duration = status_duration
        self.challenges = build_campaign(sequence_length_value)
        self.victory_pill = victory_pill
        self.start_hold_duration = 1.0
        self.reset_hold_duration = 1.0
        self.transition_lock_duration = 0.35
        self._now = 0.0

    def current_challenge(self) -> Challenge | None:
        if 0 <= self.state.round_index < len(self.challenges):
            return self.challenges[self.state.round_index]
        return None

    def start_campaign(self, now: float) -> None:
        self.state.phase = IN_ROUND
        self.state.round_index = 0
        self.state.round_started_at = now
        self.state.round_deadline = now + self.challenges[0].duration_sec if self.challenges else now
        self.state.round_transition_at = now
        self.state.status = "SYSTEM ONLINE"
        self.state.status_until = now + self.status_duration
        self.state.game_over_reason = ""
        self.state.victory_mqtt_sent = False

    def restart_campaign(self, now: float) -> None:
        self.start_campaign(now)

    def _finalize_victory(self, now: float, publish_pill: Callable[[str], bool], challenge: Challenge) -> GameEvent:
        pill = challenge.victory_pill if self.victory_pill == "auto" else self.victory_pill
        published = False
        if not self.state.victory_mqtt_sent:
            published = publish_pill(pill)
            self.state.victory_mqtt_sent = True
        self.state.phase = VICTORY
        self.state.status = "VICTOIRE"
        self.state.status_until = now + self.status_duration
        return self._snapshot(challenge=challenge, action_taken=True, pill=pill, published=published)

    def _snapshot(
        self,
        challenge: Challenge | None = None,
        action_taken: bool = False,
        pill: str | None = None,
        published: bool = False,
    ) -> GameEvent:
        active_challenge = challenge or self.current_challenge()
        if active_challenge is None and self.challenges:
            active_challenge = self.challenges[min(self.state.round_index, len(self.challenges) - 1)]

        timer_left = max(0.0, self.state.round_deadline - self._now) if self.state.phase == IN_ROUND else 0.0
        display_round = 0
        if self.state.phase == IN_ROUND:
            display_round = self.state.round_index + 1
        elif self.state.phase in (VICTORY, GAME_OVER):
            display_round = len(self.challenges)
        return GameEvent(
            status=self.state.status,
            phase=self.state.phase,
            round_index=display_round,
            round_total=len(self.challenges),
            rain_boost=self.state.rain_boost,
            action_taken=action_taken,
            pill=pill,
            published=published,
            timer_left=timer_left,
            challenge_title=active_challenge.title if active_challenge else "",
            challenge_prompt=active_challenge.prompt if active_challenge else "",
            victory_pill=active_challenge.victory_pill if active_challenge else "blue",
            challenge_background_asset=active_challenge.background_asset if active_challenge else "",
        )

    def tick(self, now: float) -> None:
        if self.state.status and now > self.state.status_until:
            self.state.status = ""

    def update(self, hand_gestures: Sequence[str], now: float, publish_pill: Callable[[str], bool]) -> GameEvent:
        self._now = now
        self.tick(now)

        open_palm_count = sum(gesture == "open_palm" for gesture in hand_gestures)
        if self.state.phase == IDLE:
            if open_palm_count >= 2:
                if self.state.start_hold_since is None:
                    self.state.start_hold_since = now
                if now - self.state.start_hold_since >= self.start_hold_duration:
                    self.start_campaign(now)
                    return self._snapshot(action_taken=True)
            else:
                self.state.start_hold_since = None
            return self._snapshot()

        if self.state.phase in (VICTORY, GAME_OVER):
            if open_palm_count >= 2:
                if self.state.reset_hold_since is None:
                    self.state.reset_hold_since = now
                if now - self.state.reset_hold_since >= self.reset_hold_duration:
                    self.restart_campaign(now)
                    self.state.start_hold_since = now
                    return self._snapshot(action_taken=True)
            else:
                self.state.reset_hold_since = None
            return self._snapshot()

        challenge = self.current_challenge()
        if challenge is None:
            self.state.phase = GAME_OVER
            self.state.status = "CAMPAIGN INVALIDE"
            self.state.status_until = now + self.status_duration
            self.state.game_over_reason = "CAMPAIGN INVALIDE"
            return self._snapshot()

        if self.state.round_transition_at and now < self.state.round_transition_at:
            return self._snapshot()

        if now >= self.state.round_deadline:
            self.state.phase = GAME_OVER
            self.state.status = "TEMPS ECOULE"
            self.state.status_until = now + self.status_duration
            self.state.game_over_reason = "TEMPS ECOULE"
            return self._snapshot()

        if challenge_matches(challenge, hand_gestures):
            self.state.round_index += 1
            self.state.round_transition_at = now + 0.35
            if self.state.round_index >= len(self.challenges):
                return self._finalize_victory(now, publish_pill, challenge)

            next_challenge = self.current_challenge()
            self.state.round_started_at = now
            self.state.round_deadline = now + (next_challenge.duration_sec if next_challenge else 0.0)
            self.state.status = f"ROUND {self.state.round_index}/{len(self.challenges)}"
            self.state.status_until = now + self.status_duration
            return self._snapshot(challenge=challenge, action_taken=True)

        return self._snapshot()

    def process_gesture(self, gesture: str, now: float, publish_pill: Callable[[str], bool]) -> GameEvent:
        return self.update([gesture], now, publish_pill)

    def _set_status(self, status: str, now: float) -> None:
        self.state.status = status
        self.state.status_until = now + self.status_duration
