from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.challenges import (
    BUNNY_HOLD_SEC,
    PILL_HOLD_SEC,
    Challenge,
    HandObservation,
    HoldTracker,
    PoseObservation,
    ScanTracker,
    SpoonTracker,
    build_breizhcamp_campaign,
    bunny_ears_active,
    dodge_matches,
    pill_hover,
)

ATTRACT = "ATTRACT"
COUNTDOWN = "COUNTDOWN"
IN_ROUND = "IN_ROUND"
SCORE = "SCORE"

# Scoring : 100 pts par figure reussie + bonus vitesse de 10 pts par seconde
# restante au moment de la validation.
FIGURE_POINTS = 100
TIME_BONUS_PER_SEC = 10

# 2 paumes ouvertes maintenues 1 s pour (re)lancer une partie : zero
# calibration, demarre de loin, et tres peu de faux positifs.
START_HOLD_SEC = 1.0
# Pause entre deux figures : le temps d'afficher le flash de succes/echec.
ROUND_TRANSITION_SEC = 1.2
FLASH_DURATION_SEC = 1.2

TIMEOUT_MESSAGE = "TOO SLOW..."


@dataclass(slots=True)
class RoundResult:
    title: str
    success: bool
    points: int


@dataclass(slots=True)
class GameState:
    phase: str = ATTRACT
    score: int = 0
    round_index: int = 0
    round_deadline: float = 0.0
    round_transition_at: float = 0.0
    countdown_until: float = 0.0
    flash_message: str = ""
    flash_until: float = 0.0
    chosen_pill: str | None = None
    start_hold_since: float | None = None
    score_published: bool = False
    round_results: list[RoundResult] = field(default_factory=list)


@dataclass(slots=True)
class GameEvent:
    phase: str = ATTRACT
    score: int = 0
    round_index: int = 0  # 1-based pour l'affichage
    round_total: int = 0
    timer_left: float = 0.0
    round_duration: float = 0.0
    challenge_key: str = ""
    challenge_title: str = ""
    challenge_prompt: str = ""
    challenge_kind: str = ""
    challenge_background_asset: str = ""
    flash_message: str = ""
    countdown_value: int = 0
    figure_progress: float = 0.0  # maintien pilule/bunny, rotation cuillere, scan
    pill_hover: str | None = None
    spoon_anchor: tuple[float, float] | None = None
    spoon_angle: float = 0.0
    start_hold_progress: float = 0.0
    chosen_pill: str | None = None
    rain_boost: bool = False
    round_results: list[RoundResult] = field(default_factory=list)
    published: bool = False


class GameEngine:
    """State machine du mini-jeu : ATTRACT -> COUNTDOWN -> IN_ROUND -> SCORE.

    Une figure ratee (timeout) passe simplement a la suivante avec 0 point :
    la partie se termine toujours sur l'ecran de score.
    """

    def __init__(
        self,
        round_duration: float = 8.0,
        countdown_duration: float = 3.0,
        victory_pill: str = "auto",
        campaign: list[Challenge] | None = None,
        rng: random.Random | None = None,
        state: GameState | None = None,
    ) -> None:
        self.state = state or GameState()
        self.round_duration = round_duration
        self.countdown_duration = countdown_duration
        self.victory_pill = victory_pill
        self._rng = rng or random.Random()
        self._fixed_campaign = campaign
        self.challenges: list[Challenge] = list(campaign) if campaign else build_breizhcamp_campaign(self._rng)

        self._pill_hold = HoldTracker(PILL_HOLD_SEC)
        self._bunny_hold = HoldTracker(BUNNY_HOLD_SEC)
        self._spoon = SpoonTracker()
        self._scan = ScanTracker()
        self._figure_progress = 0.0
        self._pill_hover: str | None = None
        self._now = 0.0

    def current_challenge(self) -> Challenge | None:
        if 0 <= self.state.round_index < len(self.challenges):
            return self.challenges[self.state.round_index]
        return None

    def start_game(self, now: float) -> None:
        if self._fixed_campaign is None:
            self.challenges = build_breizhcamp_campaign(self._rng)
        state = self.state
        state.phase = COUNTDOWN
        state.countdown_until = now + self.countdown_duration
        state.score = 0
        state.round_index = 0
        state.round_results = []
        state.chosen_pill = None
        state.score_published = False
        state.flash_message = ""
        state.flash_until = 0.0
        state.start_hold_since = None

    def update(
        self,
        hands: Sequence[HandObservation],
        pose: PoseObservation | None,
        now: float,
        publish_pill: Callable[[str], bool],
    ) -> GameEvent:
        self._now = now
        state = self.state

        if state.phase in (ATTRACT, SCORE):
            self._update_start_hold(hands, now)
            return self._snapshot()

        if state.phase == COUNTDOWN:
            if now >= state.countdown_until:
                self._begin_round(now)
            return self._snapshot()

        # IN_ROUND
        challenge = self.current_challenge()
        if challenge is None:  # garde-fou, ne doit pas arriver
            state.phase = SCORE
            return self._snapshot()

        if now < state.round_transition_at:
            return self._snapshot()

        if now >= state.round_deadline:
            state.round_results.append(RoundResult(challenge.title, False, 0))
            self._set_flash(TIMEOUT_MESSAGE, now)
            published = self._advance(now, publish_pill)
            return self._snapshot(published=published)

        success, message = self._detect_figure(challenge, hands, pose, now)
        if success:
            points = FIGURE_POINTS + int(max(0.0, state.round_deadline - now)) * TIME_BONUS_PER_SEC
            state.score += points
            state.round_results.append(RoundResult(challenge.title, True, points))
            self._set_flash(message, now)
            published = self._advance(now, publish_pill)
            return self._snapshot(published=published)

        return self._snapshot()

    # -- internals ----------------------------------------------------------

    def _update_start_hold(self, hands: Sequence[HandObservation], now: float) -> None:
        state = self.state
        open_palms = sum(hand.gesture == "open_palm" for hand in hands)
        if open_palms >= 2:
            if state.start_hold_since is None:
                state.start_hold_since = now
            if now - state.start_hold_since >= START_HOLD_SEC:
                self.start_game(now)
        else:
            state.start_hold_since = None

    def _begin_round(self, now: float) -> None:
        state = self.state
        state.phase = IN_ROUND
        state.round_transition_at = now
        state.round_deadline = now + self.round_duration
        self._reset_round_trackers()

    def _reset_round_trackers(self) -> None:
        self._pill_hold.reset()
        self._bunny_hold.reset()
        self._spoon.reset()
        self._scan.reset()
        self._figure_progress = 0.0
        self._pill_hover = None

    def _detect_figure(
        self,
        challenge: Challenge,
        hands: Sequence[HandObservation],
        pose: PoseObservation | None,
        now: float,
    ) -> tuple[bool, str]:
        state = self.state
        message = challenge.success_message
        self._figure_progress = 0.0
        self._pill_hover = None

        if challenge.kind == "dodge":
            return dodge_matches(pose), message

        if challenge.kind == "pill":
            hover = pill_hover(hands)
            self._pill_hover = hover
            validated, progress = self._pill_hold.update(hover, now)
            self._figure_progress = progress
            if validated:
                state.chosen_pill = validated
                return True, f"{validated.upper()} PILL ACCEPTED"
            return False, message

        if challenge.kind == "bunny":
            active = bunny_ears_active(hands, pose)
            validated, progress = self._bunny_hold.update("bunny" if active else None, now)
            self._figure_progress = progress
            return validated is not None, message

        if challenge.kind == "spoon":
            progress = self._spoon.update(hands)
            self._figure_progress = progress
            return progress >= 1.0, message

        if challenge.kind == "scan":
            progress = self._scan.update(pose, now)
            self._figure_progress = progress
            return progress >= 1.0, message

        return False, message

    def _advance(self, now: float, publish_pill: Callable[[str], bool]) -> bool:
        state = self.state
        state.round_index += 1
        if state.round_index >= len(self.challenges):
            state.phase = SCORE
            state.start_hold_since = None
            if not state.score_published:
                state.score_published = True
                pill = state.chosen_pill or (self.victory_pill if self.victory_pill != "auto" else "blue")
                return publish_pill(pill)
            return False

        state.round_transition_at = now + ROUND_TRANSITION_SEC
        state.round_deadline = state.round_transition_at + self.round_duration
        self._reset_round_trackers()
        return False

    def _set_flash(self, message: str, now: float) -> None:
        self.state.flash_message = message
        self.state.flash_until = now + FLASH_DURATION_SEC

    def _snapshot(self, published: bool = False) -> GameEvent:
        state = self.state
        now = self._now

        challenge = self.current_challenge() if state.phase == IN_ROUND else None
        timer_left = 0.0
        if state.phase == IN_ROUND:
            timer_left = min(self.round_duration, max(0.0, state.round_deadline - now))

        if state.phase == IN_ROUND:
            display_round = state.round_index + 1
        elif state.phase == SCORE:
            display_round = len(self.challenges)
        else:
            display_round = 0

        flash = state.flash_message if now < state.flash_until else ""
        countdown_value = max(0, math.ceil(state.countdown_until - now)) if state.phase == COUNTDOWN else 0
        start_hold_progress = 0.0
        if state.start_hold_since is not None:
            start_hold_progress = min(1.0, (now - state.start_hold_since) / START_HOLD_SEC)

        return GameEvent(
            phase=state.phase,
            score=state.score,
            round_index=display_round,
            round_total=len(self.challenges),
            timer_left=timer_left,
            round_duration=self.round_duration,
            challenge_key=challenge.key if challenge else "",
            challenge_title=challenge.title if challenge else "",
            challenge_prompt=challenge.prompt if challenge else "",
            challenge_kind=challenge.kind if challenge else "",
            challenge_background_asset=challenge.background_asset if challenge else "",
            flash_message=flash,
            countdown_value=countdown_value,
            figure_progress=self._figure_progress,
            pill_hover=self._pill_hover,
            spoon_anchor=self._spoon.anchor,
            spoon_angle=self._spoon.angle,
            start_hold_progress=start_hold_progress,
            chosen_pill=state.chosen_pill,
            rain_boost=state.phase in (ATTRACT, SCORE) or bool(flash),
            round_results=list(state.round_results),
            published=published,
        )
