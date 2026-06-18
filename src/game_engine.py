from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.config import (
    BULLET_STOP_VIDEO_SEC,
    BULLET_TIME_REPLAY_SEC,
    QUIZ_DWELL_SEC,
    QUIZ_ENABLED,
    QUIZ_GRACE_SEC,
    QUIZ_INTRO_SEC,
    QUIZ_MAX_PER_GAME,
    QUIZ_QUESTIONS_PATH,
    QUIZ_REVEAL_SEC,
    QUIZ_ZONE_MARGIN,
    TRINITY_FREEZE_HOLD_SEC,
    TRINITY_VIDEO_END,
    TRINITY_VIDEO_START,
)
from src.challenges import (
    BULLET_STOP_GRACE_SEC,
    BULLET_STOP_HOLD_SEC,
    BUNNY_HOLD_SEC,
    DODGE_HOLD_SEC,
    PILL_HOLD_SEC,
    Challenge,
    HandObservation,
    HoldTracker,
    PoseObservation,
    QuizRuntime,
    SpoonTracker,
    build_breizhcamp_campaign,
    bullet_stop_palm,
    bunny_ears_active,
    dodge_matches,
    load_quiz_questions,
    pill_hover,
)

ATTRACT = "ATTRACT"
INTRO = "INTRO"
PILL_CHOICE = "PILL_CHOICE"
BLUE_ENDING = "BLUE_ENDING"
COUNTDOWN = "COUNTDOWN"
IN_ROUND = "IN_ROUND"
SCORE = "SCORE"
BADGE_SCAN = "BADGE_SCAN"
TRINITY_OUTRO = "TRINITY_OUTRO"

# Outro "Trinity" : l'ecran de score reste visible SCORE_HOLD_SEC APRES la fin de
# la celebration de la derniere figure (sinon, une figure finale a longue
# celebration -- ex. le clip bullet_stop de 10 s -- recouvrirait tout l'ecran de
# score). Ensuite on enchaine sur le clip Trinity (gele + texte) pour
# TRINITY_OUTRO_DURATION avant le retour auto a l'attract (ESPACE court-circuite).
SCORE_HOLD_SEC = 3.0
BADGE_SCAN_SEC = 30.0  # durée de l'écran de scan badge entre le score et l'outro
TRINITY_OUTRO_DURATION = (
    TRINITY_VIDEO_END - TRINITY_VIDEO_START
) + TRINITY_FREEZE_HOLD_SEC

# Le choix des pilules et la fin "pilule bleue" n'ont pas de timer de jeu,
# mais un timeout d'inactivite ramene a l'attract si le passant est parti.
PILL_CHOICE_IDLE_SEC = 30.0
BLUE_ENDING_IDLE_SEC = 25.0

# Scoring : points = (base + bonus vitesse) x multiplicateur de combo.
# Le bonus vitesse est lineaire sur le temps restant (max 50) ; le combo
# compte les succes consecutifs (x1 -> x5, un echec le casse). Une partie
# moyenne vaut ~150-400 pts, une partie parfaite et rapide ~1400 pts :
# vraie marge de progression pour donner envie de rejouer.
BASE_POINTS = 50
MAX_SPEED_BONUS = 50
MAX_COMBO = 5

# 2 paumes ouvertes maintenues 1 s pour (re)lancer une partie : zero
# calibration, demarre de loin, et tres peu de faux positifs.
START_HOLD_SEC = 1.0
# Passage d'une figure a la suivante, en deux temps pour eviter tout overlap :
#  1. CELEBRATION_SEC : l'effet de recompense de la figure reussie joue SEUL ;
#  2. READY_SEC : un ecran "prepare-toi" annonce la figure suivante (titre +
#     consigne), chrono fige, avant que la detection ne demarre.
# Les figures a celebration speciale remplacent l'etape 1 par leur duree propre
# (replay bullet-time, clip video) mais gardent le meme beat "prepare-toi".
CELEBRATION_SEC = 3.0
READY_SEC = 2.0
# Pause par defaut sur timeout (echec) : pas de celebration, juste le message
# "TOO SLOW" puis l'annonce de la figure suivante.
ROUND_TRANSITION_SEC = 3.5
FLASH_DURATION_SEC = 2.2

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
    phase_entered_at: float = 0.0
    combo: int = 0
    celebration_key: str = ""
    celebration_started_at: float = 0.0
    celebration_until: float = 0.0
    new_record: bool = False
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
    figure_active: bool = False  # False pendant l'ecran "prepare-toi" (chrono fige)
    round_starts_in: float = 0.0  # compte a rebours avant l'activation de la figure
    flash_message: str = ""
    countdown_value: int = 0
    figure_progress: float = 0.0  # maintien pilule/bunny, rotation cuillere, scan
    pill_hover: str | None = None
    spoon_anchor: tuple[float, float] | None = None
    spoon_angle: float = 0.0
    palm_anchor: tuple[float, float] | None = None
    start_hold_progress: float = 0.0
    chosen_pill: str | None = None
    rain_boost: bool = False
    combo: int = 0
    celebration_key: str = ""
    celebration_progress: float = 0.0  # 0 -> 1 sur la fenetre de celebration
    best_score: int = 0
    new_record: bool = False
    round_results: list[RoundResult] = field(default_factory=list)
    badge_scan_time_left: float = 0.0
    # -- Quiz Matrix (defi QCM) --------------------------------------------
    quiz_active: bool = False
    quiz_substate: str = ""
    quiz_question: str = ""
    quiz_answers: list[str] = field(default_factory=list)
    quiz_zones: list[tuple[float, float, float, float]] = field(default_factory=list)
    quiz_cursor: tuple[float, float] | None = None
    quiz_dwell: list[float] = field(default_factory=list)
    quiz_selected_index: int | None = None
    quiz_correct_index: int = -1
    quiz_result: str = ""
    quiz_explanation: str = ""
    quiz_timer_left: float = 0.0
    quiz_timer_total: float = 0.0
    quiz_hand_detected: bool = False


class GameEngine:
    """State machine du mini-jeu.

    ATTRACT -> INTRO (clip video, si disponible) -> PILL_CHOICE, puis :
    - pilule bleue -> BLUE_ENDING (camera normale, invite a rejouer) ;
    - pilule rouge -> COUNTDOWN -> IN_ROUND x4 -> SCORE.

    Une figure ratee (timeout) passe simplement a la suivante avec 0 point :
    le parcours rouge se termine toujours sur l'ecran de score.
    """

    def __init__(
        self,
        round_duration: float = 8.0,
        countdown_duration: float = 3.0,
        campaign: list[Challenge] | None = None,
        rng: random.Random | None = None,
        state: GameState | None = None,
        best_score_path: str | Path | None = None,
        has_intro: bool = False,
    ) -> None:
        self.has_intro = has_intro
        self.state = state or GameState()
        self.round_duration = round_duration
        self.countdown_duration = countdown_duration
        self._rng = rng or random.Random()
        self._fixed_campaign = campaign
        self._quiz_questions = (
            load_quiz_questions(QUIZ_QUESTIONS_PATH) if QUIZ_ENABLED else []
        )
        self.challenges: list[Challenge] = (
            list(campaign) if campaign else self._build_campaign()
        )
        self._best_score_path = Path(best_score_path) if best_score_path else None
        self.best_score = self._load_best_score()

        self._pill_hold = HoldTracker(PILL_HOLD_SEC)
        self._bunny_hold = HoldTracker(BUNNY_HOLD_SEC)
        self._dodge_hold = HoldTracker(DODGE_HOLD_SEC)
        self._spoon = SpoonTracker()
        self._bullet_stop = HoldTracker(BULLET_STOP_HOLD_SEC)
        self._quiz: QuizRuntime | None = None
        self._palm_anchor: tuple[float, float] | None = None
        self._figure_progress = 0.0
        self._pill_hover: str | None = None
        self._now = 0.0

    def _build_campaign(self) -> list[Challenge]:
        """Campagne par defaut : figures + quiz intercales (alternance stricte)."""
        return build_breizhcamp_campaign(
            self._rng,
            quiz_questions=self._quiz_questions or None,
            max_quiz=QUIZ_MAX_PER_GAME,
        )

    def current_challenge(self) -> Challenge | None:
        if 0 <= self.state.round_index < len(self.challenges):
            return self.challenges[self.state.round_index]
        return None

    def start_game(self, now: float) -> None:
        if self._fixed_campaign is None:
            self.challenges = self._build_campaign()
        state = self.state
        state.phase = INTRO if self.has_intro else PILL_CHOICE
        state.phase_entered_at = now
        self._pill_hold.reset()
        state.score = 0
        state.round_index = 0
        state.round_results = []
        state.chosen_pill = None
        state.flash_message = ""
        state.flash_until = 0.0
        state.start_hold_since = None
        state.combo = 0
        state.celebration_key = ""
        state.celebration_until = 0.0
        state.new_record = False

    def finish_intro(self, now: float) -> None:
        """Appele par la boucle principale quand le clip d'intro se termine
        ou est passe (touche ESPACE)."""
        if self.state.phase == INTRO:
            self.state.phase = PILL_CHOICE
            self.state.phase_entered_at = now
            self._pill_hold.reset()

    def skip_outro(self, now: float) -> None:
        """Appele par la boucle principale (touche ESPACE) pour court-circuiter
        l'outro Trinity et revenir tout de suite a l'attract."""
        if self.state.phase == TRINITY_OUTRO:
            self._go_attract(now)

    def _go_attract(self, now: float) -> None:
        self.state.phase = ATTRACT
        self.state.phase_entered_at = now
        self.state.start_hold_since = None

    def update(
        self,
        hands: Sequence[HandObservation],
        pose: PoseObservation | None,
        now: float,
    ) -> GameEvent:
        self._now = now
        state = self.state

        if state.phase in (ATTRACT, BLUE_ENDING):
            if (
                state.phase == BLUE_ENDING
                and now - state.phase_entered_at >= BLUE_ENDING_IDLE_SEC
            ):
                self._go_attract(now)
                return self._snapshot()
            self._update_start_hold(hands, now)
            return self._snapshot()

        if state.phase == SCORE:
            # Rejouer reste possible pendant tout l'affichage des scores. L'ecran
            # badge_scan ne demarre que SCORE_HOLD_SEC apres la fin de la celebration.
            self._update_start_hold(hands, now)
            score_visible_since = max(state.phase_entered_at, state.celebration_until)
            if state.phase == SCORE and now - score_visible_since >= SCORE_HOLD_SEC:
                state.phase = BADGE_SCAN
                state.phase_entered_at = now
                state.start_hold_since = None
            return self._snapshot()

        if state.phase == BADGE_SCAN:
            # Écran intermédiaire : les participants scannent leur badge.
            # Rejouer reste possible (2 paumes) ; après BADGE_SCAN_SEC on passe à l'outro.
            self._update_start_hold(hands, now)
            if now - state.phase_entered_at >= BADGE_SCAN_SEC:
                state.phase = TRINITY_OUTRO
                state.phase_entered_at = now
                state.start_hold_since = None
            return self._snapshot()

        if state.phase == TRINITY_OUTRO:
            # La lecture du clip est pilotee par la boucle principale (comme INTRO) ;
            # ici on ne gere que le retour automatique a l'attract.
            if now - state.phase_entered_at >= TRINITY_OUTRO_DURATION:
                self._go_attract(now)
            return self._snapshot()

        if state.phase == INTRO:
            # La video est pilotee par la boucle principale (finish_intro).
            return self._snapshot()

        if state.phase == PILL_CHOICE:
            if now - state.phase_entered_at >= PILL_CHOICE_IDLE_SEC:
                self._go_attract(now)
                return self._snapshot()
            hover = pill_hover(hands)
            self._pill_hover = hover
            validated, progress = self._pill_hold.update(hover, now)
            self._figure_progress = progress
            if validated:
                state.chosen_pill = validated
                if validated == "blue":
                    # Fin de l'histoire : retour a la realite, camera normale.
                    state.phase = BLUE_ENDING
                    state.phase_entered_at = now
                else:
                    state.phase = COUNTDOWN
                    state.countdown_until = now + self.countdown_duration
                    self._set_flash("RED PILL ACCEPTED", now)
                self._figure_progress = 0.0
                self._pill_hover = None
                return self._snapshot()
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

        # Defi quiz : machine a etats interne (INTRO/ACTIVE/REVEAL), timer propre.
        if challenge.kind == "quiz":
            return self._update_quiz(challenge, hands, now)

        if now >= state.round_deadline:
            state.round_results.append(RoundResult(challenge.title, False, 0))
            state.combo = 0  # un echec casse le combo
            self._set_flash(TIMEOUT_MESSAGE, now)
            self._advance(now)
            return self._snapshot()

        success, message = self._detect_figure(challenge, hands, pose, now)
        if success:
            state.combo = min(MAX_COMBO, state.combo + 1)
            timer_ratio = max(0.0, state.round_deadline - now) / self.round_duration
            points = (BASE_POINTS + int(MAX_SPEED_BONUS * timer_ratio)) * state.combo
            state.score += points
            state.round_results.append(RoundResult(challenge.title, True, points))
            if state.combo > 1:
                message = f"{message}  COMBO x{state.combo}"
            self._set_flash(message, now)
            state.celebration_key = challenge.key
            state.celebration_started_at = now
            # La celebration finit a celebration_until ; la figure suivante ne
            # devient active que READY_SEC plus tard (ecran "prepare-toi").
            if challenge.key == "neo_dodge":
                celeb_dur = BULLET_TIME_REPLAY_SEC
            elif challenge.key == "bullet_stop":
                celeb_dur = BULLET_STOP_VIDEO_SEC
            else:
                celeb_dur = CELEBRATION_SEC
            state.celebration_until = now + celeb_dur
            self._advance(now, celeb_dur + READY_SEC)
            return self._snapshot()

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
        self._dodge_hold.reset()
        self._spoon.reset()
        self._bullet_stop.reset()
        self._quiz = None
        self._palm_anchor = None
        self._figure_progress = 0.0
        self._pill_hover = None

    def _detect_figure(
        self,
        challenge: Challenge,
        hands: Sequence[HandObservation],
        pose: PoseObservation | None,
        now: float,
    ) -> tuple[bool, str]:
        message = challenge.success_message
        self._figure_progress = 0.0
        self._pill_hover = None
        self._palm_anchor = None

        if challenge.kind == "dodge":
            # Posture tenue DODGE_HOLD_SEC : le joueur a le temps de se voir
            # en esquive, pas de validation sur un simple passage.
            active = dodge_matches(pose)
            validated, progress = self._dodge_hold.update(
                "dodge" if active else None, now
            )
            self._figure_progress = progress
            return validated is not None, message

        if challenge.kind == "bunny":
            active = bunny_ears_active(hands, pose)
            validated, progress = self._bunny_hold.update(
                "bunny" if active else None, now
            )
            self._figure_progress = progress
            return validated is not None, message

        if challenge.kind == "spoon":
            progress = self._spoon.update(hands)
            self._figure_progress = progress
            return progress >= 1.0, message

        if challenge.kind == "bullet_stop":
            # Grace : ignore la paume pendant les premieres secondes du round
            # pour eviter une validation involontaire depuis le decompte.
            if now < self.state.round_transition_at + BULLET_STOP_GRACE_SEC:
                return False, message
            anchor = bullet_stop_palm(hands)
            self._palm_anchor = anchor
            validated, progress = self._bullet_stop.update(
                "stop" if anchor else None, now
            )
            self._figure_progress = progress
            return validated is not None, message

        return False, message

    def _update_quiz(
        self,
        challenge: Challenge,
        hands: Sequence[HandObservation],
        now: float,
    ) -> GameEvent:
        """Pilote un defi quiz via QuizRuntime (cree paresseusement au 1er frame
        actif du round). Encaisse les points et avance quand la question est
        terminee (apres l'ecran REVEAL)."""
        if self._quiz is None:
            question = challenge.quiz_question
            if question is None:  # garde-fou : quiz sans question -> on saute
                self._advance(now, READY_SEC)
                return self._snapshot()
            self._quiz = QuizRuntime(
                question,
                duration_s=challenge.duration_sec,
                intro_sec=QUIZ_INTRO_SEC,
                reveal_sec=QUIZ_REVEAL_SEC,
                dwell_sec=QUIZ_DWELL_SEC,
                grace_sec=QUIZ_GRACE_SEC,
                zone_margin=QUIZ_ZONE_MARGIN,
            )

        quiz = self._quiz
        quiz.update(hands, now)
        self._figure_progress = max(quiz.dwell) if quiz.dwell else 0.0

        if quiz.finished:
            self._finish_quiz(challenge, quiz, now)

        return self._snapshot()

    def _finish_quiz(self, challenge: Challenge, quiz: QuizRuntime, now: float) -> None:
        state = self.state
        if quiz.result == "correct":
            state.combo = min(MAX_COMBO, state.combo + 1)
            points = (
                BASE_POINTS + int(MAX_SPEED_BONUS * quiz.timer_ratio)
            ) * state.combo
            state.score += points
            state.round_results.append(RoundResult(challenge.title, True, points))
            message = "BONNE REPONSE"
            if state.combo > 1:
                message = f"{message}  COMBO x{state.combo}"
            self._set_flash(message, now)
        else:
            # Mauvaise reponse OU timeout : 0 point, le combo casse.
            state.combo = 0
            state.round_results.append(RoundResult(challenge.title, False, 0))
            self._set_flash(
                "MAUVAISE REPONSE" if quiz.result == "incorrect" else TIMEOUT_MESSAGE,
                now,
            )
        self._advance(now, READY_SEC)

    def _advance(self, now: float, transition: float = ROUND_TRANSITION_SEC) -> None:
        state = self.state
        state.round_index += 1
        if state.round_index >= len(self.challenges):
            state.phase = SCORE
            state.phase_entered_at = now
            state.start_hold_since = None
            if state.score > self.best_score:
                self.best_score = state.score
                state.new_record = True
                self._save_best_score()
            return

        state.round_transition_at = now + transition
        state.round_deadline = state.round_transition_at + self.round_duration
        self._reset_round_trackers()

    def _set_flash(self, message: str, now: float) -> None:
        self.state.flash_message = message
        self.state.flash_until = now + FLASH_DURATION_SEC

    def _load_best_score(self) -> int:
        if self._best_score_path is None or not self._best_score_path.exists():
            return 0
        try:
            return int(
                json.loads(self._best_score_path.read_text()).get("best_score", 0)
            )
        except (ValueError, OSError, AttributeError):
            return 0

    def _save_best_score(self) -> None:
        if self._best_score_path is None:
            return
        try:
            self._best_score_path.write_text(
                json.dumps({"best_score": self.best_score})
            )
        except OSError as exc:
            print(f"[SCORE] could not persist best score: {exc}")

    def _snapshot(self) -> GameEvent:
        state = self.state
        now = self._now

        challenge = self.current_challenge() if state.phase == IN_ROUND else None
        timer_left = 0.0
        figure_active = False
        round_starts_in = 0.0
        if state.phase == IN_ROUND:
            timer_left = min(self.round_duration, max(0.0, state.round_deadline - now))
            figure_active = now >= state.round_transition_at
            round_starts_in = max(0.0, state.round_transition_at - now)

        if state.phase == IN_ROUND:
            display_round = state.round_index + 1
        elif state.phase == SCORE:
            display_round = len(self.challenges)
        else:
            display_round = 0

        flash = state.flash_message if now < state.flash_until else ""
        countdown_value = (
            max(0, math.ceil(state.countdown_until - now))
            if state.phase == COUNTDOWN
            else 0
        )
        start_hold_progress = 0.0
        if state.start_hold_since is not None:
            start_hold_progress = min(
                1.0, (now - state.start_hold_since) / START_HOLD_SEC
            )

        celebration_key = ""
        celebration_progress = 0.0
        if state.celebration_key and now < state.celebration_until:
            celebration_key = state.celebration_key
            window = max(1e-6, state.celebration_until - state.celebration_started_at)
            celebration_progress = min(
                1.0, max(0.0, (now - state.celebration_started_at) / window)
            )

        quiz = self._quiz if state.phase == IN_ROUND else None
        quiz_active = quiz is not None and not quiz.finished
        if quiz is not None:
            quiz_substate = quiz.substate
            quiz_question = quiz.question.question
            quiz_answers = list(quiz.question.answers)
            quiz_zones = list(quiz.zones)
            quiz_cursor = quiz.cursor
            quiz_dwell = list(quiz.dwell)
            quiz_selected_index = quiz.selected_index
            quiz_correct_index = quiz.question.correct_index
            quiz_result = quiz.result
            quiz_explanation = quiz.question.explanation
            quiz_timer_left = quiz.timer_left
            quiz_timer_total = quiz.duration_s
            quiz_hand_detected = quiz.hand_detected
        else:
            quiz_substate = ""
            quiz_question = ""
            quiz_answers = []
            quiz_zones = []
            quiz_cursor = None
            quiz_dwell = []
            quiz_selected_index = None
            quiz_correct_index = -1
            quiz_result = ""
            quiz_explanation = ""
            quiz_timer_left = 0.0
            quiz_timer_total = 0.0
            quiz_hand_detected = False

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
            figure_active=figure_active,
            round_starts_in=round_starts_in,
            flash_message=flash,
            countdown_value=countdown_value,
            figure_progress=self._figure_progress,
            pill_hover=self._pill_hover,
            spoon_anchor=self._spoon.anchor,
            spoon_angle=self._spoon.angle,
            palm_anchor=self._palm_anchor,
            start_hold_progress=start_hold_progress,
            chosen_pill=state.chosen_pill,
            rain_boost=state.phase in (ATTRACT, SCORE)
            or bool(flash),  # BLUE_ENDING/INTRO : pas de pluie (bypass dans main)
            combo=state.combo,
            celebration_key=celebration_key,
            celebration_progress=celebration_progress,
            best_score=self.best_score,
            new_record=state.new_record,
            round_results=list(state.round_results),
            badge_scan_time_left=(
                max(0.0, BADGE_SCAN_SEC - (now - state.phase_entered_at))
                if state.phase == BADGE_SCAN
                else 0.0
            ),
            quiz_active=quiz_active,
            quiz_substate=quiz_substate,
            quiz_question=quiz_question,
            quiz_answers=quiz_answers,
            quiz_zones=quiz_zones,
            quiz_cursor=quiz_cursor,
            quiz_dwell=quiz_dwell,
            quiz_selected_index=quiz_selected_index,
            quiz_correct_index=quiz_correct_index,
            quiz_result=quiz_result,
            quiz_explanation=quiz_explanation,
            quiz_timer_left=quiz_timer_left,
            quiz_timer_total=quiz_timer_total,
            quiz_hand_detected=quiz_hand_detected,
        )
