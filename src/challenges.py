from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

import json
import logging
from pathlib import Path

from src.config import (
    QUIZ_DEFAULT_DURATION_S,
    QUIZ_MAX_ANSWERS,
    QUIZ_QUESTIONS_PATH,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# ---------------------------------------------------------------------------
# Seuils geometriques des figures (coordonnees MediaPipe normalisees 0..1)
# ---------------------------------------------------------------------------

# The Neo Dodge : offset lateral du nez par rapport au centre des epaules,
# normalise par l'envergure d'epaules. 0.30 impose de pencher franchement le
# buste et la tete (0.22, l'ancien seuil, se declenchait sur un simple
# mouvement de tete). La posture doit etre TENUE 0.5 s : le joueur a le temps
# de se voir en esquive, et pas de validation au passage.
DODGE_OFFSET_THRESHOLD = 0.30
DODGE_HOLD_SEC = 0.5

# Red Pill / Blue Pill : centres des pilules affichees en overlay (x, y
# normalises, rouge a gauche / bleue a droite dans l'image miroir) et rayon
# de la hitbox en distance euclidienne normalisee. 0.13 est volontairement
# genereux : sur un stand, on valide l'intention, pas la precision.
PILL_ZONES: dict[str, tuple[float, float]] = {"red": (0.30, 0.42), "blue": (0.70, 0.42)}
PILL_HITBOX_RADIUS = 0.13
# Maintien de la paume sur la pilule : 0.9 s pour laisser le temps de voir la
# jauge se remplir et eviter de valider une main qui ne fait que passer.
PILL_HOLD_SEC = 0.9

# Follow the White Rabbit : deux mains en "oreilles" (index + majeur leves,
# annulaire + auriculaire plies) avec le centre des paumes au-dessus du nez.
BUNNY_HOLD_SEC = 0.9

# There Is No Spoon : pince pouce-index (distance < 0.06, soit environ la
# largeur d'un doigt) qui tient la cuillere virtuelle, puis rotation cumulee
# du vecteur poignet -> base du majeur jusqu'a 80 degres : on a le temps de
# voir la cuillere se tordre.
SPOON_PINCH_MAX_DIST = 0.06
SPOON_BEND_TARGET_RAD = math.radians(80.0)
# Au-dela de ~35 deg entre deux frames c'est un saut de tracking, pas une
# rotation du joueur : on l'ignore pour ne pas gonfler le cumul.
SPOON_MAX_STEP_RAD = 0.6

# Neo Stops The Bullets : une paume ouverte LEVEE (palm_y au-dessus du milieu)
# maintenue BULLET_STOP_HOLD_SEC. Les balles convergent et se figent pendant le
# maintien ; la jauge pleine = Neo arrete les balles.
BULLET_STOP_PALM_MAX_Y = 0.60
BULLET_STOP_HOLD_SEC   = 2.5   # maintien exige pour arreter les balles
BULLET_STOP_GRACE_SEC  = 1.5   # grace en debut de round : evite la validation par carry-over


# ---------------------------------------------------------------------------
# Observations extraites des resultats MediaPipe (construites 1 fois / frame)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HandObservation:
    gesture: str
    palm_x: float
    palm_y: float
    is_pinch: bool
    hand_angle: float  # radians, vecteur poignet -> base du majeur
    # Bout de l'index (landmark 8), normalise 0..1 : sert de curseur de pointage
    # pour le quiz. Valeur par defaut = centre (compat ascendante des tests).
    index_x: float = 0.5
    index_y: float = 0.5


@dataclass(slots=True)
class PoseObservation:
    nose_x: float
    shoulder_center_x: float
    shoulder_span: float
    nose_y: float = 0.5

    @property
    def normalized_nose_offset(self) -> float:
        span = max(self.shoulder_span, 1e-6)
        return (self.nose_x - self.shoulder_center_x) / span


@dataclass(slots=True)
class Challenge:
    key: str
    title: str
    prompt: str
    target_gesture: str = ""
    kind: str = "gesture"
    motion_direction: str | None = None
    min_hands: int = 1
    duration_sec: float = 6.0
    background_asset: str = ""
    victory_pill: str = "blue"
    success_message: str = ""
    # Question portee par un defi de type "quiz" (None pour les figures geste).
    quiz_question: "QuizQuestion | None" = None


# ---------------------------------------------------------------------------
# Quiz Matrix : banque de questions + machine a etats du defi QCM
# ---------------------------------------------------------------------------

# Sous-etats d'un defi quiz, pilotes par QuizRuntime :
#  INTRO  : la question s'affiche, chrono fige (le temps de lire) ;
#  ACTIVE : le chrono decompte, le joueur pointe + maintient sa reponse ;
#  REVEAL : bonne/mauvaise reponse revelee + explication ;
#  DONE   : termine, le moteur encaisse les points et passe au defi suivant.
QUIZ_INTRO = "intro"
QUIZ_ACTIVE = "active"
QUIZ_REVEAL = "reveal"
QUIZ_DONE = "done"

# Ordre de difficulte pour la progression croissante des questions.
QUIZ_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


@dataclass(slots=True)
class QuizQuestion:
    id: str
    question: str
    answers: list[str]
    correct_index: int
    difficulty: str = "medium"
    duration_s: float | None = None
    explanation: str = ""


# Fallback integre : si la banque JSON manque ou est invalide, le jeu ne doit
# JAMAIS crasher sur le stand. Quelques questions de secours suffisent.
_FALLBACK_QUIZ_QUESTIONS: list[QuizQuestion] = [
    QuizQuestion(
        id="fallback_pill",
        question="Quelle pilule fait decouvrir la verite a Neo ?",
        answers=["La rouge", "La bleue"],
        correct_index=0,
        difficulty="easy",
        explanation="La pilule rouge revele la realite de la Matrice.",
    ),
    QuizQuestion(
        id="fallback_neo",
        question="Comment s'appelle l'Elu ?",
        answers=["Morpheus", "Neo", "Cypher", "Smith"],
        correct_index=1,
        difficulty="easy",
        explanation="Neo est l'anagramme de One (l'Elu).",
    ),
    QuizQuestion(
        id="fallback_spoon",
        question="Que dit l'enfant a propos de la cuillere ?",
        answers=["Il n'y a pas de cuillere", "La cuillere est tordue"],
        correct_index=0,
        difficulty="medium",
        explanation="\"There is no spoon\" : c'est toi qui plies, pas la cuillere.",
    ),
]


def quiz_answer_zones(
    answer_count: int, max_answers: int = QUIZ_MAX_ANSWERS
) -> list[tuple[float, float, float, float]]:
    """Rectangles normalises (x0, y0, x1, y1) des cartes de reponses.

    2 reponses -> 1 ligne de 2 cartes ; 3-4 reponses -> grille 2x2. Les cartes
    sont volontairement BASSES et PEU HAUTES, ancrees au bas de l'ecran : tout
    le haut/milieu reste disponible pour la question, qui doit rester visible
    meme en 720p ou en plein ecran (ratio quelconque)."""
    n = max(1, min(max_answers, answer_count))
    area_x0, area_x1 = 0.06, 0.94
    bottom = 0.94  # bas des cartes
    gap = 0.025
    cell_h = 0.15  # hauteur fixe et compacte d'une carte
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)
    cell_w = (area_x1 - area_x0 - gap * (cols - 1)) / cols
    total_h = cell_h * rows + gap * (rows - 1)
    area_y0 = bottom - total_h  # ancre la grille au bas de l'ecran
    zones: list[tuple[float, float, float, float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        x0 = area_x0 + c * (cell_w + gap)
        y0 = area_y0 + r * (cell_h + gap)
        zones.append((x0, y0, x0 + cell_w, y0 + cell_h))
    return zones


def _validate_quiz_question(raw: object) -> QuizQuestion | None:
    """Valide une entree brute (dict JSON). Retourne None si incoherente."""
    if not isinstance(raw, dict):
        return None
    try:
        qid = str(raw["id"])
        text = str(raw["question"])
        answers = [str(a) for a in raw["answers"]]
        correct = int(raw["correct_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (2 <= len(answers) <= QUIZ_MAX_ANSWERS):
        return None
    if not (0 <= correct < len(answers)):
        return None
    difficulty = str(raw.get("difficulty", "medium")).lower()
    if difficulty not in QUIZ_DIFFICULTY_RANK:
        difficulty = "medium"
    raw_duration = raw.get("duration_s")
    duration = (
        float(raw_duration)
        if isinstance(raw_duration, (int, float)) and raw_duration > 0
        else None
    )
    explanation = str(raw.get("explanation", "") or "")
    return QuizQuestion(
        id=qid,
        question=text,
        answers=answers,
        correct_index=correct,
        difficulty=difficulty,
        duration_s=duration,
        explanation=explanation,
    )


def load_quiz_questions(path: str | Path | None = None) -> list[QuizQuestion]:
    """Charge la banque de questions de maniere tolerante.

    JSON attendu : {"questions": [...]} ou directement une liste. Toute entree
    invalide est ignoree. Si le fichier manque/est illisible ou ne contient
    aucune question valide, on retombe sur le fallback integre (+ avertissement),
    pour que le jeu ne crashe jamais sur le stand."""
    target = Path(path) if path is not None else QUIZ_QUESTIONS_PATH
    questions: list[QuizQuestion] = []
    try:
        raw = json.loads(Path(target).read_text(encoding="utf-8"))
        items = raw["questions"] if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError("le JSON ne contient pas de liste de questions")
        questions = [
            q for q in (_validate_quiz_question(item) for item in items) if q is not None
        ]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logging.warning(
            "[QUIZ] chargement de %s impossible (%s) : fallback integre", target, exc
        )
        questions = []
    if not questions:
        logging.warning(
            "[QUIZ] aucune question valide trouvee, utilisation du fallback integre"
        )
        questions = [replace(q) for q in _FALLBACK_QUIZ_QUESTIONS]
    return questions


def select_quiz_questions(
    questions: Sequence[QuizQuestion],
    count: int,
    rng: random.Random | None = None,
    shuffle: bool = True,
) -> list[QuizQuestion]:
    """Selectionne `count` questions distinctes, en difficulte croissante."""
    if not questions or count <= 0:
        return []
    pool = list(questions)
    if shuffle:
        (rng or random).shuffle(pool)
    # Tri stable par difficulte : ramp easy -> medium -> hard, ordre aleatoire
    # conserve a difficulte egale.
    pool.sort(key=lambda q: QUIZ_DIFFICULTY_RANK.get(q.difficulty, 1))
    return pool[:count]


def make_quiz_challenge(question: QuizQuestion) -> Challenge:
    """Construit un defi de type quiz a partir d'une question."""
    return Challenge(
        key=f"quiz_{question.id}",
        title="Matrix Quiz",
        prompt="Pointe ta reponse et maintiens",
        kind="quiz",
        duration_sec=question.duration_s or QUIZ_DEFAULT_DURATION_S,
        quiz_question=question,
    )


class QuizRuntime:
    """Machine a etats d'une question : INTRO -> ACTIVE -> REVEAL -> DONE.

    `update(hands, now)` fait avancer l'etat ; les attributs publics (substate,
    cursor, dwell, timer_left, selected_index, result...) sont lus par le rendu
    et par le moteur (scoring a la fin)."""

    def __init__(
        self,
        question: QuizQuestion,
        *,
        duration_s: float,
        intro_sec: float,
        reveal_sec: float,
        dwell_sec: float,
        grace_sec: float,
        zone_margin: float,
    ) -> None:
        self.question = question
        self.duration_s = duration_s
        self.intro_sec = intro_sec
        self.reveal_sec = reveal_sec
        self.dwell_sec = dwell_sec
        self.grace_sec = grace_sec
        self.zone_margin = zone_margin
        self.zones = quiz_answer_zones(len(question.answers))

        self.substate = QUIZ_INTRO
        self.created_at: float | None = None
        self.active_at: float | None = None
        self.reveal_at: float | None = None
        self.selected_index: int | None = None
        self.result: str = ""  # "correct" / "incorrect" / "timeout"
        self.timer_left = duration_s
        self.timer_ratio = 0.0  # temps restant (0..1) au verrouillage, pour le bonus
        self.cursor: tuple[float, float] | None = None
        self.hand_detected = False
        self.dwell = [0.0] * len(question.answers)
        self._dwell_zone: int | None = None
        self._dwell_since: float | None = None

    @property
    def finished(self) -> bool:
        return self.substate == QUIZ_DONE

    def update(self, hands: Sequence[HandObservation], now: float) -> None:
        if self.created_at is None:
            self.created_at = now

        # Curseur = bout de l'index de la premiere main detectee (image miroir :
        # les coordonnees correspondent deja a ce que voit le joueur).
        self.hand_detected = bool(hands)
        self.cursor = (hands[0].index_x, hands[0].index_y) if hands else None

        if self.substate == QUIZ_INTRO:
            self.timer_left = self.duration_s
            if now - self.created_at >= self.intro_sec:
                self.substate = QUIZ_ACTIVE
                self.active_at = now
            return

        if self.substate == QUIZ_ACTIVE:
            elapsed = now - (self.active_at or now)
            self.timer_left = max(0.0, self.duration_s - elapsed)
            # Grace : on ignore le pointage le temps que la main arrive.
            zone_idx = None
            if elapsed >= self.grace_sec and self.cursor is not None:
                zone_idx = self._zone_at(self.cursor)
            self._update_dwell(zone_idx, now)
            if zone_idx is not None and self.dwell[zone_idx] >= 1.0:
                self._lock(zone_idx, now)
                return
            if self.timer_left <= 0.0:
                self._lock(None, now)  # timeout : aucune reponse
            return

        if self.substate == QUIZ_REVEAL:
            if now - (self.reveal_at or now) >= self.reveal_sec:
                self.substate = QUIZ_DONE
            return

    def _zone_at(self, point: tuple[float, float]) -> int | None:
        x, y = point
        for i, (x0, y0, x1, y1) in enumerate(self.zones):
            mx = (x1 - x0) * self.zone_margin
            my = (y1 - y0) * self.zone_margin
            if x0 + mx <= x <= x1 - mx and y0 + my <= y <= y1 - my:
                return i
        return None

    def _update_dwell(self, zone_idx: int | None, now: float) -> None:
        if zone_idx != self._dwell_zone:
            # Changement de zone (ou main sortie) : la jauge repart de zero.
            self._dwell_zone = zone_idx
            self._dwell_since = now
            self.dwell = [0.0] * len(self.dwell)
        if zone_idx is None or self._dwell_since is None:
            return
        if self.dwell_sec <= 0:
            self.dwell[zone_idx] = 1.0
            return
        progress = (now - self._dwell_since) / self.dwell_sec
        self.dwell[zone_idx] = min(1.0, max(0.0, progress))

    def _lock(self, idx: int | None, now: float) -> None:
        self.selected_index = idx
        if idx is None:
            self.result = "timeout"
            self.timer_ratio = 0.0
        elif idx == self.question.correct_index:
            self.result = "correct"
            self.timer_ratio = max(0.0, self.timer_left / self.duration_s)
        else:
            self.result = "incorrect"
            self.timer_ratio = 0.0
        self.substate = QUIZ_REVEAL
        self.reveal_at = now


# ---------------------------------------------------------------------------
# Campagne BreizhCamp 2026 : les 5 figures Matrix
# ---------------------------------------------------------------------------

_BREIZHCAMP_FIGURES: list[Challenge] = [
    Challenge(
        key="neo_dodge",
        title="The Neo Dodge",
        prompt="Penche-toi fort sur le cote, comme Neo !",
        kind="dodge",
        duration_sec=8.0,
        background_asset="neo_dodge.png",
        success_message="DODGE SUCCESSFUL !",
    ),
    Challenge(
        key="white_rabbit",
        title="Follow The White Rabbit",
        prompt="Oreilles de lapin au-dessus de la tete !",
        kind="bunny",
        duration_sec=8.0,
        background_asset="bunny_ears.png",
        success_message="FOLLOW THE WHITE RABBIT !",
    ),
    Challenge(
        key="no_spoon",
        title="There Is No Spoon",
        prompt="Pince la cuillere et tourne la main",
        kind="spoon",
        duration_sec=8.0,
        background_asset="spoon.png",
        success_message="THERE IS NO SPOON",
    ),
    Challenge(
        key="bullet_stop",
        title="Stop The Bullets",
        prompt="Leve la paume ouverte... arrete les balles !",
        kind="bullet_stop",
        duration_sec=8.0,
        background_asset="bullet_stop.png",
        success_message="YOU ARE THE ONE",
    ),
]


def build_breizhcamp_campaign(
    rng: random.Random | None = None,
    shuffle: bool = True,
    quiz_questions: Sequence[QuizQuestion] | None = None,
    max_quiz: int | None = None,
) -> list[Challenge]:
    """Campagne d'une partie : les 4 epreuves de la pilule rouge, en ordre
    aleatoire par defaut. Le choix des pilules est une phase a part, jouee
    avant la campagne (voir GameEngine).

    Si `quiz_questions` est fourni, on intercale STRICTEMENT un defi quiz apres
    chaque figure (figure, quiz, figure, quiz...), dans la limite de `max_quiz`
    et du nombre de questions disponibles. Sinon, comportement historique
    (figures seules) : les tests existants restent valides."""
    campaign = [replace(challenge) for challenge in _BREIZHCAMP_FIGURES]
    if shuffle:
        (rng or random).shuffle(campaign)
    if not quiz_questions:
        return campaign

    quota = len(campaign) if max_quiz is None else min(max_quiz, len(campaign))
    selected = select_quiz_questions(quiz_questions, quota, rng, shuffle)
    interleaved: list[Challenge] = []
    qi = 0
    for figure in campaign:
        interleaved.append(figure)
        if qi < len(selected):
            interleaved.append(make_quiz_challenge(selected[qi]))
            qi += 1
    return interleaved


# ---------------------------------------------------------------------------
# Pool legacy (mode campagne historique, conserve pour compatibilite)
# ---------------------------------------------------------------------------

_CHALLENGE_POOL: list[Challenge] = [
    Challenge(
        key="neo_dodge",
        title="Neo Dodge",
        prompt="Neo dodge en bullet time",
        target_gesture="neo_dodge",
        kind="pose",
        motion_direction="either",
        duration_sec=6.0,
        background_asset="neo_dodge.png",
    ),
    Challenge(
        key="red_pill",
        title="Pilule Rouge",
        prompt="Pointe la pilule rouge",
        target_gesture="point",
        duration_sec=6.0,
        background_asset="red_pill.png",
    ),
    Challenge(
        key="bunny_ears",
        title="Bunny Ears",
        prompt="Fais des oreilles de lapin",
        target_gesture="open_palm",
        min_hands=2,
        duration_sec=7.0,
        background_asset="bunny_ears.png",
    ),
    Challenge(
        key="bullet_stop",
        title="Stop Bullets",
        prompt="Main ouverte pour arrêter les balles",
        target_gesture="open_palm",
        duration_sec=6.0,
        background_asset="bullet_stop.png",
    ),
    Challenge(
        key="kung_fu",
        title="Kung Fu",
        prompt="Position de karaté / kung fu",
        target_gesture="fist",
        duration_sec=6.0,
        background_asset="kung_fu.png",
    ),
    Challenge(
        key="agent_smith",
        title="Agent Smith",
        prompt="Montre le signe OK",
        target_gesture="ok_sign",
        duration_sec=6.0,
        background_asset="agent_smith.png",
    ),
    Challenge(
        key="trinity",
        title="Trinity",
        prompt="Main ouverte, maintien du signal",
        target_gesture="open_palm",
        duration_sec=6.0,
        background_asset="trinity.png",
    ),
    Challenge(
        key="sentinel",
        title="Sentinel",
        prompt="Pointe l'ennemi",
        target_gesture="point",
        duration_sec=6.0,
        background_asset="sentinel.png",
    ),
    Challenge(
        key="the_one",
        title="The One",
        prompt="Deux mains ouvertes pour passer au niveau suivant",
        target_gesture="open_palm",
        min_hands=2,
        duration_sec=7.0,
        background_asset="the_one.png",
    ),
    Challenge(
        key="victory",
        title="Liberation",
        prompt="Dernier mouvement, mains ouvertes",
        target_gesture="open_palm",
        min_hands=2,
        duration_sec=7.0,
        background_asset="victory.png",
        victory_pill="red",
    ),
]


def build_campaign(sequence_length: int) -> list[Challenge]:
    campaign = [replace(challenge) for challenge in _CHALLENGE_POOL[:sequence_length]]
    if campaign:
        campaign[-1].victory_pill = "blue" if sequence_length == 5 else "red"
    return campaign


# ---------------------------------------------------------------------------
# Reconnaissance des gestes main
# ---------------------------------------------------------------------------


def finger_states(landmarks: list) -> tuple[bool, bool, bool, bool, bool]:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    # Pouce : test geometrique auto-reference. L'image est passee en miroir
    # (cv2.flip) AVANT MediaPipe, donc le label Right/Left est inverse et ne
    # peut pas servir. Le pouce est etendu s'il pointe du cote de l'index,
    # c'est-a-dire si (thumb_tip - thumb_ip) suit la direction auriculaire ->
    # index de la main elle-meme : insensible au miroir, a la lateralite et
    # au cote montre (paume ou revers).
    hand_dir_x = landmarks[5].x - landmarks[17].x
    hand_dir_y = landmarks[5].y - landmarks[17].y
    thumb_up = (thumb_tip.x - thumb_ip.x) * hand_dir_x + (thumb_tip.y - thumb_ip.y) * hand_dir_y > 0

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def detect_gesture(landmarks: list) -> str:
    thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states(landmarks)

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist_thumb_index = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    if dist_thumb_index < 0.05 and middle_up and ring_up and pinky_up:
        return "ok_sign"

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"

    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open_palm"

    # Oreilles de lapin : index + majeur leves, annulaire + auriculaire plies
    # (le pouce est libre : replie ou non selon les joueurs).
    if index_up and middle_up and not ring_up and not pinky_up:
        return "bunny_ears"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point"

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"


def classify_hand_gestures(hand_results) -> list[str]:
    gestures: list[str] = []

    if hand_results.hand_landmarks:
        for lm_list in hand_results.hand_landmarks:
            gestures.append(detect_gesture(lm_list))

    return gestures


def observe_hands(hand_results) -> list[HandObservation]:
    """Construit les observations de mains (geste + position + pince + angle)."""
    observations: list[HandObservation] = []

    if hand_results.hand_landmarks:
        for lm_list in hand_results.hand_landmarks:
            gesture = detect_gesture(lm_list)

            wrist, index_mcp, middle_mcp, pinky_mcp = lm_list[0], lm_list[5], lm_list[9], lm_list[17]
            palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
            palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0

            thumb_tip, index_tip = lm_list[4], lm_list[8]
            is_pinch = float(np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)) < SPOON_PINCH_MAX_DIST
            hand_angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)

            observations.append(
                HandObservation(
                    gesture=gesture,
                    palm_x=palm_x,
                    palm_y=palm_y,
                    is_pinch=is_pinch,
                    hand_angle=hand_angle,
                    index_x=float(index_tip.x),
                    index_y=float(index_tip.y),
                )
            )

    return observations


def observe_pose(pose_results) -> PoseObservation | None:
    pose_landmarks = getattr(pose_results, "pose_landmarks", None)
    if not pose_landmarks:
        return None

    landmarks = pose_landmarks[0]
    if len(landmarks) < 25:
        return None

    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    if not hasattr(nose, "x") or not hasattr(left_shoulder, "x") or not hasattr(right_shoulder, "x"):
        return None

    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
    shoulder_span = abs(right_shoulder.x - left_shoulder.x)
    return PoseObservation(
        nose_x=nose.x,
        shoulder_center_x=shoulder_center_x,
        shoulder_span=shoulder_span,
        nose_y=nose.y,
    )


# ---------------------------------------------------------------------------
# Matchers des figures BreizhCamp
# ---------------------------------------------------------------------------


def dodge_matches(pose: PoseObservation | None) -> bool:
    if pose is None:
        return False
    return abs(pose.normalized_nose_offset) >= DODGE_OFFSET_THRESHOLD


def pill_hover(hands: Sequence[HandObservation]) -> str | None:
    """Pilule survolee par une paume ouverte (l'image est en miroir : les
    coordonnees des landmarks correspondent deja a ce que voit le joueur)."""
    for hand in hands:
        if hand.gesture != "open_palm":
            continue
        for pill, (zone_x, zone_y) in PILL_ZONES.items():
            if math.hypot(hand.palm_x - zone_x, hand.palm_y - zone_y) <= PILL_HITBOX_RADIUS:
                return pill
    return None


def bunny_ears_active(hands: Sequence[HandObservation], pose: PoseObservation | None) -> bool:
    # Sans pose detectee, on exige les mains dans le tiers superieur de l'image.
    head_y = pose.nose_y if pose is not None else 0.38
    ears = [hand for hand in hands if hand.gesture == "bunny_ears" and hand.palm_y < head_y]
    return len(ears) >= 2


def bullet_stop_palm(hands: Sequence[HandObservation]) -> tuple[float, float] | None:
    """Renvoie (x, y) de la paume ouverte levee qui arrete les balles, ou None.
    Sert a la fois a valider la figure et a ancrer le rendu des balles.
    (image en miroir : coords = ce que voit le joueur.)"""
    for hand in hands:
        if hand.gesture == "open_palm" and hand.palm_y < BULLET_STOP_PALM_MAX_Y:
            return (hand.palm_x, hand.palm_y)
    return None


class HoldTracker:
    """Valide une condition maintenue `duration` secondes sans interruption.

    `update` renvoie (cle_validee | None, progression 0..1). Changer de cle
    (ex : passer de la pilule rouge a la bleue) remet la progression a zero.
    """

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self._key: str | None = None
        self._since = 0.0

    def reset(self) -> None:
        self._key = None

    def update(self, key: str | None, now: float) -> tuple[str | None, float]:
        if key != self._key:
            self._key = key
            self._since = now
        if self._key is None:
            return None, 0.0
        if self.duration <= 0:
            return self._key, 1.0
        progress = (now - self._since) / self.duration
        if progress >= 1.0:
            return self._key, 1.0
        return None, max(0.0, progress)


class SpoonTracker:
    """Cumule la rotation de la main pincee jusqu'a SPOON_BEND_TARGET_RAD."""

    def __init__(self) -> None:
        self.total_rotation = 0.0
        self.anchor: tuple[float, float] | None = None
        self.angle = 0.0
        self._last_angle: float | None = None

    def reset(self) -> None:
        self.total_rotation = 0.0
        self.anchor = None
        self._last_angle = None

    @property
    def progress(self) -> float:
        return min(1.0, self.total_rotation / SPOON_BEND_TARGET_RAD)

    def update(self, hands: Sequence[HandObservation]) -> float:
        pinched = next((hand for hand in hands if hand.is_pinch), None)
        if pinched is None:
            # Perdre la pince ne penalise pas : on garde le cumul, on
            # reprendra la mesure d'angle a la prochaine pince.
            self.anchor = None
            self._last_angle = None
            return self.progress

        self.anchor = (pinched.palm_x, pinched.palm_y)
        self.angle = pinched.hand_angle
        if self._last_angle is not None:
            delta = math.atan2(
                math.sin(self.angle - self._last_angle),
                math.cos(self.angle - self._last_angle),
            )
            if abs(delta) <= SPOON_MAX_STEP_RAD:
                self.total_rotation += abs(delta)
        self._last_angle = self.angle
        return self.progress



# ---------------------------------------------------------------------------
# Matchers legacy (mode campagne historique)
# ---------------------------------------------------------------------------


def count_gestures(gestures: Sequence[str], target: str) -> int:
    return sum(gesture == target for gesture in gestures)


def challenge_matches(challenge: Challenge, gestures: Sequence[str]) -> bool:
    return count_gestures(gestures, challenge.target_gesture) >= challenge.min_hands


def pose_matches(challenge: Challenge, pose_observation: PoseObservation | None) -> bool:
    if challenge.kind != "pose" or pose_observation is None:
        return False

    offset = pose_observation.normalized_nose_offset
    if challenge.motion_direction == "left":
        return offset <= -0.22
    if challenge.motion_direction == "right":
        return offset >= 0.22

    return abs(offset) >= 0.22
