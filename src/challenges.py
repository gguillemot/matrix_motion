from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

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

# Agent Smith : immobilite totale. Le nez ne doit pas deriver de plus de 8 %
# de l'envergure d'epaules (~2.5 % de l'image) autour de sa position de
# reference pendant SCAN_HOLD_SEC : tolere juste le jitter du tracker, le
# moindre mouvement volontaire fait disparaitre les lunettes et relance le scan.
SCAN_MAX_DRIFT_RATIO = 0.08
SCAN_HOLD_SEC = 2.0


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
        key="smith_scan",
        title="Agent Smith",
        prompt="Ne bouge plus... scan en cours",
        kind="scan",
        duration_sec=8.0,
        background_asset="agent_smith.png",
        success_message="SCAN COMPLETE, MR. ANDERSON",
    ),
]


def build_breizhcamp_campaign(rng: random.Random | None = None, shuffle: bool = True) -> list[Challenge]:
    """Campagne d'une partie : les 4 epreuves de la pilule rouge, en ordre
    aleatoire par defaut. Le choix des pilules est une phase a part, jouee
    avant la campagne (voir GameEngine)."""
    campaign = [replace(challenge) for challenge in _BREIZHCAMP_FIGURES]
    if shuffle:
        (rng or random).shuffle(campaign)
    return campaign


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


class ScanTracker:
    """Progression d'immobilite : nez stable pendant SCAN_HOLD_SEC."""

    def __init__(self) -> None:
        self._ref: tuple[float, float] | None = None
        self._since = 0.0

    def reset(self) -> None:
        self._ref = None

    def update(self, pose: PoseObservation | None, now: float) -> float:
        if pose is None:
            self._ref = None
            return 0.0
        if self._ref is None:
            self._ref = (pose.nose_x, pose.nose_y)
            self._since = now
            return 0.0

        span = max(pose.shoulder_span, 1e-6)
        drift = math.hypot(pose.nose_x - self._ref[0], pose.nose_y - self._ref[1]) / span
        if drift > SCAN_MAX_DRIFT_RATIO:
            # Mouvement volontaire : on re-ancre et le scan repart de zero.
            self._ref = (pose.nose_x, pose.nose_y)
            self._since = now
            return 0.0
        return min(1.0, (now - self._since) / SCAN_HOLD_SEC)


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
