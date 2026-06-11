from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass(slots=True)
class Challenge:
    key: str
    title: str
    prompt: str
    target_gesture: str
    min_hands: int = 1
    duration_sec: float = 6.0
    background_asset: str = ""
    victory_pill: str = "blue"


_CHALLENGE_POOL: list[Challenge] = [
    Challenge(
        key="neo_dodge",
        title="Neo Dodge",
        prompt="Neo dodge en bullet time",
        target_gesture="fist",
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


def finger_states(landmarks: list, hand_label: str) -> tuple[bool, bool, bool, bool, bool]:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    if hand_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def detect_gesture(landmarks: list, hand_label: str) -> str:
    thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states(landmarks, hand_label)

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist_thumb_index = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    if dist_thumb_index < 0.05 and middle_up and ring_up and pinky_up:
        return "ok_sign"

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"

    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open_palm"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point"

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"


def classify_hand_gestures(hand_results) -> list[str]:
    gestures: list[str] = []

    if hand_results.hand_landmarks and hand_results.handedness:
        for lm_list, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
            hand_label = handedness[0].category_name
            gestures.append(detect_gesture(lm_list, hand_label))

    return gestures


def count_gestures(gestures: Sequence[str], target: str) -> int:
    return sum(gesture == target for gesture in gestures)


def challenge_matches(challenge: Challenge, gestures: Sequence[str]) -> bool:
    return count_gestures(gestures, challenge.target_gesture) >= challenge.min_hands
