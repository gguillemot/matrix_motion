from __future__ import annotations

import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


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

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"
