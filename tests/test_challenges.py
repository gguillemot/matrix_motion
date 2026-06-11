from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.challenges import detect_gesture, finger_states


def landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def blank_hand() -> list[SimpleNamespace]:
    return [landmark(0.5, 0.5) for _ in range(21)]


class ChallengeTests(unittest.TestCase):
    def test_detects_open_palm(self) -> None:
        landmarks = blank_hand()
        landmarks[3].x = 0.2
        landmarks[4].x = 0.1
        landmarks[6].y = 0.6
        landmarks[8].y = 0.4
        landmarks[10].y = 0.6
        landmarks[12].y = 0.4
        landmarks[14].y = 0.6
        landmarks[16].y = 0.4
        landmarks[18].y = 0.6
        landmarks[20].y = 0.4

        self.assertEqual(finger_states(landmarks, "Right"), (True, True, True, True, True))
        self.assertEqual(detect_gesture(landmarks, "Right"), "open_palm")

    def test_detects_thumbs_up(self) -> None:
        landmarks = blank_hand()
        landmarks[3].x = 0.2
        landmarks[4].x = 0.1
        landmarks[6].y = 0.6
        landmarks[8].y = 0.7
        landmarks[10].y = 0.6
        landmarks[12].y = 0.7
        landmarks[14].y = 0.6
        landmarks[16].y = 0.7
        landmarks[18].y = 0.6
        landmarks[20].y = 0.7

        self.assertEqual(detect_gesture(landmarks, "Right"), "thumbs_up")

    def test_detects_ok_sign(self) -> None:
        landmarks = blank_hand()
        landmarks[3].x = 0.2
        landmarks[4].x = 0.15
        landmarks[4].y = 0.16
        landmarks[8].x = 0.17
        landmarks[8].y = 0.16
        landmarks[12].y = 0.4
        landmarks[10].y = 0.6
        landmarks[16].y = 0.4
        landmarks[14].y = 0.6
        landmarks[20].y = 0.4
        landmarks[18].y = 0.6

        self.assertEqual(detect_gesture(landmarks, "Right"), "ok_sign")

    def test_detects_fist(self) -> None:
        landmarks = blank_hand()
        landmarks[3].x = 0.2
        landmarks[4].x = 0.3
        landmarks[6].y = 0.4
        landmarks[8].y = 0.6
        landmarks[10].y = 0.4
        landmarks[12].y = 0.6
        landmarks[14].y = 0.4
        landmarks[16].y = 0.6
        landmarks[18].y = 0.4
        landmarks[20].y = 0.6

        self.assertEqual(detect_gesture(landmarks, "Right"), "fist")
