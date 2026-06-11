from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.challenges import PoseObservation, build_campaign, challenge_matches, classify_hand_gestures, detect_gesture, finger_states, pose_matches


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

    def test_detects_point(self) -> None:
        landmarks = blank_hand()
        landmarks[3].x = 0.4
        landmarks[4].x = 0.45
        landmarks[6].y = 0.6
        landmarks[8].y = 0.4
        landmarks[10].y = 0.7
        landmarks[12].y = 0.8
        landmarks[14].y = 0.7
        landmarks[16].y = 0.8
        landmarks[18].y = 0.7
        landmarks[20].y = 0.8

        self.assertEqual(detect_gesture(landmarks, "Right"), "point")

    def test_campaign_sequence_lengths(self) -> None:
        self.assertEqual(len(build_campaign(5)), 5)
        self.assertEqual(len(build_campaign(10)), 10)

    def test_challenge_match_requires_two_hands(self) -> None:
        campaign = build_campaign(5)
        bunny_ears = campaign[2]

        self.assertFalse(challenge_matches(bunny_ears, ["open_palm"]))
        self.assertTrue(challenge_matches(bunny_ears, ["open_palm", "open_palm"]))

    def test_classify_hand_gestures_returns_all_hands(self) -> None:
        hand_results = SimpleNamespace(
            hand_landmarks=[[landmark(0.5, 0.5) for _ in range(21)], [landmark(0.5, 0.5) for _ in range(21)]],
            handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]],
        )
        hand_results.hand_landmarks[0][3].x = 0.2
        hand_results.hand_landmarks[0][4].x = 0.1
        hand_results.hand_landmarks[0][6].y = 0.6
        hand_results.hand_landmarks[0][8].y = 0.7
        hand_results.hand_landmarks[0][10].y = 0.6
        hand_results.hand_landmarks[0][12].y = 0.7
        hand_results.hand_landmarks[0][14].y = 0.6
        hand_results.hand_landmarks[0][16].y = 0.7
        hand_results.hand_landmarks[0][18].y = 0.6
        hand_results.hand_landmarks[0][20].y = 0.7
        hand_results.hand_landmarks[1][3].x = 0.2
        hand_results.hand_landmarks[1][4].x = 0.3
        hand_results.hand_landmarks[1][6].y = 0.6
        hand_results.hand_landmarks[1][8].y = 0.4
        hand_results.hand_landmarks[1][10].y = 0.6
        hand_results.hand_landmarks[1][12].y = 0.4
        hand_results.hand_landmarks[1][14].y = 0.6
        hand_results.hand_landmarks[1][16].y = 0.4
        hand_results.hand_landmarks[1][18].y = 0.6
        hand_results.hand_landmarks[1][20].y = 0.4

        self.assertEqual(classify_hand_gestures(hand_results), ["thumbs_up", "open_palm"])

    def test_pose_matches_detects_lateral_dodge(self) -> None:
        campaign = build_campaign(5)
        neo_dodge = campaign[0]

        dodge_left = PoseObservation(nose_x=0.20, shoulder_center_x=0.52, shoulder_span=0.40)
        dodge_center = PoseObservation(nose_x=0.50, shoulder_center_x=0.52, shoulder_span=0.40)

        self.assertTrue(pose_matches(neo_dodge, dodge_left))
        self.assertFalse(pose_matches(neo_dodge, dodge_center))
