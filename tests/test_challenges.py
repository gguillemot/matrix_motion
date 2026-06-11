from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.challenges import (
    DODGE_OFFSET_THRESHOLD,
    PILL_ZONES,
    HandObservation,
    HoldTracker,
    PoseObservation,
    ScanTracker,
    SpoonTracker,
    build_breizhcamp_campaign,
    build_campaign,
    bunny_ears_active,
    challenge_matches,
    classify_hand_gestures,
    detect_gesture,
    dodge_matches,
    finger_states,
    pill_hover,
    pose_matches,
)


def landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def blank_hand() -> list[SimpleNamespace]:
    return [landmark(0.5, 0.5) for _ in range(21)]


def hand_obs(
    gesture: str = "open_palm",
    palm_x: float = 0.5,
    palm_y: float = 0.5,
    is_pinch: bool = False,
    hand_angle: float = 0.0,
) -> HandObservation:
    return HandObservation(gesture=gesture, palm_x=palm_x, palm_y=palm_y, is_pinch=is_pinch, hand_angle=hand_angle)


def open_palm_hand() -> list[SimpleNamespace]:
    """Paume ouverte : direction auriculaire->index vers la gauche (lm[5] < lm[17]),
    pouce etendu dans le meme sens, 4 doigts leves."""
    landmarks = blank_hand()
    landmarks[5].x = 0.3
    landmarks[17].x = 0.7
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
    return landmarks


class GestureTests(unittest.TestCase):
    def test_detects_open_palm(self) -> None:
        landmarks = open_palm_hand()

        self.assertEqual(finger_states(landmarks), (True, True, True, True, True))
        self.assertEqual(detect_gesture(landmarks), "open_palm")

    def test_open_palm_is_mirror_invariant(self) -> None:
        # Le flux camera est en miroir : la detection doit etre identique si
        # tous les x sont inverses (x -> 1 - x).
        landmarks = open_palm_hand()
        for lm in landmarks:
            lm.x = 1.0 - lm.x

        self.assertEqual(detect_gesture(landmarks), "open_palm")

    def test_open_palm_detected_on_back_of_hand(self) -> None:
        # Revers de main : la direction auriculaire->index est inversee, le
        # pouce aussi. La main ouverte doit rester detectee (choix stand :
        # paume OU revers acceptes).
        landmarks = open_palm_hand()
        landmarks[5].x = 0.7
        landmarks[17].x = 0.3
        landmarks[3].x = 0.8
        landmarks[4].x = 0.9

        self.assertEqual(detect_gesture(landmarks), "open_palm")

    def test_detects_thumbs_up(self) -> None:
        landmarks = blank_hand()
        landmarks[5].x = 0.3
        landmarks[17].x = 0.7
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

        self.assertEqual(detect_gesture(landmarks), "thumbs_up")

    def test_detects_fist(self) -> None:
        landmarks = blank_hand()
        # Pouce replie : il pointe a l'oppose de la direction auriculaire->index.
        landmarks[5].x = 0.3
        landmarks[17].x = 0.7
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

        self.assertEqual(detect_gesture(landmarks), "fist")

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

        self.assertEqual(detect_gesture(landmarks), "point")

    def test_detects_bunny_ears(self) -> None:
        # Index + majeur leves, annulaire + auriculaire plies, pouce replie.
        landmarks = blank_hand()
        landmarks[3].x = 0.4
        landmarks[4].x = 0.45
        landmarks[6].y = 0.6
        landmarks[8].y = 0.4
        landmarks[10].y = 0.6
        landmarks[12].y = 0.4
        landmarks[14].y = 0.7
        landmarks[16].y = 0.8
        landmarks[18].y = 0.7
        landmarks[20].y = 0.8

        self.assertEqual(detect_gesture(landmarks), "bunny_ears")

    def test_classify_hand_gestures_returns_all_hands(self) -> None:
        hand_results = SimpleNamespace(
            hand_landmarks=[blank_hand(), blank_hand()],
            handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]],
        )
        hand_results.hand_landmarks[0][5].x = 0.3
        hand_results.hand_landmarks[0][17].x = 0.7
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
        hand_results.hand_landmarks[1][5].x = 0.7
        hand_results.hand_landmarks[1][17].x = 0.3
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


class BreizhcampCampaignTests(unittest.TestCase):
    def test_campaign_has_five_figures(self) -> None:
        campaign = build_breizhcamp_campaign(shuffle=False)
        self.assertEqual(len(campaign), 5)
        self.assertEqual(
            [challenge.key for challenge in campaign],
            ["neo_dodge", "pill_choice", "white_rabbit", "no_spoon", "smith_scan"],
        )

    def test_campaign_always_contains_pill_figure(self) -> None:
        import random

        campaign = build_breizhcamp_campaign(random.Random(7))
        self.assertIn("pill_choice", [challenge.key for challenge in campaign])
        self.assertEqual(len(campaign), 5)


class DodgeTests(unittest.TestCase):
    def test_dodge_requires_strong_lean(self) -> None:
        span = 0.40
        center = 0.50
        below = PoseObservation(
            nose_x=center + (DODGE_OFFSET_THRESHOLD - 0.02) * span,
            shoulder_center_x=center,
            shoulder_span=span,
        )
        above = PoseObservation(
            nose_x=center + (DODGE_OFFSET_THRESHOLD + 0.02) * span,
            shoulder_center_x=center,
            shoulder_span=span,
        )

        self.assertFalse(dodge_matches(below))
        self.assertTrue(dodge_matches(above))
        self.assertFalse(dodge_matches(None))

    def test_dodge_works_both_sides(self) -> None:
        span = 0.40
        left = PoseObservation(nose_x=0.50 - 0.35 * span, shoulder_center_x=0.50, shoulder_span=span)
        self.assertTrue(dodge_matches(left))


class PillTests(unittest.TestCase):
    def test_open_palm_on_red_pill(self) -> None:
        red_x, red_y = PILL_ZONES["red"]
        self.assertEqual(pill_hover([hand_obs(palm_x=red_x + 0.02, palm_y=red_y - 0.02)]), "red")

    def test_open_palm_on_blue_pill(self) -> None:
        blue_x, blue_y = PILL_ZONES["blue"]
        self.assertEqual(pill_hover([hand_obs(palm_x=blue_x, palm_y=blue_y)]), "blue")

    def test_closed_hand_does_not_grab(self) -> None:
        red_x, red_y = PILL_ZONES["red"]
        self.assertIsNone(pill_hover([hand_obs(gesture="fist", palm_x=red_x, palm_y=red_y)]))

    def test_palm_outside_zones(self) -> None:
        self.assertIsNone(pill_hover([hand_obs(palm_x=0.5, palm_y=0.9)]))


class BunnyTests(unittest.TestCase):
    def test_two_bunny_hands_above_nose(self) -> None:
        pose = PoseObservation(nose_x=0.5, shoulder_center_x=0.5, shoulder_span=0.3, nose_y=0.45)
        hands = [
            hand_obs(gesture="bunny_ears", palm_x=0.42, palm_y=0.25),
            hand_obs(gesture="bunny_ears", palm_x=0.58, palm_y=0.28),
        ]
        self.assertTrue(bunny_ears_active(hands, pose))

    def test_hands_below_nose_do_not_count(self) -> None:
        pose = PoseObservation(nose_x=0.5, shoulder_center_x=0.5, shoulder_span=0.3, nose_y=0.45)
        hands = [
            hand_obs(gesture="bunny_ears", palm_x=0.42, palm_y=0.60),
            hand_obs(gesture="bunny_ears", palm_x=0.58, palm_y=0.28),
        ]
        self.assertFalse(bunny_ears_active(hands, pose))

    def test_one_hand_is_not_enough(self) -> None:
        pose = PoseObservation(nose_x=0.5, shoulder_center_x=0.5, shoulder_span=0.3, nose_y=0.45)
        self.assertFalse(bunny_ears_active([hand_obs(gesture="bunny_ears", palm_y=0.2)], pose))


class HoldTrackerTests(unittest.TestCase):
    def test_validates_after_duration(self) -> None:
        tracker = HoldTracker(0.4)
        self.assertEqual(tracker.update("red", 0.0), (None, 0.0))
        key, progress = tracker.update("red", 0.2)
        self.assertIsNone(key)
        self.assertAlmostEqual(progress, 0.5)
        self.assertEqual(tracker.update("red", 0.45), ("red", 1.0))

    def test_changing_key_resets_progress(self) -> None:
        tracker = HoldTracker(0.4)
        tracker.update("red", 0.0)
        tracker.update("blue", 0.3)
        key, _ = tracker.update("blue", 0.5)
        self.assertIsNone(key)
        self.assertEqual(tracker.update("blue", 0.75), ("blue", 1.0))

    def test_losing_key_resets_progress(self) -> None:
        tracker = HoldTracker(0.4)
        tracker.update("red", 0.0)
        tracker.update(None, 0.3)
        key, progress = tracker.update("red", 0.4)
        self.assertIsNone(key)
        self.assertAlmostEqual(progress, 0.0)


class SpoonTrackerTests(unittest.TestCase):
    def test_accumulates_rotation_until_target(self) -> None:
        tracker = SpoonTracker()
        self.assertAlmostEqual(tracker.update([hand_obs(is_pinch=True, hand_angle=0.0)]), 0.0)
        progress = 0.0
        for angle in (0.45, 0.9, 1.35, 1.8):
            progress = tracker.update([hand_obs(is_pinch=True, hand_angle=angle)])
        self.assertEqual(progress, 1.0)  # 1.8 rad cumules > 80 deg

    def test_tracking_jump_is_ignored(self) -> None:
        tracker = SpoonTracker()
        tracker.update([hand_obs(is_pinch=True, hand_angle=0.0)])
        # Saut de 2 rad entre deux frames : artefact de tracking, pas une rotation.
        progress = tracker.update([hand_obs(is_pinch=True, hand_angle=2.0)])
        self.assertAlmostEqual(progress, 0.0)

    def test_losing_pinch_keeps_progress(self) -> None:
        tracker = SpoonTracker()
        tracker.update([hand_obs(is_pinch=True, hand_angle=0.0)])
        tracker.update([hand_obs(is_pinch=True, hand_angle=0.4)])
        before = tracker.progress
        progress = tracker.update([hand_obs(is_pinch=False)])
        self.assertAlmostEqual(progress, before)
        self.assertIsNone(tracker.anchor)


class ScanTrackerTests(unittest.TestCase):
    def still_pose(self) -> PoseObservation:
        return PoseObservation(nose_x=0.5, shoulder_center_x=0.5, shoulder_span=0.3, nose_y=0.4)

    def test_still_pose_completes_scan(self) -> None:
        tracker = ScanTracker()
        self.assertEqual(tracker.update(self.still_pose(), 0.0), 0.0)
        self.assertAlmostEqual(tracker.update(self.still_pose(), 1.0), 0.5)
        self.assertEqual(tracker.update(self.still_pose(), 2.1), 1.0)

    def test_movement_resets_scan(self) -> None:
        tracker = ScanTracker()
        tracker.update(self.still_pose(), 0.0)
        tracker.update(self.still_pose(), 1.5)
        moved = PoseObservation(nose_x=0.58, shoulder_center_x=0.5, shoulder_span=0.3, nose_y=0.4)
        self.assertEqual(tracker.update(moved, 1.6), 0.0)
        self.assertAlmostEqual(tracker.update(moved, 2.6), 0.5)

    def test_lost_pose_resets_scan(self) -> None:
        tracker = ScanTracker()
        tracker.update(self.still_pose(), 0.0)
        self.assertEqual(tracker.update(None, 1.0), 0.0)
        self.assertEqual(tracker.update(self.still_pose(), 1.1), 0.0)


class LegacyCampaignTests(unittest.TestCase):
    def test_campaign_sequence_lengths(self) -> None:
        self.assertEqual(len(build_campaign(5)), 5)
        self.assertEqual(len(build_campaign(10)), 10)

    def test_challenge_match_requires_two_hands(self) -> None:
        campaign = build_campaign(5)
        bunny_ears = campaign[2]

        self.assertFalse(challenge_matches(bunny_ears, ["open_palm"]))
        self.assertTrue(challenge_matches(bunny_ears, ["open_palm", "open_palm"]))

    def test_pose_matches_detects_lateral_dodge(self) -> None:
        campaign = build_campaign(5)
        neo_dodge = campaign[0]

        dodge_left = PoseObservation(nose_x=0.20, shoulder_center_x=0.52, shoulder_span=0.40)
        dodge_center = PoseObservation(nose_x=0.50, shoulder_center_x=0.52, shoulder_span=0.40)

        self.assertTrue(pose_matches(neo_dodge, dodge_left))
        self.assertFalse(pose_matches(neo_dodge, dodge_center))
