from __future__ import annotations

import unittest

from src.challenges import HandObservation, PoseObservation, build_breizhcamp_campaign
from src.game_engine import ATTRACT, COUNTDOWN, IN_ROUND, SCORE, GameEngine


def palm() -> HandObservation:
    return HandObservation(gesture="open_palm", palm_x=0.5, palm_y=0.5, is_pinch=False, hand_angle=0.0)


def dodge_pose() -> PoseObservation:
    return PoseObservation(nose_x=0.20, shoulder_center_x=0.52, shoulder_span=0.40, nose_y=0.4)


def make_engine() -> GameEngine:
    return GameEngine(
        round_duration=8.0,
        countdown_duration=3.0,
        campaign=build_breizhcamp_campaign(shuffle=False),
    )


class GameEngineTests(unittest.TestCase):
    def test_starts_in_attract(self) -> None:
        engine = make_engine()
        event = engine.update([], None, 0.0, lambda pill: True)

        self.assertEqual(event.phase, ATTRACT)
        self.assertEqual(event.round_total, 5)
        self.assertTrue(event.rain_boost)

    def test_palm_hold_starts_countdown(self) -> None:
        engine = make_engine()

        first = engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        midway = engine.update([palm(), palm()], None, 0.5, lambda pill: True)
        second = engine.update([palm(), palm()], None, 1.05, lambda pill: True)

        self.assertEqual(first.phase, ATTRACT)
        self.assertAlmostEqual(midway.start_hold_progress, 0.5)
        self.assertEqual(second.phase, COUNTDOWN)

    def test_releasing_palms_resets_hold(self) -> None:
        engine = make_engine()
        engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        engine.update([palm()], None, 0.5, lambda pill: True)
        event = engine.update([palm(), palm()], None, 1.2, lambda pill: True)

        self.assertEqual(event.phase, ATTRACT)

    def test_countdown_reaches_first_round(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)

        during = engine.update([], None, 1.5, lambda pill: True)
        after = engine.update([], None, 3.1, lambda pill: True)

        self.assertEqual(during.phase, COUNTDOWN)
        self.assertEqual(during.countdown_value, 2)
        self.assertEqual(after.phase, IN_ROUND)
        self.assertEqual(after.round_index, 1)
        self.assertEqual(after.challenge_key, "neo_dodge")

    def test_timeout_skips_to_next_round_with_zero_points(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.update([], None, 3.1, lambda pill: True)

        event = engine.update([], None, 12.0, lambda pill: True)

        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.round_index, 2)
        self.assertEqual(event.score, 0)
        self.assertEqual(event.flash_message, "TOO SLOW...")
        self.assertFalse(event.round_results[0].success)

    def test_all_timeouts_end_on_score_screen(self) -> None:
        published: list[str] = []
        engine = make_engine()
        engine.start_game(0.0)
        engine.update([], None, 3.1, lambda pill: True)

        now = 3.1
        for _ in range(5):
            now = engine.state.round_deadline + 0.1
            event = engine.update([], None, now, lambda pill: published.append(pill) or True)

        self.assertEqual(event.phase, SCORE)
        self.assertEqual(event.score, 0)
        self.assertEqual(len(event.round_results), 5)
        # Pilule par defaut publiee meme sans figure pilule reussie
        self.assertEqual(published, ["blue"])

    def test_dodge_success_scores_with_time_bonus(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.update([], None, 3.1, lambda pill: True)  # round 1 = neo_dodge, deadline 11.1

        event = engine.update([], dodge_pose(), 4.0, lambda pill: True)

        # 100 pts + 10 pts par seconde restante (7.1 s -> 7)
        self.assertEqual(event.score, 170)
        self.assertTrue(event.round_results[0].success)
        self.assertEqual(event.flash_message, "DODGE SUCCESSFUL !")
        self.assertEqual(event.round_index, 2)

    def test_detection_paused_during_round_transition(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.update([], None, 3.1, lambda pill: True)
        engine.update([], dodge_pose(), 4.0, lambda pill: True)  # transition jusqu'a 5.2

        event = engine.update([], dodge_pose(), 4.5, lambda pill: True)

        self.assertEqual(event.score, 170)
        self.assertEqual(len(event.round_results), 1)

    def test_restart_from_score_screen(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.state.phase = SCORE
        engine.state.score = 300

        engine.update([palm(), palm()], None, 10.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 11.05, lambda pill: True)

        self.assertEqual(event.phase, COUNTDOWN)
        self.assertEqual(event.score, 0)
