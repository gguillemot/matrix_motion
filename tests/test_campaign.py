from __future__ import annotations

import unittest

from src.challenges import PILL_ZONES, HandObservation, PoseObservation, build_breizhcamp_campaign
from src.game_engine import COUNTDOWN, IN_ROUND, SCORE, GameEngine


def palm(x: float = 0.5, y: float = 0.5) -> HandObservation:
    return HandObservation(gesture="open_palm", palm_x=x, palm_y=y, is_pinch=False, hand_angle=0.0)


def bunny(x: float, y: float) -> HandObservation:
    return HandObservation(gesture="bunny_ears", palm_x=x, palm_y=y, is_pinch=False, hand_angle=0.0)


def pinch(angle: float) -> HandObservation:
    return HandObservation(gesture="none", palm_x=0.5, palm_y=0.6, is_pinch=True, hand_angle=angle)


def still_pose() -> PoseObservation:
    return PoseObservation(nose_x=0.5, shoulder_center_x=0.5, shoulder_span=0.35, nose_y=0.42)


def dodge_pose() -> PoseObservation:
    return PoseObservation(nose_x=0.20, shoulder_center_x=0.52, shoulder_span=0.40, nose_y=0.42)


class FullPlaythroughTests(unittest.TestCase):
    """Deroule une partie complete en ordre canonique :
    neo_dodge -> pill_choice -> white_rabbit -> no_spoon -> smith_scan.
    round_duration=8, countdown=3, transition entre figures=2.4,
    maintiens : dodge 0.5 s, pilule/bunny 0.9 s, cuillere 80 deg."""

    def test_full_run_publishes_chosen_pill_once(self) -> None:
        calls: list[str] = []

        def publish(pill: str) -> bool:
            calls.append(pill)
            return True

        engine = GameEngine(
            round_duration=8.0,
            countdown_duration=3.0,
            campaign=build_breizhcamp_campaign(shuffle=False),
        )

        # Attract : 2 paumes maintenues 1 s
        engine.update([palm(), palm()], None, 0.0, publish)
        event = engine.update([palm(), palm()], None, 1.05, publish)
        self.assertEqual(event.phase, COUNTDOWN)

        # Fin du decompte -> round 1 (deadline 12.1)
        event = engine.update([], None, 4.1, publish)
        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.challenge_key, "neo_dodge")

        # 1. Neo Dodge : posture tenue 0.55 s
        engine.update([], dodge_pose(), 4.2, publish)
        event = engine.update([], dodge_pose(), 4.75, publish)
        # (50 + int(50 * 7.35/8) = 45) x combo 1 = 95
        self.assertEqual(event.score, 95)
        self.assertEqual(event.celebration_key, "neo_dodge")

        # 2. Pilule rouge : paume maintenue 0.95 s (transition finit a 7.15, deadline 15.15)
        red_x, red_y = PILL_ZONES["red"]
        engine.update([palm(red_x, red_y)], None, 7.3, publish)
        event = engine.update([palm(red_x, red_y)], None, 8.25, publish)
        self.assertEqual(event.chosen_pill, "red")
        self.assertIn("RED PILL ACCEPTED", event.flash_message)
        self.assertIn("COMBO x2", event.flash_message)
        # (50 + int(50 * 6.9/8) = 43) x combo 2 = 186 -> total 281
        self.assertEqual(event.score, 281)

        # 3. Bunny ears : 2 mains au-dessus du nez, 0.95 s (deadline 18.65)
        ears = [bunny(0.42, 0.25), bunny(0.58, 0.28)]
        engine.update(ears, still_pose(), 10.7, publish)
        event = engine.update(ears, still_pose(), 11.65, publish)
        # (50 + 43) x combo 3 = 279 -> total 560
        self.assertEqual(event.score, 560)
        self.assertEqual(event.combo, 3)

        # 4. Cuillere : pince + rotation cumulee 1.5 rad > 80 deg (deadline 22.05)
        engine.update([pinch(0.0)], None, 14.1, publish)
        engine.update([pinch(0.5)], None, 14.2, publish)
        engine.update([pinch(1.0)], None, 14.3, publish)
        event = engine.update([pinch(1.5)], None, 14.4, publish)
        # (50 + int(50 * 7.65/8) = 47) x combo 4 = 388 -> total 948
        self.assertEqual(event.score, 948)
        self.assertIn("THERE IS NO SPOON", event.flash_message)

        # 5. Scan : immobile 2.1 s (deadline 24.8)
        engine.update([], still_pose(), 16.9, publish)
        event = engine.update([], still_pose(), 19.0, publish)

        self.assertEqual(event.phase, SCORE)
        # (50 + int(50 * 5.8/8) = 36) x combo 5 = 430 -> total 1378
        self.assertEqual(event.score, 1378)
        self.assertTrue(event.published)
        self.assertTrue(event.new_record)
        self.assertEqual([result.success for result in event.round_results], [True] * 5)
        # La pilule choisie par le joueur part en MQTT, une seule fois
        self.assertEqual(calls, ["red"])

        repeat = engine.update([], None, 19.5, publish)
        self.assertEqual(repeat.phase, SCORE)
        self.assertEqual(calls, ["red"])

    def test_restart_after_score_shuffles_new_campaign(self) -> None:
        engine = GameEngine(round_duration=8.0, countdown_duration=3.0)
        engine.state.phase = SCORE

        engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 1.05, lambda pill: True)

        self.assertEqual(event.phase, COUNTDOWN)
        self.assertEqual(event.round_total, 5)
        self.assertEqual(event.score, 0)
        self.assertEqual(event.round_results, [])
