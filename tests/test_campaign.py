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
    round_duration=8, countdown=3, transition entre figures=1.2."""

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

        # 1. Neo Dodge
        event = engine.update([], dodge_pose(), 4.2, publish)
        self.assertEqual(event.score, 170)

        # 2. Pilule rouge : paume maintenue 0.5 s sur la zone (deadline 13.4)
        red_x, red_y = PILL_ZONES["red"]
        engine.update([palm(red_x, red_y)], None, 5.5, publish)
        event = engine.update([palm(red_x, red_y)], None, 6.0, publish)
        self.assertEqual(event.chosen_pill, "red")
        self.assertEqual(event.flash_message, "RED PILL ACCEPTED")
        self.assertEqual(event.score, 340)

        # 3. Bunny ears : 2 mains au-dessus du nez, 0.5 s (deadline 15.2)
        ears = [bunny(0.42, 0.25), bunny(0.58, 0.28)]
        engine.update(ears, still_pose(), 7.3, publish)
        event = engine.update(ears, still_pose(), 7.8, publish)
        self.assertEqual(event.score, 510)

        # 4. Cuillere : pince + rotation cumulee 0.9 rad > 45 deg (deadline 17.0)
        engine.update([pinch(0.0)], None, 9.1, publish)
        engine.update([pinch(0.45)], None, 9.2, publish)
        event = engine.update([pinch(0.9)], None, 9.3, publish)
        self.assertEqual(event.score, 680)
        self.assertEqual(event.flash_message, "THERE IS NO SPOON")

        # 5. Scan : immobile 2 s (deadline 18.5)
        engine.update([], still_pose(), 10.6, publish)
        event = engine.update([], still_pose(), 12.7, publish)

        self.assertEqual(event.phase, SCORE)
        self.assertEqual(event.score, 830)
        self.assertTrue(event.published)
        self.assertEqual([result.success for result in event.round_results], [True] * 5)
        # La pilule choisie par le joueur part en MQTT, une seule fois
        self.assertEqual(calls, ["red"])

        repeat = engine.update([], None, 13.0, publish)
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
