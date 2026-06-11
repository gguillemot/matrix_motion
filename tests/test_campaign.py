from __future__ import annotations

import unittest

from src.challenges import PILL_ZONES, HandObservation, PoseObservation, build_breizhcamp_campaign
from src.game_engine import BLUE_ENDING, COUNTDOWN, IN_ROUND, PILL_CHOICE, SCORE, GameEngine


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
    """Deroule une partie complete : attract -> choix pilule rouge -> les 4
    epreuves en ordre canonique (neo_dodge, white_rabbit, no_spoon,
    smith_scan). round_duration=8, countdown=3, transition=2.4,
    maintiens : pilule 0.9 s, dodge 0.5 s, bunny 0.9 s, cuillere 80 deg."""

    def test_red_pill_full_run(self) -> None:
        calls: list[str] = []

        def publish(pill: str) -> bool:
            calls.append(pill)
            return True

        engine = GameEngine(
            round_duration=8.0,
            countdown_duration=3.0,
            campaign=build_breizhcamp_campaign(shuffle=False),
        )

        # Attract : 2 paumes maintenues 1 s -> choix des pilules
        engine.update([palm(), palm()], None, 0.0, publish)
        event = engine.update([palm(), palm()], None, 1.05, publish)
        self.assertEqual(event.phase, PILL_CHOICE)

        # Pilule rouge : paume maintenue 0.95 s -> publication MQTT immediate
        red_x, red_y = PILL_ZONES["red"]
        engine.update([palm(red_x, red_y)], None, 1.2, publish)
        event = engine.update([palm(red_x, red_y)], None, 2.15, publish)
        self.assertEqual(event.phase, COUNTDOWN)
        self.assertEqual(calls, ["red"])

        # Fin du decompte -> round 1 (deadline 13.2)
        event = engine.update([], None, 5.2, publish)
        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.challenge_key, "neo_dodge")

        # 1. Neo Dodge : posture tenue 0.55 s
        engine.update([], dodge_pose(), 5.3, publish)
        event = engine.update([], dodge_pose(), 5.85, publish)
        # (50 + int(50 * 7.35/8) = 45) x combo 1 = 95
        self.assertEqual(event.score, 95)
        self.assertEqual(event.celebration_key, "neo_dodge")

        # 2. Bunny ears : 2 mains au-dessus du nez, 0.95 s (deadline 16.25)
        ears = [bunny(0.42, 0.25), bunny(0.58, 0.28)]
        engine.update(ears, still_pose(), 8.3, publish)
        event = engine.update(ears, still_pose(), 9.25, publish)
        # (50 + int(50 * 7.0/8) = 43) x combo 2 = 186 -> total 281
        self.assertEqual(event.score, 281)
        self.assertIn("COMBO x2", event.flash_message)

        # 3. Cuillere : pince + rotation cumulee 1.5 rad > 80 deg (deadline 19.65)
        engine.update([pinch(0.0)], None, 11.7, publish)
        engine.update([pinch(0.5)], None, 11.8, publish)
        engine.update([pinch(1.0)], None, 11.9, publish)
        event = engine.update([pinch(1.5)], None, 12.0, publish)
        # (50 + int(50 * 7.65/8) = 47) x combo 3 = 291 -> total 572
        self.assertEqual(event.score, 572)
        self.assertIn("THERE IS NO SPOON", event.flash_message)

        # 4. Scan : immobile 2.1 s (deadline 22.4)
        engine.update([], still_pose(), 14.5, publish)
        event = engine.update([], still_pose(), 16.6, publish)

        self.assertEqual(event.phase, SCORE)
        # (50 + int(50 * 5.8/8) = 36) x combo 4 = 344 -> total 916
        self.assertEqual(event.score, 916)
        self.assertTrue(event.new_record)
        self.assertEqual([result.success for result in event.round_results], [True] * 4)
        # Pas de seconde publication au score : la pilule part au choix
        self.assertEqual(calls, ["red"])

        repeat = engine.update([], None, 17.0, publish)
        self.assertEqual(repeat.phase, SCORE)
        self.assertEqual(calls, ["red"])

    def test_blue_pill_skips_the_campaign(self) -> None:
        calls: list[str] = []
        engine = GameEngine(round_duration=8.0, countdown_duration=3.0)
        engine.start_game(0.0)

        blue_x, blue_y = PILL_ZONES["blue"]
        engine.update([palm(blue_x, blue_y)], None, 0.1, lambda p: calls.append(p) or True)
        event = engine.update([palm(blue_x, blue_y)], None, 1.05, lambda p: calls.append(p) or True)

        self.assertEqual(event.phase, BLUE_ENDING)
        self.assertEqual(calls, ["blue"])
        self.assertEqual(event.score, 0)
        self.assertEqual(event.round_results, [])

    def test_restart_after_score_shuffles_new_campaign(self) -> None:
        engine = GameEngine(round_duration=8.0, countdown_duration=3.0)
        engine.state.phase = SCORE

        engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 1.05, lambda pill: True)

        self.assertEqual(event.phase, PILL_CHOICE)
        self.assertEqual(event.round_total, 4)
        self.assertEqual(event.score, 0)
        self.assertEqual(event.round_results, [])
