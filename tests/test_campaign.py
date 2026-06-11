from __future__ import annotations

import unittest
from typing import cast

from src.challenges import PoseObservation
from src.game_engine import GAME_OVER, IDLE, IN_ROUND, VICTORY, GameEngine, GameEvent, GameState


class CampaignTests(unittest.TestCase):
    def test_start_gesture_enters_round_one(self) -> None:
        engine = GameEngine(sequence_length=5, state=GameState())

        first = engine.update(["open_palm", "open_palm"], 0.0, lambda pill: True)
        second = engine.update(["open_palm", "open_palm"], 1.05, lambda pill: True)

        self.assertEqual(first.phase, IDLE)
        self.assertEqual(second.phase, IN_ROUND)
        self.assertEqual(second.round_index, 1)

    def test_victory_publishes_once(self) -> None:
        calls: list[str] = []
        engine = GameEngine(sequence_length=5, state=GameState())
        engine.update(["open_palm", "open_palm"], 0.0, lambda pill: True)
        engine.update(["open_palm", "open_palm"], 1.05, lambda pill: True)

        steps = [
            (([], PoseObservation(nose_x=0.18, shoulder_center_x=0.52, shoulder_span=0.34)), 2.15),
            ((["point"], None), 3.25),
            ((["open_palm", "open_palm"], None), 4.35),
            ((["open_palm"], None), 5.45),
            ((["fist"], None), 6.55),
        ]
        event = None
        for (gestures, pose_observation), now in steps:
            event = engine.update_with_pose(gestures, pose_observation, now, lambda pill: calls.append(pill) or True)

        self.assertIsNotNone(event)
        event = cast(GameEvent, event)
        self.assertEqual(event.phase, VICTORY)
        self.assertEqual(calls, ["blue"])

        repeat = engine.update_with_pose(["open_palm", "open_palm"], None, 6.55, lambda pill: calls.append(pill) or True)
        self.assertEqual(repeat.phase, VICTORY)
        self.assertEqual(calls, ["blue"])
        self.assertTrue(engine.state.victory_mqtt_sent)

    def test_timeout_goes_game_over(self) -> None:
        engine = GameEngine(sequence_length=5, state=GameState())
        engine.start_campaign(0.0)
        engine.state.round_deadline = 0.5

        event = engine.update([], 0.8, lambda pill: True)

        self.assertEqual(event.phase, GAME_OVER)
        self.assertEqual(engine.state.phase, GAME_OVER)

    def test_reset_after_victory_restarts_campaign(self) -> None:
        engine = GameEngine(sequence_length=5, state=GameState())
        engine.start_campaign(0.0)
        engine.state.phase = VICTORY

        event = engine.update(["open_palm", "open_palm"], 0.0, lambda pill: True)
        restarted = engine.update(["open_palm", "open_palm"], 1.05, lambda pill: True)

        self.assertEqual(event.phase, VICTORY)
        self.assertEqual(restarted.phase, IN_ROUND)
