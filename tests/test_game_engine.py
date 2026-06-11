from __future__ import annotations

import unittest

from src.game_engine import GAME_OVER, IDLE, IN_ROUND, GameEngine, GameState


class GameEngineTests(unittest.TestCase):
    def test_constructor_accepts_state_first_for_compatibility(self) -> None:
        engine = GameEngine(GameState(status="SYSTEM ONLINE", status_until=10.0))

        self.assertEqual(engine.state.status, "SYSTEM ONLINE")
        self.assertEqual(len(engine.challenges), 5)

    def test_process_gesture_keeps_single_gesture_api(self) -> None:
        engine = GameEngine(5)

        event = engine.process_gesture("point", 0.0, lambda pill: True)

        self.assertEqual(event.phase, IDLE)
        self.assertEqual(event.round_total, 5)

    def test_timeout_changes_to_game_over(self) -> None:
        engine = GameEngine(5)
        engine.start_campaign(0.0)
        engine.state.round_deadline = 0.1

        event = engine.update([], 0.2, lambda pill: True)

        self.assertEqual(event.phase, GAME_OVER)
        self.assertEqual(engine.state.phase, GAME_OVER)

    def test_round_starts_after_hold(self) -> None:
        engine = GameEngine(5)

        first = engine.update(["open_palm", "open_palm"], 0.0, lambda pill: True)
        second = engine.update(["open_palm", "open_palm"], 1.05, lambda pill: True)

        self.assertEqual(first.phase, IDLE)
        self.assertEqual(second.phase, IN_ROUND)
        self.assertEqual(second.round_index, 1)
