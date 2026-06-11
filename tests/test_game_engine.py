from __future__ import annotations

import unittest

from src.game_engine import GameEngine, GameState


class GameEngineTests(unittest.TestCase):
    def test_thumbs_up_publishes_blue_and_sets_status(self) -> None:
        calls: list[str] = []
        engine = GameEngine(GameState(status="SYSTEM ONLINE", status_until=10.0))

        event = engine.process_gesture("thumbs_up", 12.0, lambda pill: calls.append(pill) or True)

        self.assertEqual(calls, ["blue"])
        self.assertTrue(event.action_taken)
        self.assertEqual(event.pill, "blue")
        self.assertTrue(event.published)
        self.assertEqual(engine.state.status, "PILULE BLEUE")
        self.assertEqual(engine.state.last_action_ts, 12.0)
        self.assertGreater(engine.state.status_until, 12.0)

    def test_cooldown_blocks_repeat_action(self) -> None:
        calls: list[str] = []
        engine = GameEngine(GameState(status="SYSTEM ONLINE", status_until=10.0, last_action_ts=5.0))

        event = engine.process_gesture("ok_sign", 6.0, lambda pill: calls.append(pill) or True)

        self.assertFalse(event.action_taken)
        self.assertEqual(calls, [])

    def test_tick_clears_expired_status(self) -> None:
        engine = GameEngine(GameState(status="PILULE ROUGE", status_until=5.0))

        engine.tick(6.0)

        self.assertEqual(engine.state.status, "")

    def test_open_palm_toggles_boost(self) -> None:
        engine = GameEngine(GameState(status="SYSTEM ONLINE", status_until=1.0))

        event = engine.process_gesture("open_palm", 2.0, lambda pill: True)

        self.assertTrue(event.action_taken)
        self.assertTrue(engine.state.rain_boost)
        self.assertEqual(engine.state.status, "MATRIX BOOST ON")
