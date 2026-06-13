from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.challenges import (
    PILL_ZONES,
    HandObservation,
    PoseObservation,
    build_breizhcamp_campaign,
)
from src.game_engine import (
    ATTRACT,
    BLUE_ENDING,
    COUNTDOWN,
    IN_ROUND,
    INTRO,
    PILL_CHOICE,
    SCORE,
    SCORE_HOLD_SEC,
    TRINITY_OUTRO,
    TRINITY_OUTRO_DURATION,
    GameEngine,
)


def palm(x: float = 0.5, y: float = 0.5) -> HandObservation:
    return HandObservation(
        gesture="open_palm", palm_x=x, palm_y=y, is_pinch=False, hand_angle=0.0
    )


def dodge_pose() -> PoseObservation:
    return PoseObservation(
        nose_x=0.20, shoulder_center_x=0.52, shoulder_span=0.40, nose_y=0.4
    )


def make_engine(**kwargs) -> GameEngine:
    return GameEngine(
        round_duration=8.0,
        countdown_duration=3.0,
        campaign=build_breizhcamp_campaign(shuffle=False),
        **kwargs,
    )


def choose_pill(engine: GameEngine, pill: str, publish=lambda p: True):
    """Depuis PILL_CHOICE (start_game a t=0) : paume maintenue 0.95 s sur la pilule."""
    zone_x, zone_y = PILL_ZONES[pill]
    engine.update([palm(zone_x, zone_y)], None, 0.1, publish)
    return engine.update([palm(zone_x, zone_y)], None, 1.05, publish)


def start_red_game(engine: GameEngine, publish=lambda p: True):
    """Partie lancee + pilule rouge : premier round a 4.1 s (deadline 12.1)."""
    engine.start_game(0.0)
    choose_pill(engine, "red", publish)  # COUNTDOWN jusqu'a 4.05
    return engine.update([], None, 4.1, publish)


def succeed_dodge(engine: GameEngine, hold_start: float):
    """Tient la posture d'esquive DODGE_HOLD_SEC pour valider la figure 1."""
    engine.update([], dodge_pose(), hold_start, lambda pill: True)
    return engine.update([], dodge_pose(), hold_start + 0.55, lambda pill: True)


class GameEngineTests(unittest.TestCase):
    def test_starts_in_attract(self) -> None:
        engine = make_engine()
        event = engine.update([], None, 0.0, lambda pill: True)

        self.assertEqual(event.phase, ATTRACT)
        self.assertEqual(event.round_total, 4)
        self.assertTrue(event.rain_boost)

    def test_palm_hold_opens_pill_choice(self) -> None:
        engine = make_engine()

        first = engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        midway = engine.update([palm(), palm()], None, 0.5, lambda pill: True)
        second = engine.update([palm(), palm()], None, 1.05, lambda pill: True)

        self.assertEqual(first.phase, ATTRACT)
        self.assertAlmostEqual(midway.start_hold_progress, 0.5)
        # Sans clip d'intro (defaut), on arrive directement au choix des pilules
        self.assertEqual(second.phase, PILL_CHOICE)

    def test_intro_plays_before_pill_choice(self) -> None:
        engine = make_engine(has_intro=True)
        engine.update([palm(), palm()], None, 0.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 1.05, lambda pill: True)
        self.assertEqual(event.phase, INTRO)

        engine.finish_intro(15.0)
        after = engine.update([], None, 15.1, lambda pill: True)
        self.assertEqual(after.phase, PILL_CHOICE)

    def test_red_pill_starts_countdown_and_publishes(self) -> None:
        published: list[str] = []
        engine = make_engine()
        engine.start_game(0.0)

        event = choose_pill(engine, "red", lambda p: published.append(p) or True)

        self.assertEqual(event.phase, COUNTDOWN)
        self.assertEqual(event.chosen_pill, "red")
        self.assertTrue(event.published)
        self.assertEqual(published, ["red"])
        self.assertEqual(event.flash_message, "RED PILL ACCEPTED")

    def test_blue_pill_returns_to_reality(self) -> None:
        published: list[str] = []
        engine = make_engine()
        engine.start_game(0.0)

        event = choose_pill(engine, "blue", lambda p: published.append(p) or True)

        self.assertEqual(event.phase, BLUE_ENDING)
        self.assertEqual(event.chosen_pill, "blue")
        self.assertEqual(published, ["blue"])
        self.assertFalse(event.rain_boost)

    def test_pill_choice_idle_returns_to_attract(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)

        event = engine.update([], None, 30.1, lambda pill: True)

        self.assertEqual(event.phase, ATTRACT)

    def test_blue_ending_replay_starts_new_game(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        choose_pill(engine, "blue")

        engine.update([palm(), palm()], None, 2.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 3.05, lambda pill: True)

        self.assertEqual(event.phase, PILL_CHOICE)
        self.assertIsNone(event.chosen_pill)

    def test_blue_ending_idle_returns_to_attract(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        choose_pill(engine, "blue")  # BLUE_ENDING a 1.05

        event = engine.update([], None, 1.05 + 25.1, lambda pill: True)

        self.assertEqual(event.phase, ATTRACT)

    def test_countdown_reaches_first_round(self) -> None:
        engine = make_engine()
        event = start_red_game(engine)

        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.round_index, 1)
        self.assertEqual(event.challenge_key, "neo_dodge")

    def test_timeout_skips_to_next_round_with_zero_points(self) -> None:
        engine = make_engine()
        start_red_game(engine)

        event = engine.update([], None, 12.2, lambda pill: True)

        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.round_index, 2)
        self.assertEqual(event.score, 0)
        self.assertEqual(event.flash_message, "TOO SLOW...")
        self.assertFalse(event.round_results[0].success)

    def test_all_timeouts_end_on_score_screen(self) -> None:
        published: list[str] = []
        publish = lambda pill: published.append(pill) or True  # noqa: E731
        engine = make_engine()
        start_red_game(engine, publish)

        for _ in range(4):
            now = engine.state.round_deadline + 0.1
            event = engine.update([], None, now, publish)

        self.assertEqual(event.phase, SCORE)
        self.assertEqual(event.score, 0)
        self.assertEqual(len(event.round_results), 4)
        self.assertFalse(event.new_record)  # score nul ne bat pas de record
        # La seule publication MQTT est celle du choix de pilule
        self.assertEqual(published, ["red"])

    def test_dodge_requires_holding_the_pose(self) -> None:
        engine = make_engine()
        start_red_game(engine)

        instant = engine.update([], dodge_pose(), 4.2, lambda pill: True)
        midway = engine.update([], dodge_pose(), 4.5, lambda pill: True)
        held = engine.update([], dodge_pose(), 4.75, lambda pill: True)

        self.assertEqual(instant.score, 0)
        self.assertEqual(midway.score, 0)
        self.assertGreater(midway.figure_progress, 0.0)
        self.assertGreater(held.score, 0)

    def test_dodge_success_scores_with_speed_bonus(self) -> None:
        engine = make_engine()
        start_red_game(engine)  # round 1 = neo_dodge, deadline 12.1

        event = succeed_dodge(engine, 4.2)

        # (50 base + bonus vitesse int(50 * 7.35/8) = 45) x combo 1 = 95
        self.assertEqual(event.score, 95)
        self.assertEqual(event.combo, 1)
        self.assertTrue(event.round_results[0].success)
        self.assertEqual(event.flash_message, "DODGE SUCCESSFUL !")
        self.assertEqual(event.round_index, 2)

    def test_combo_resets_after_timeout(self) -> None:
        engine = make_engine()
        start_red_game(engine)
        first = succeed_dodge(engine, 4.2)
        self.assertEqual(first.combo, 1)

        event = engine.update(
            [], None, engine.state.round_deadline + 0.1, lambda pill: True
        )

        self.assertEqual(event.combo, 0)

    def test_celebration_runs_during_transition_then_stops(self) -> None:
        engine = make_engine()
        start_red_game(engine)
        # Neo Dodge : succes a 4.75, fenetre BULLET_TIME_REPLAY_SEC=4.5 s -> jusqu'a 9.25
        success = succeed_dodge(engine, 4.2)

        during = engine.update([], None, 5.7, lambda pill: True)
        after = engine.update([], None, 9.3, lambda pill: True)

        self.assertEqual(success.celebration_key, "neo_dodge")
        self.assertEqual(during.celebration_key, "neo_dodge")
        self.assertGreater(during.celebration_progress, 0.0)
        self.assertEqual(after.celebration_key, "")

    def test_detection_paused_during_round_transition(self) -> None:
        engine = make_engine()
        start_red_game(engine)
        # Neo Dodge : fenetre BULLET_TIME_REPLAY_SEC=4.5 s -> transition jusqu'a 9.25
        succeed_dodge(engine, 4.2)

        event = engine.update([], dodge_pose(), 5.7, lambda pill: True)

        self.assertEqual(event.score, 95)
        self.assertEqual(len(event.round_results), 1)

    def test_best_score_persists_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "highscore.json"
            single_dodge = [build_breizhcamp_campaign(shuffle=False)[0]]

            engine = GameEngine(
                round_duration=8.0,
                countdown_duration=3.0,
                campaign=single_dodge,
                best_score_path=path,
            )
            start_red_game(engine)
            event = succeed_dodge(engine, 4.2)

            self.assertEqual(event.phase, SCORE)
            self.assertTrue(event.new_record)
            self.assertEqual(event.best_score, 95)
            self.assertTrue(path.exists())

            reloaded = GameEngine(campaign=single_dodge, best_score_path=path)
            self.assertEqual(reloaded.best_score, 95)

    def test_restart_from_score_screen(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.state.phase = SCORE
        engine.state.phase_entered_at = 10.0  # ecran de score affiche a t=10
        engine.state.score = 300

        # Rejouer reste possible pendant la fenetre de score (avant l'outro).
        engine.update([palm(), palm()], None, 10.0, lambda pill: True)
        event = engine.update([palm(), palm()], None, 11.05, lambda pill: True)

        self.assertEqual(event.phase, PILL_CHOICE)
        self.assertEqual(event.score, 0)

    def test_score_advances_to_trinity_outro(self) -> None:
        engine = make_engine()
        engine.start_game(0.0)
        engine.state.phase = SCORE
        engine.state.phase_entered_at = 10.0

        before = engine.update([], None, 10.0 + SCORE_HOLD_SEC - 0.1, lambda pill: True)
        after = engine.update([], None, 10.0 + SCORE_HOLD_SEC + 0.1, lambda pill: True)

        self.assertEqual(before.phase, SCORE)
        self.assertEqual(after.phase, TRINITY_OUTRO)

    def test_score_hold_waits_for_final_celebration(self) -> None:
        # Une celebration finale qui deborde sur l'ecran de score retarde l'outro :
        # le compte des SCORE_HOLD_SEC ne demarre qu'apres celebration_until.
        engine = make_engine()
        engine.start_game(0.0)
        engine.state.phase = SCORE
        engine.state.phase_entered_at = 10.0
        engine.state.celebration_until = 20.0  # celebration finale jusqu'a t=20

        during = engine.update([], None, 19.9, lambda pill: True)  # celeb pas finie
        just_after = engine.update(
            [], None, 20.0 + SCORE_HOLD_SEC - 0.1, lambda pill: True
        )
        done = engine.update([], None, 20.0 + SCORE_HOLD_SEC + 0.1, lambda pill: True)

        self.assertEqual(during.phase, SCORE)
        self.assertEqual(just_after.phase, SCORE)
        self.assertEqual(done.phase, TRINITY_OUTRO)

    def test_trinity_outro_returns_to_attract(self) -> None:
        engine = make_engine()
        engine.state.phase = TRINITY_OUTRO
        engine.state.phase_entered_at = 20.0

        still = engine.update(
            [], None, 20.0 + TRINITY_OUTRO_DURATION - 0.1, lambda pill: True
        )
        done = engine.update(
            [], None, 20.0 + TRINITY_OUTRO_DURATION + 0.1, lambda pill: True
        )

        self.assertEqual(still.phase, TRINITY_OUTRO)
        self.assertEqual(done.phase, ATTRACT)

    def test_skip_outro_returns_to_attract(self) -> None:
        engine = make_engine()
        engine.state.phase = TRINITY_OUTRO
        engine.state.phase_entered_at = 20.0

        engine.skip_outro(21.0)

        self.assertEqual(engine.state.phase, ATTRACT)
