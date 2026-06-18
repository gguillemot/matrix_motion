from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from src.challenges import (
    HandObservation,
    QuizQuestion,
    QuizRuntime,
    build_breizhcamp_campaign,
    load_quiz_questions,
    make_quiz_challenge,
    quiz_answer_zones,
    select_quiz_questions,
)
from src.game_engine import IN_ROUND, SCORE, GameEngine
from src.challenges import PILL_ZONES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def two_answer_question(correct: int = 1) -> QuizQuestion:
    return QuizQuestion(
        id="t",
        question="Question ?",
        answers=["Reponse A", "Reponse B"],
        correct_index=correct,
        difficulty="easy",
        explanation="Parce que.",
    )


def point_at(zone: tuple[float, float, float, float]) -> HandObservation:
    """Main dont l'index pointe le centre d'une carte de reponse."""
    cx = (zone[0] + zone[2]) / 2.0
    cy = (zone[1] + zone[3]) / 2.0
    return HandObservation(
        gesture="point",
        palm_x=cx,
        palm_y=cy,
        is_pinch=False,
        hand_angle=0.0,
        index_x=cx,
        index_y=cy,
    )


def palm(x: float = 0.5, y: float = 0.5) -> HandObservation:
    return HandObservation(
        gesture="open_palm", palm_x=x, palm_y=y, is_pinch=False, hand_angle=0.0
    )


def make_runtime(question: QuizQuestion) -> QuizRuntime:
    return QuizRuntime(
        question,
        duration_s=10.0,
        intro_sec=1.0,
        reveal_sec=1.0,
        dwell_sec=1.0,
        grace_sec=0.3,
        zone_margin=0.06,
    )


# ---------------------------------------------------------------------------
# Chargement tolerant de la banque de questions
# ---------------------------------------------------------------------------


class LoaderTests(unittest.TestCase):
    def _write(self, payload) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "questions.json"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        return tmp

    def test_keeps_valid_filters_invalid(self) -> None:
        path = self._write(
            {
                "questions": [
                    {
                        "id": "ok",
                        "question": "Q ?",
                        "answers": ["a", "b"],
                        "correct_index": 1,
                    },
                    {"id": "no_answers", "question": "Q ?", "correct_index": 0},
                    {
                        "id": "bad_index",
                        "question": "Q ?",
                        "answers": ["a", "b"],
                        "correct_index": 5,
                    },
                    {
                        "id": "too_few",
                        "question": "Q ?",
                        "answers": ["a"],
                        "correct_index": 0,
                    },
                ]
            }
        )
        questions = load_quiz_questions(path)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].id, "ok")
        self.assertEqual(questions[0].correct_index, 1)

    def test_accepts_bare_list(self) -> None:
        path = self._write(
            [{"id": "x", "question": "Q ?", "answers": ["a", "b"], "correct_index": 0}]
        )
        questions = load_quiz_questions(path)
        self.assertEqual(len(questions), 1)

    def test_missing_file_falls_back(self) -> None:
        questions = load_quiz_questions(Path("/nope/does/not/exist.json"))
        self.assertTrue(questions)  # fallback integre non vide

    def test_malformed_json_falls_back(self) -> None:
        tmp = Path(tempfile.mkdtemp()) / "bad.json"
        tmp.write_text("{ this is not json", encoding="utf-8")
        questions = load_quiz_questions(tmp)
        self.assertTrue(questions)

    def test_clamps_difficulty_and_duration(self) -> None:
        path = self._write(
            [
                {
                    "id": "x",
                    "question": "Q ?",
                    "answers": ["a", "b", "c"],
                    "correct_index": 2,
                    "difficulty": "impossible",
                    "duration_s": -3,
                }
            ]
        )
        q = load_quiz_questions(path)[0]
        self.assertEqual(q.difficulty, "medium")
        self.assertIsNone(q.duration_s)


# ---------------------------------------------------------------------------
# Zones de reponses
# ---------------------------------------------------------------------------


class ZoneTests(unittest.TestCase):
    def test_two_answers_one_row(self) -> None:
        zones = quiz_answer_zones(2)
        self.assertEqual(len(zones), 2)
        # Meme bande verticale (une seule ligne).
        self.assertAlmostEqual(zones[0][1], zones[1][1])

    def test_four_answers_grid(self) -> None:
        zones = quiz_answer_zones(4)
        self.assertEqual(len(zones), 4)
        # 2 lignes : la 3e carte est plus basse que la 1ere.
        self.assertGreater(zones[2][1], zones[0][1])

    def test_zones_stay_normalized(self) -> None:
        for zone in quiz_answer_zones(4):
            for v in zone:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


# ---------------------------------------------------------------------------
# Machine a etats QuizRuntime
# ---------------------------------------------------------------------------


class QuizRuntimeTests(unittest.TestCase):
    def test_intro_freezes_then_activates(self) -> None:
        rt = make_runtime(two_answer_question())
        rt.update([], 0.0)
        self.assertEqual(rt.substate, "intro")
        self.assertEqual(rt.timer_left, 10.0)
        rt.update([], 0.5)
        self.assertEqual(rt.substate, "intro")  # toujours fige
        rt.update([], 1.1)
        self.assertEqual(rt.substate, "active")

    def test_dwell_selects_correct(self) -> None:
        question = two_answer_question(correct=1)
        rt = make_runtime(question)
        hand = point_at(rt.zones[1])
        rt.update([], 0.0)  # intro
        rt.update([hand], 1.1)  # active
        rt.update([hand], 1.5)  # apres grace : la jauge demarre
        rt.update([hand], 2.6)  # dwell atteint 1.0 -> verrou
        self.assertEqual(rt.substate, "reveal")
        self.assertEqual(rt.result, "correct")
        self.assertEqual(rt.selected_index, 1)
        self.assertGreater(rt.timer_ratio, 0.0)
        rt.update([hand], 3.7)  # fin du reveal
        self.assertTrue(rt.finished)

    def test_dwell_selects_incorrect(self) -> None:
        question = two_answer_question(correct=1)
        rt = make_runtime(question)
        wrong = point_at(rt.zones[0])
        rt.update([], 0.0)
        rt.update([wrong], 1.1)
        rt.update([wrong], 1.5)
        rt.update([wrong], 2.6)
        self.assertEqual(rt.result, "incorrect")
        self.assertEqual(rt.selected_index, 0)
        self.assertEqual(rt.timer_ratio, 0.0)

    def test_grace_ignores_early_pointing(self) -> None:
        rt = make_runtime(two_answer_question(correct=0))
        hand = point_at(rt.zones[0])
        rt.update([], 0.0)
        rt.update([hand], 1.1)  # active_at = 1.1
        rt.update([hand], 1.25)  # elapsed 0.15 < grace : ignore
        self.assertEqual(rt.dwell[0], 0.0)

    def test_timeout_when_no_answer(self) -> None:
        rt = make_runtime(two_answer_question())
        rt.update([], 0.0)
        rt.update([], 1.1)  # active_at = 1.1
        rt.update([], 11.2)  # timer ecoule (10 s)
        self.assertEqual(rt.result, "timeout")
        self.assertIsNone(rt.selected_index)

    def test_leaving_zone_resets_dwell(self) -> None:
        rt = make_runtime(two_answer_question(correct=0))
        hand = point_at(rt.zones[0])
        rt.update([], 0.0)
        rt.update([hand], 1.1)
        rt.update([hand], 1.5)
        rt.update([hand], 2.0)  # dwell en cours (~0.5)
        self.assertGreater(rt.dwell[0], 0.0)
        rt.update([], 2.2)  # main perdue -> remise a zero
        self.assertEqual(rt.dwell[0], 0.0)


# ---------------------------------------------------------------------------
# Campagne : alternance stricte figure / quiz
# ---------------------------------------------------------------------------


class CampaignTests(unittest.TestCase):
    def _questions(self, n: int) -> list[QuizQuestion]:
        return [
            QuizQuestion(
                id=str(i),
                question=f"Q{i} ?",
                answers=["a", "b"],
                correct_index=0,
                difficulty="easy",
            )
            for i in range(n)
        ]

    def test_no_quiz_keeps_legacy_behaviour(self) -> None:
        campaign = build_breizhcamp_campaign(shuffle=False)
        self.assertTrue(all(c.kind != "quiz" for c in campaign))

    def test_strict_alternation(self) -> None:
        campaign = build_breizhcamp_campaign(
            rng=random.Random(0),
            shuffle=False,
            quiz_questions=self._questions(4),
            max_quiz=4,
        )
        kinds = [c.kind for c in campaign]
        self.assertEqual(len(campaign), 8)
        self.assertTrue(all(kinds[i] != "quiz" for i in range(0, 8, 2)))
        self.assertTrue(all(kinds[i] == "quiz" for i in range(1, 8, 2)))

    def test_max_quiz_caps_count(self) -> None:
        campaign = build_breizhcamp_campaign(
            rng=random.Random(0),
            shuffle=False,
            quiz_questions=self._questions(10),
            max_quiz=2,
        )
        quiz_rounds = [c for c in campaign if c.kind == "quiz"]
        self.assertEqual(len(quiz_rounds), 2)

    def test_select_questions_ramps_difficulty(self) -> None:
        pool = [
            QuizQuestion("h", "?", ["a", "b"], 0, difficulty="hard"),
            QuizQuestion("e", "?", ["a", "b"], 0, difficulty="easy"),
            QuizQuestion("m", "?", ["a", "b"], 0, difficulty="medium"),
        ]
        selected = select_quiz_questions(pool, 3, rng=random.Random(0), shuffle=False)
        self.assertEqual([q.difficulty for q in selected], ["easy", "medium", "hard"])


# ---------------------------------------------------------------------------
# Integration moteur : un round quiz attribue des points
# ---------------------------------------------------------------------------


class QuizEngineTests(unittest.TestCase):
    def _start_quiz_round(self, engine: GameEngine) -> None:
        """Lance la partie + pilule rouge, jusqu'au debut du round quiz."""
        engine.start_game(0.0)
        red_x, red_y = PILL_ZONES["red"]
        engine.update([palm(red_x, red_y)], None, 0.1)
        engine.update([palm(red_x, red_y)], None, 1.05)  # COUNTDOWN
        engine.update([], None, 4.1)  # fin du decompte -> IN_ROUND
        event = engine.update([], None, 4.2)  # creation du quiz (intro)
        self.assertEqual(event.phase, IN_ROUND)
        self.assertEqual(event.challenge_kind, "quiz")
        self.assertEqual(event.quiz_substate, "intro")

    def test_correct_answer_scores(self) -> None:
        question = two_answer_question(correct=1)
        engine = GameEngine(campaign=[make_quiz_challenge(question)])
        self._start_quiz_round(engine)
        zone = engine._quiz.zones[1]
        hand = point_at(zone)
        engine.update([], None, 5.7)  # -> active (intro 1.5 s)
        engine.update([hand], None, 6.1)  # dwell start (apres grace)
        engine.update([hand], None, 7.4)  # verrou correct
        event = engine.update([hand], None, 7.5)
        self.assertEqual(event.quiz_substate, "reveal")
        self.assertEqual(event.quiz_result, "correct")
        # Fin du reveal (2.5 s) -> SCORE avec points > 0.
        event = engine.update([], None, 10.1)
        self.assertEqual(event.phase, SCORE)
        self.assertGreater(event.score, 0)

    def test_wrong_answer_no_points(self) -> None:
        question = two_answer_question(correct=1)
        engine = GameEngine(campaign=[make_quiz_challenge(question)])
        self._start_quiz_round(engine)
        wrong = point_at(engine._quiz.zones[0])
        engine.update([], None, 5.7)
        engine.update([wrong], None, 6.1)
        engine.update([wrong], None, 7.4)
        event = engine.update([], None, 10.2)
        self.assertEqual(event.phase, SCORE)
        self.assertEqual(event.score, 0)
        self.assertEqual(engine.state.combo, 0)

    def test_timeout_no_points(self) -> None:
        question = two_answer_question(correct=1)
        engine = GameEngine(campaign=[make_quiz_challenge(question)])
        self._start_quiz_round(engine)
        engine.update([], None, 5.7)  # active_at = 5.7
        engine.update([], None, 16.0)  # timer 10 s ecoule -> timeout
        event = engine.update([], None, 19.0)  # fin reveal -> SCORE
        self.assertEqual(event.phase, SCORE)
        self.assertEqual(event.score, 0)


if __name__ == "__main__":
    unittest.main()
