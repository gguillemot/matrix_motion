from __future__ import annotations

import math
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np

from src.challenges import HAND_CONNECTIONS, PILL_ZONES
from src.game_engine import GameEvent

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
MATRIX_PALE_GREEN = (190, 255, 190)
HUD_BG = (10, 25, 10)
PILL_COLORS = {"red": (60, 60, 255), "blue": (255, 140, 40)}  # BGR

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (17, 19), (19, 21),
    (16, 18), (18, 20), (20, 22),
]

POSE_BODY_LANDMARKS = range(11, 33)


class MatrixRain:
    def __init__(self, width: int, height: int, spacing: int = 12, trail_length: int = 4) -> None:
        self.width = width
        self.height = height
        self.spacing = spacing
        self.trail_length = trail_length
        self.columns = max(1, width // spacing)
        self.drops = [random.randint(-height, 0) for _ in range(self.columns)]
        self.charset = string.digits

    def draw(self, frame: np.ndarray, boost: bool = False) -> None:
        overlay = np.zeros_like(frame)
        speed = 25 if boost else 14

        for col in range(self.columns):
            x = col * self.spacing
            y = self.drops[col]
            for trail_idx in range(self.trail_length):
                trail_y = y - trail_idx * self.spacing
                if trail_y < 0:
                    continue

                char = random.choice(self.charset)
                brightness = max(95, 255 - trail_idx * 35)
                color = (35, brightness, 70)

                cv2.putText(overlay, char, (x, trail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

            self.drops[col] += speed
            if self.drops[col] > self.height + random.randint(0, 150):
                self.drops[col] = random.randint(-140, 0)

        cv2.addWeighted(overlay, 0.35, frame, 1.0, 0, frame)


def add_scanlines(frame: np.ndarray, step: int = 4) -> None:
    frame[::step, :] = (frame[::step, :] * 0.55).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers texte / overlay
# ---------------------------------------------------------------------------


def _put_centered(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    glow: bool = False,
) -> None:
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (frame.shape[1] - text_w) // 2)
    if glow:
        cv2.putText(frame, text, (x, y), font, scale, MATRIX_DARK_GREEN, thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _darken(frame: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(np.zeros_like(frame), alpha, frame, 1.0 - alpha, 0, frame)


def _blink(period: float = 0.8) -> bool:
    return int(time.monotonic() / period) % 2 == 0


def _draw_progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: tuple[int, int, int] = MATRIX_GREEN,
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), MATRIX_DARK_GREEN, 1)
    fill = int(width * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)


# ---------------------------------------------------------------------------
# Detections (debug / feedback visuel)
# ---------------------------------------------------------------------------


def draw_yolo_detections(frame: np.ndarray, cached_boxes: list, model_names, draw: bool = True) -> int:
    persons = 0

    for box in cached_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_names.get(cls_id, str(cls_id)) if hasattr(model_names, "get") else str(cls_id)

        if label == "person":
            persons += 1

        if not draw:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), MATRIX_DARK_GREEN, 1)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MATRIX_GREEN, 2, cv2.LINE_AA)

    return persons


def draw_face_detections(frame: np.ndarray, detections) -> int:
    faces = 0

    if detections:
        for det in detections:
            faces += 1
            bb = det.bounding_box
            x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (80, 255, 140), 2)

    return faces


def draw_pose_skeleton(frame: np.ndarray, pose_results, frame_width: int, frame_height: int) -> int:
    poses = 0

    if pose_results.pose_landmarks:
        for pose_landmarks in pose_results.pose_landmarks:
            poses += 1

            for start, end in POSE_CONNECTIONS:
                x1 = int(pose_landmarks[start].x * frame_width)
                y1 = int(pose_landmarks[start].y * frame_height)
                x2 = int(pose_landmarks[end].x * frame_width)
                y2 = int(pose_landmarks[end].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)

            for index in POSE_BODY_LANDMARKS:
                landmark = pose_landmarks[index]
                cx = int(landmark.x * frame_width)
                cy = int(landmark.y * frame_height)
                cv2.circle(frame, (cx, cy), 3, (120, 255, 150), -1)

    return poses


def draw_hand_detections(frame: np.ndarray, hand_results, frame_width: int, frame_height: int) -> int:
    hands = 0

    if hand_results.hand_landmarks:
        for lm_list in hand_results.hand_landmarks:
            hands += 1
            for start, end in HAND_CONNECTIONS:
                x1 = int(lm_list[start].x * frame_width)
                y1 = int(lm_list[start].y * frame_height)
                x2 = int(lm_list[end].x * frame_width)
                y2 = int(lm_list[end].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), MATRIX_DARK_GREEN, 2)

            for lm in lm_list:
                cx = int(lm.x * frame_width)
                cy = int(lm.y * frame_height)
                cv2.circle(frame, (cx, cy), 4, MATRIX_GREEN, -1)

    return hands


# ---------------------------------------------------------------------------
# Ecrans du jeu
# ---------------------------------------------------------------------------


def draw_attract_screen(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.45)

    _put_centered(frame, "ENTER THE MATRIX", height // 2 - 70, 2.2, MATRIX_GREEN, 4, glow=True)
    if _blink(1.4):
        _put_centered(frame, "WAKE UP, NEO...", height // 2, 1.0, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)
    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(frame, "MONTRE TES 2 PAUMES POUR COMMENCER", height // 2 + 80, 0.95, MATRIX_GREEN, 2)

    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (frame.shape[1] - bar_width) // 2, height // 2 + 110, bar_width, 14, event.start_hold_progress)


def draw_countdown(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.35)
    value = max(1, event.countdown_value)
    # Pulsation : le chiffre grossit a mesure que la seconde s'ecoule.
    fraction = time.monotonic() % 1.0
    scale = 5.0 + 1.5 * fraction
    _put_centered(frame, "GET READY", height // 2 - 130, 1.1, MATRIX_PALE_GREEN, 2)
    _put_centered(frame, str(value), height // 2 + 70, scale, MATRIX_GREEN, 10, glow=True)


def draw_round_overlay(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    # Bandeau figure (sous le HUD)
    round_label = f"ROUND {event.round_index}/{event.round_total}"
    cv2.putText(frame, round_label, (24, 116), cv2.FONT_HERSHEY_DUPLEX, 0.85, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, event.challenge_title.upper(), (24, 158), cv2.FONT_HERSHEY_DUPLEX, 1.15, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, event.challenge_prompt, (24, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.78, MATRIX_PALE_GREEN, 2, cv2.LINE_AA)

    # Progression de la figure (maintien pilule/bunny, cuillere, scan)
    if event.figure_progress > 0:
        _draw_progress_bar(frame, 24, 212, 280, 12, event.figure_progress)

    # Timer : barre + texte. Verte, puis rouge sous 2 s.
    ratio = event.timer_left / event.round_duration if event.round_duration > 0 else 0.0
    timer_color = MATRIX_GREEN if event.timer_left > 2.0 else (60, 60, 255)
    bar_margin = 24
    bar_y = height - 46
    _draw_progress_bar(frame, bar_margin, bar_y, width - 2 * bar_margin, 16, ratio, timer_color)
    cv2.putText(
        frame,
        f"{event.timer_left:0.1f}s",
        (bar_margin, bar_y - 12),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        timer_color,
        2,
        cv2.LINE_AA,
    )


def draw_pills(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    for name, (zone_x, zone_y) in PILL_ZONES.items():
        cx, cy = int(zone_x * width), int(zone_y * height)
        color = PILL_COLORS[name]
        hovered = event.pill_hover == name

        # Halo
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 70 if hovered else 58, color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Capsule + reflet
        axes = (52, 26)
        cv2.ellipse(frame, (cx, cy), axes, 25, 0, 360, color, -1)
        cv2.ellipse(frame, (cx, cy), axes, 25, 0, 360, (255, 255, 255), 2)
        cv2.ellipse(frame, (cx - 12, cy - 10), (16, 6), 25, 0, 360, (255, 255, 255), -1)

        if hovered:
            cv2.circle(frame, (cx, cy), 80, (255, 255, 255), 2)

        label = name.upper()
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(frame, label, (cx - text_w // 2, cy + 78), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_spoon(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    # Cuillere geante fixe au centre : le joueur la tord "par telekinesie" en
    # pincant et tournant sa main, comme l'enfant de l'Oracle. Toujours
    # visible pendant la figure, lisible de loin sur le stand.
    length = int(height * 0.42)
    base_x = width // 2
    base_y = int(height * 0.80)  # bas du manche, fixe
    bend = event.figure_progress
    steps = 16

    points: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        # Le bas du manche reste droit, la moitie haute se courbe de plus en
        # plus (offset quadratique, max ~35 % de la longueur).
        curve = max(0.0, t - 0.45) / 0.55
        offset = bend * 0.35 * length * curve * curve
        points.append((int(base_x + offset), int(base_y - length * t)))

    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], False, (60, 60, 70), 22, cv2.LINE_AA)  # contour sombre
    cv2.polylines(frame, [pts], False, (210, 210, 220), 14, cv2.LINE_AA)  # corps metal
    cv2.polylines(frame, [pts], False, (255, 255, 255), 4, cv2.LINE_AA)  # reflet

    # Cuilleron au sommet, oriente selon la torsion du bout du manche
    tip_x, tip_y = points[-1]
    prev_x, prev_y = points[-2]
    tip_angle = math.degrees(math.atan2(tip_y - prev_y, tip_x - prev_x)) + 90
    bowl_axes = (int(height * 0.06), int(height * 0.085))
    cv2.ellipse(frame, (tip_x, tip_y), bowl_axes, tip_angle, 0, 360, (60, 60, 70), -1)
    cv2.ellipse(frame, (tip_x, tip_y), (bowl_axes[0] - 5, bowl_axes[1] - 5), tip_angle, 0, 360, (225, 225, 235), -1)
    cv2.ellipse(
        frame,
        (tip_x - bowl_axes[0] // 3, tip_y - bowl_axes[1] // 3),
        (bowl_axes[0] // 3, bowl_axes[1] // 4),
        tip_angle,
        0,
        360,
        (255, 255, 255),
        -1,
    )

    if event.spoon_anchor is None:
        if _blink(0.8):
            _put_centered(frame, "PINCE POUCE + INDEX ET TOURNE LA MAIN", height - 90, 0.85, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)
        return

    # Rayon de telekinesie : main pincee -> cuillere
    hand_x = int(event.spoon_anchor[0] * width)
    hand_y = int(event.spoon_anchor[1] * height)
    mid_y = base_y - length // 2
    cv2.line(frame, (hand_x, hand_y), (base_x, mid_y), MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.circle(frame, (hand_x, hand_y), 12, MATRIX_GREEN, 2)
    cv2.putText(
        frame,
        f"{int(bend * 100)}%",
        (base_x + int(height * 0.09), mid_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )


def draw_agent_glasses(frame: np.ndarray, detections, progress: float) -> int:
    faces = 0
    if not detections:
        return faces

    for det in detections:
        faces += 1
        bb = det.bounding_box
        x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height

        # Cadre + consigne toujours visibles pour signaler la figure active
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), MATRIX_DARK_GREEN, 1)
        if progress <= 0.0:
            if _blink(0.5):
                cv2.putText(frame, "DON'T MOVE", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
            continue

        cv2.putText(
            frame,
            f"SCANNING {int(progress * 100)}%",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            MATRIX_GREEN,
            2,
            cv2.LINE_AA,
        )

        # Les lunettes sont la recompense de l'immobilite : elles apparaissent
        # en fondu (opacite = progression du scan) et disparaissent des que le
        # joueur bouge. Opacite totale = victoire.
        overlay = frame.copy()
        eye_y = y + int(0.38 * bh)
        eye_dx = int(0.22 * bw)
        eye_axes = (max(6, int(0.15 * bw)), max(4, int(0.10 * bh)))
        left_eye = (x + bw // 2 - eye_dx, eye_y)
        right_eye = (x + bw // 2 + eye_dx, eye_y)
        for center in (left_eye, right_eye):
            cv2.ellipse(overlay, center, eye_axes, 0, 0, 360, (20, 20, 20), -1)
            cv2.ellipse(overlay, center, eye_axes, 0, 0, 360, (90, 90, 90), 2)
        cv2.line(overlay, (left_eye[0] + eye_axes[0], eye_y), (right_eye[0] - eye_axes[0], eye_y), (20, 20, 20), 3)
        cv2.line(overlay, (x, eye_y), (left_eye[0] - eye_axes[0], eye_y), (20, 20, 20), 3)
        cv2.line(overlay, (right_eye[0] + eye_axes[0], eye_y), (x + bw, eye_y), (20, 20, 20), 3)

        # Ligne de scan animee par la progression d'immobilite
        scan_y = y + int(progress * bh)
        cv2.line(overlay, (x, scan_y), (x + bw, scan_y), MATRIX_GREEN, 2)

        alpha = min(1.0, progress)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    return faces


def draw_flash(frame: np.ndarray, event: GameEvent) -> None:
    if not event.flash_message:
        return

    height, width = frame.shape[:2]
    band_top, band_bottom = height // 2 - 60, height // 2 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, band_top), (width, band_bottom), (5, 15, 5), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    _put_centered(frame, event.flash_message, height // 2, 1.6, MATRIX_GREEN, 3, glow=True)


def draw_score_screen(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.55)

    successes = sum(result.success for result in event.round_results)
    headline = "ACCESS GRANTED" if successes else "SYSTEM FAILURE"
    _put_centered(frame, headline, 150, 1.8, MATRIX_GREEN, 3, glow=True)
    _put_centered(frame, f"SCORE  {event.score}", 215, 1.3, MATRIX_PALE_GREEN, 2)

    y = 280
    for result in event.round_results:
        mark = "[OK]" if result.success else "[--]"
        points = f"+{result.points}" if result.success else "  0"
        color = MATRIX_GREEN if result.success else (90, 130, 95)
        _put_centered(frame, f"{mark} {result.title.upper()}  {points}", y, 0.75, color, 2, cv2.FONT_HERSHEY_SIMPLEX)
        y += 38

    if event.chosen_pill:
        _put_centered(frame, f"PILL OF CHOICE : {event.chosen_pill.upper()}", y + 10, 0.85, PILL_COLORS[event.chosen_pill], 2)
        y += 48

    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(frame, "2 PAUMES OUVERTES POUR REJOUER", height - 70, 0.9, MATRIX_GREEN, 2)
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (frame.shape[1] - bar_width) // 2, height - 50, bar_width, 14, event.start_hold_progress)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def load_challenge_background(asset_name: str, frame_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not asset_name:
        return None

    asset_path = Path(__file__).resolve().parent.parent / "assets" / asset_name
    if not asset_path.exists():
        return None

    background = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
    if background is None:
        return None

    height, width = frame_shape[:2]
    return cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)


def draw_hud(frame: np.ndarray, fps: float, persons: int, faces: int, poses: int, event: GameEvent) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 74), HUD_BG, -1)
    cv2.rectangle(frame, (0, 74), (w, 76), MATRIX_DARK_GREEN, -1)

    cv2.putText(frame, "MATRIX VISION", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERSONS: {persons}", (170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FACES: {faces}", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"POSES: {poses}", (500, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)

    status = f"{event.phase}"
    if event.round_total:
        status += f" | ROUND {event.round_index}/{event.round_total}"
    cv2.putText(frame, status, (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SCORE: {event.score}", (w - 420, 58), cv2.FONT_HERSHEY_DUPLEX, 0.7, MATRIX_GREEN, 2, cv2.LINE_AA)
