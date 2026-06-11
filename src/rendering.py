from __future__ import annotations

import random
import string
from pathlib import Path

import cv2
import numpy as np

from src.challenges import HAND_CONNECTIONS
from src.game_engine import GameEvent

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
HUD_BG = (10, 25, 10)

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


def draw_yolo_detections(frame: np.ndarray, cached_boxes: list, model_names) -> int:
    persons = 0

    for box in cached_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_names.get(cls_id, str(cls_id)) if hasattr(model_names, "get") else str(cls_id)

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        if label == "person":
            persons += 1

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


def render_challenge_frame(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]
    background = load_challenge_background(event.challenge_background_asset, frame.shape)
    panel_top = 92
    panel_bottom = height - 90
    panel_left = 40
    panel_right = width - 40

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (8, 18, 8), -1)
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (10, 28, 10), 2)

    if background is not None:
        card_width = min(520, max(320, int(width * 0.36)))
        card_height = min(320, max(220, int(height * 0.34)))
        card_x = panel_left + 18
        card_y = panel_top + 18
        bg_card = cv2.resize(background, (card_width, card_height), interpolation=cv2.INTER_AREA)
        overlay[card_y : card_y + card_height, card_x : card_x + card_width] = bg_card

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    prompt = event.challenge_prompt or ""
    title = event.challenge_title or ""
    timer = max(0.0, event.timer_left)
    round_label = f"ROUND {event.round_index}/{event.round_total}" if event.round_total else "ROUND"
    cv2.putText(frame, round_label, (70, 132), cv2.FONT_HERSHEY_DUPLEX, 1.0, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, title.upper(), (70, 180), cv2.FONT_HERSHEY_DUPLEX, 0.95, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, prompt, (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (190, 255, 190), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIMER {timer:0.1f}s", (70, 280), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)


def draw_hand_detections(frame: np.ndarray, hand_results, frame_width: int, frame_height: int) -> list[str]:
    gestures: list[str] = []

    if hand_results.hand_landmarks and hand_results.handedness:
        for lm_list, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
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

            hand_label = handedness[0].category_name
            thumb_up, index_up, middle_up, ring_up, pinky_up = _finger_states(lm_list, hand_label)
            local_gesture = _detect_gesture_from_states(lm_list, thumb_up, index_up, middle_up, ring_up, pinky_up)
            gestures.append(local_gesture)

    return gestures


def _finger_states(landmarks: list, hand_label: str) -> tuple[bool, bool, bool, bool, bool]:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    if hand_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def _detect_gesture_from_states(landmarks: list, thumb_up: bool, index_up: bool, middle_up: bool, ring_up: bool, pinky_up: bool) -> str:
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist_thumb_index = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    if dist_thumb_index < 0.05 and middle_up and ring_up and pinky_up:
        return "ok_sign"

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"

    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open_palm"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point"

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"


def draw_hud(frame: np.ndarray, fps: float, persons: int, faces: int, poses: int, event: GameEvent) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 74), HUD_BG, -1)
    cv2.rectangle(frame, (0, 74), (w, 76), MATRIX_DARK_GREEN, -1)

    cv2.putText(frame, "MATRIX VISION", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERSONS: {persons}", (170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FACES: {faces}", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"POSES: {poses}", (500, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{event.phase} | ROUND {event.round_index}/{event.round_total}", (690, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)

    if event.phase == "IN_ROUND":
        cv2.putText(frame, f"TIMER: {event.timer_left:0.1f}s", (20, h - 24), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
    elif event.phase in ("VICTORY", "GAME_OVER"):
        cv2.putText(frame, "DEUX MAINS OUVERTES 1s POUR RESTART", (20, h - 24), cv2.FONT_HERSHEY_DUPLEX, 0.8, MATRIX_GREEN, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "DEUX MAINS OUVERTES 1s POUR DEMARRER", (20, h - 24), cv2.FONT_HERSHEY_DUPLEX, 0.8, MATRIX_GREEN, 2, cv2.LINE_AA)

    if event.status:
        cv2.putText(frame, event.status, (20, h - 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
