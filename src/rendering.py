from __future__ import annotations

import random
import string

import cv2
import numpy as np

from src.challenges import HAND_CONNECTIONS, detect_gesture

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
HUD_BG = (10, 25, 10)


class MatrixRain:
    def __init__(self, width: int, height: int, spacing: int = 12, trail_length: int = 4) -> None:
        self.width = width
        self.height = height
        self.spacing = spacing
        self.trail_length = trail_length
        self.columns = max(1, width // spacing)
        self.drops = [random.randint(-height, 0) for _ in range(self.columns)]
        self.charset = string.ascii_letters + string.digits + "#$%&@!?+-*/"

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
            cv2.putText(frame, "face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 140), 2, cv2.LINE_AA)

    return faces


def draw_hand_detections(frame: np.ndarray, hand_results, frame_width: int, frame_height: int) -> str:
    gesture = "none"

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
            local_gesture = detect_gesture(lm_list, hand_label)
            if local_gesture != "none":
                gesture = local_gesture

    return gesture


def draw_hud(frame: np.ndarray, fps: float, persons: int, faces: int, gesture: str, status: str) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 74), HUD_BG, -1)
    cv2.rectangle(frame, (0, 74), (w, 76), MATRIX_DARK_GREEN, -1)

    cv2.putText(frame, "MATRIX VISION", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERSONS: {persons}", (170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FACES: {faces}", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"GESTURE: {gesture}", (500, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)

    if status:
        cv2.putText(frame, status, (20, h - 24), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
