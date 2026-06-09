#!/usr/bin/env python3
"""Matrix-style real-time vision demo.

Features:
- Object and person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand landmark tracking + simple gesture recognition
- Optional MQTT publish to your NodeMCU on hand gestures
- Matrix-inspired HUD, rain effect, scanlines, and status overlays
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import string
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import mediapipe.tasks as mpt
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
HUD_BG = (10, 25, 10)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

_MEDIAPIPE_MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_detector.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
}


class YoloWorker:
    """Runs YOLO inference in a background thread to avoid blocking the main loop."""

    def __init__(self, model: YOLO, imgsz: int, conf: float) -> None:
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self._in: queue.Queue = queue.Queue(maxsize=1)
        self._boxes: list = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._in.put_nowait(frame)
        except queue.Full:
            pass  # skip frame — worker is still busy

    def get_boxes(self) -> list:
        with self._lock:
            return list(self._boxes)

    def _run(self) -> None:
        while True:
            frame = self._in.get()
            try:
                results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
                boxes = results[0].boxes
                with self._lock:
                    self._boxes = boxes if boxes is not None else []
            except Exception:
                pass


def ensure_model(name: str) -> str:
    path = Path(__file__).parent / name
    if not path.exists():
        url = _MEDIAPIPE_MODELS[name]
        print(f"[MODEL] Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[MODEL] Saved to {path}")
    return str(path)


@dataclass
class AppConfig:
    camera_index: int
    width: int
    height: int
    model: str
    conf: float
    imgsz: int
    yolo_stride: int
    disable_yolo: bool
    window_name: str
    windowed: bool
    mqtt_disable: bool
    mqtt_host: str
    mqtt_port: int
    mqtt_topic: str
    mqtt_token: str
    mqtt_client_id: str


class MQTTPublisher:
    def __init__(self, cfg: AppConfig) -> None:
        self.enabled = not cfg.mqtt_disable
        self.topic = cfg.mqtt_topic
        self.token = cfg.mqtt_token
        self.client: Optional[mqtt.Client] = None
        self.connected = False

        if not self.enabled:
            return

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cfg.mqtt_client_id)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        try:
            client.connect(cfg.mqtt_host, cfg.mqtt_port, keepalive=60)
            client.loop_start()
            self.client = client
        except Exception as exc:
            print(f"[MQTT] disabled (connect failed): {exc}")
            self.enabled = False

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties):
        self.connected = not reason_code.is_failure
        if self.connected:
            print("[MQTT] connected")
        else:
            print(f"[MQTT] connect failed rc={reason_code}")

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _properties):
        self.connected = False

    def publish_pill(self, pill: str) -> bool:
        if not self.enabled or self.client is None:
            return False

        payload = json.dumps({"token": self.token, "pill": pill})
        info = self.client.publish(self.topic, payload, qos=0, retain=False)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()


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

                cv2.putText(
                    overlay,
                    char,
                    (x, trail_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            self.drops[col] += speed
            if self.drops[col] > self.height + random.randint(0, 150):
                self.drops[col] = random.randint(-140, 0)

        cv2.addWeighted(overlay, 0.35, frame, 1.0, 0, frame)


def add_scanlines(frame: np.ndarray, step: int = 4) -> None:
    frame[::step, :] = (frame[::step, :] * 0.55).astype(np.uint8)


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


def finger_states(landmarks: list, hand_label: str) -> tuple[bool, bool, bool, bool, bool]:
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    # Handedness from MediaPipe may vary with mirrored previews.
    if hand_label == "Right":
        thumb_up = thumb_tip.x < thumb_ip.x
    else:
        thumb_up = thumb_tip.x > thumb_ip.x

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def detect_gesture(landmarks: list, hand_label: str) -> str:
    thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states(landmarks, hand_label)

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist_thumb_index = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    if dist_thumb_index < 0.05 and middle_up and ring_up and pinky_up:
        return "ok_sign"

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"

    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open_palm"

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Matrix-style vision demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default=str(Path(__file__).parent / "yolov8n.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--yolo-stride", type=int, default=3)
    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument("--window-name", type=str, default="THE MATRIX")
    parser.add_argument("--windowed", action="store_true")

    parser.add_argument("--mqtt-disable", action="store_true")
    parser.add_argument("--mqtt-host", type=str, default="broker.hivemq.com")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", type=str, default="thematrix/pill/CHANGE_TO_UNIQUE_ID")
    parser.add_argument("--mqtt-token", type=str, default="CHANGE_ME_TO_A_LONG_RANDOM_SECRET")
    parser.add_argument("--mqtt-client-id", type=str, default=f"matrix-motion-{random.randint(1000, 9999)}")

    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        yolo_stride=max(1, args.yolo_stride),
        disable_yolo=args.disable_yolo,
        window_name=args.window_name,
        windowed=args.windowed,
        mqtt_disable=args.mqtt_disable,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_token=args.mqtt_token,
        mqtt_client_id=args.mqtt_client_id,
    )


def main() -> None:
    cfg = parse_args()

    print("[INFO] Starting Matrix Motion")
    if not cfg.mqtt_disable and "CHANGE_" in cfg.mqtt_topic:
        print("[INFO] MQTT topic/token still on placeholders. Set --mqtt-topic and --mqtt-token.")

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try --camera-index 1 or verify camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    window_flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(cfg.window_name, window_flags)
    if not cfg.windowed:
        cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    publisher = MQTTPublisher(cfg)

    yolo_worker: Optional[YoloWorker] = None
    model: Optional[YOLO] = None
    if not cfg.disable_yolo:
        try:
            model = YOLO(cfg.model)
            yolo_worker = YoloWorker(model, cfg.imgsz, cfg.conf)
            print(f"[YOLO] loaded {cfg.model} (background thread, imgsz={cfg.imgsz})")
        except Exception as exc:
            print(f"[YOLO] disabled (load failed): {exc}")

    hand_options = mpt.vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mpt.vision.HandLandmarker.create_from_options(hand_options)

    face_options = mpt.vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("face_detector.tflite")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        min_detection_confidence=0.55,
    )
    face_detector = mpt.vision.FaceDetector.create_from_options(face_options)
    video_start_ts = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera opened but no frame received.")

    frame = cv2.flip(frame, 1)
    rain = MatrixRain(frame.shape[1], frame.shape[0])

    frame_idx = 0
    last_tick = time.perf_counter()
    fps = 0.0

    status = "SYSTEM ONLINE"
    status_until = time.monotonic() + 2.0
    last_action_ts = 0.0
    rain_boost = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.perf_counter() - video_start_ts) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_results = face_detector.detect_for_video(mp_image, timestamp_ms)
        h_frame, w_frame = frame.shape[:2]

        if yolo_worker is not None and frame_idx % cfg.yolo_stride == 0:
            yolo_worker.submit(frame.copy())

        cached_boxes = yolo_worker.get_boxes() if yolo_worker is not None else []
        persons = 0
        for box in cached_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id)) if model is not None else str(cls_id)

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if label == "person":
                persons += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)
            cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), MATRIX_DARK_GREEN, 1)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MATRIX_GREEN, 2, cv2.LINE_AA)

        faces = 0
        if face_results.detections:
            for det in face_results.detections:
                faces += 1
                bb = det.bounding_box
                x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (80, 255, 140), 2)
                cv2.putText(frame, "face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 140), 2, cv2.LINE_AA)

        gesture = "none"
        if hand_results.hand_landmarks and hand_results.handedness:
            for lm_list, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
                for start, end in HAND_CONNECTIONS:
                    x1 = int(lm_list[start].x * w_frame)
                    y1 = int(lm_list[start].y * h_frame)
                    x2 = int(lm_list[end].x * w_frame)
                    y2 = int(lm_list[end].y * h_frame)
                    cv2.line(frame, (x1, y1), (x2, y2), MATRIX_DARK_GREEN, 2)
                for lm in lm_list:
                    cx = int(lm.x * w_frame)
                    cy = int(lm.y * h_frame)
                    cv2.circle(frame, (cx, cy), 4, MATRIX_GREEN, -1)

                hand_label = handedness[0].category_name
                local_gesture = detect_gesture(lm_list, hand_label)
                if local_gesture != "none":
                    gesture = local_gesture

        now = time.monotonic()
        if gesture != "none" and now - last_action_ts > 2.5:
            if gesture == "thumbs_up":
                sent = publisher.publish_pill("blue")
                status = "PILULE BLEUE" if sent else "THUMBS UP DETECTED"
                status_until = now + 2.0
                last_action_ts = now
            elif gesture == "ok_sign":
                sent = publisher.publish_pill("red")
                status = "PILULE ROUGE" if sent else "OK SIGN DETECTED"
                status_until = now + 2.0
                last_action_ts = now
            elif gesture == "open_palm":
                rain_boost = not rain_boost
                status = "MATRIX BOOST ON" if rain_boost else "MATRIX BOOST OFF"
                status_until = now + 2.0
                last_action_ts = now

        rain.draw(frame, boost=rain_boost)
        # add_scanlines(frame, step=4)

        if now > status_until:
            status = ""

        tick = time.perf_counter()
        dt = tick - last_tick
        if dt > 0:
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)
        last_tick = tick

        draw_hud(frame, fps=fps, persons=persons, faces=faces, gesture=gesture, status=status)

        cv2.imshow(cfg.window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            current = cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN)
            if current == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    publisher.close()
    hand_landmarker.close()
    face_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
