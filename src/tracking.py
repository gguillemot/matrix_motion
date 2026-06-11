from __future__ import annotations

import queue
import threading
import urllib.request
from pathlib import Path

import mediapipe.tasks as mpt
import numpy as np
from ultralytics import YOLO


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MEDIAPIPE_MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
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
        self._in: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._boxes: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._in.put_nowait(frame)
        except queue.Full:
            pass

    def get_boxes(self) -> list:
        with self._lock:
            return list(self._boxes)

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._in.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._in.get()
            if frame is None:
                break

            try:
                results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
                boxes = results[0].boxes
                with self._lock:
                    self._boxes = boxes if boxes is not None else []
            except Exception:
                pass


def ensure_model(name: str) -> str:
    path = _PROJECT_ROOT / name
    if not path.exists():
        url = _MEDIAPIPE_MODELS[name]
        print(f"[MODEL] Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[MODEL] Saved to {path}")
    return str(path)


def create_hand_landmarker() -> mpt.vision.HandLandmarker:
    hand_options = mpt.vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mpt.vision.HandLandmarker.create_from_options(hand_options)


def create_face_detector() -> mpt.vision.FaceDetector:
    face_options = mpt.vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("face_detector.tflite")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        min_detection_confidence=0.55,
    )
    return mpt.vision.FaceDetector.create_from_options(face_options)


def create_pose_landmarker() -> mpt.vision.PoseLandmarker:
    pose_options = mpt.vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("pose_landmarker_lite.task")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mpt.vision.PoseLandmarker.create_from_options(pose_options)
