from __future__ import annotations

import queue
import ssl
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import mediapipe.tasks as mpt
import numpy as np
from ultralytics import YOLO

# Bypass SSL (meme mecanisme que les POC d'origine) : les Python installes via
# Homebrew n'ont souvent pas de bundle de certificats, ce qui fait echouer le
# telechargement auto des modeles MediaPipe au premier lancement.
ssl._create_default_https_context = ssl._create_unverified_context


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
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
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


class MediaPipeInferenceCache:
    """Keeps last inference result and schedules runs every N frames."""

    def __init__(self, stride: int) -> None:
        self.stride = max(1, stride)
        self._frame_idx = 0
        self._result = None

    def should_run(self) -> bool:
        run = (self._frame_idx % self.stride) == 0
        self._frame_idx += 1
        return run

    def set(self, result) -> None:
        self._result = result

    def get(self):
        return self._result


class PersonMaskTracker:
    """Produit un masque de presence personne lisse pour le fond code rain.

    Enchaine : inference segmentation -> seuil -> flou des bords -> lissage
    temporel (blend avec la frame precedente) pour eviter le scintillement.
    Renvoie un masque float HxW dans [0, 1] (1 = personne).
    """

    def __init__(
        self,
        segmenter: mpt.vision.ImageSegmenter,
        smoothing: float,
        threshold: float,
        blur_ksize: int,
        invert: bool,
    ) -> None:
        self.segmenter = segmenter
        self.smoothing = smoothing
        self.threshold = threshold
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.invert = invert
        self._prev: np.ndarray | None = None
        self._blur_scale = 0.5

    def update(self, mp_image: "mp.Image", timestamp_ms: int, out_hw: tuple[int, int]) -> np.ndarray | None:
        try:
            result = self.segmenter.segment_for_video(mp_image, timestamp_ms)
        except Exception:
            return self._prev

        masks = result.confidence_masks
        if not masks:
            return self._prev

        # confidence_masks[0] = probabilite "premier plan" (personne).
        mask = np.asarray(masks[0].numpy_view(), dtype=np.float32)
        if self.invert:
            mask = 1.0 - mask

        h, w = out_hw
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Seuil souple : binarise puis adoucit les bords au flou gaussien.
        mask = (mask >= self.threshold).astype(np.float32)
        if self._blur_scale < 1.0:
            hs = max(1, int(h * self._blur_scale))
            ws = max(1, int(w * self._blur_scale))
            small = cv2.resize(mask, (ws, hs), interpolation=cv2.INTER_LINEAR)
            small_k = max(3, self.blur_ksize // 2)
            if small_k % 2 == 0:
                small_k += 1
            small = cv2.GaussianBlur(small, (small_k, small_k), 0)
            mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)

        # Lissage temporel anti-scintillement.
        if self._prev is not None:
            mask = self.smoothing * mask + (1.0 - self.smoothing) * self._prev
        self._prev = mask
        return mask


def ensure_model(name: str) -> str:
    path = _PROJECT_ROOT / name
    if not path.exists():
        url = _MEDIAPIPE_MODELS[name]
        print(f"[MODEL] Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[MODEL] Saved to {path}")
    return str(path)


def _make_base_options(model_name: str, prefer_gpu: bool = True) -> mpt.BaseOptions:
    """Create BaseOptions with GPU delegate when supported by the local API.

    MediaPipe Tasks Python has slightly different delegate APIs across versions.
    This helper keeps startup robust by trying GPU first, then falling back.
    """
    model_path = ensure_model(model_name)
    if prefer_gpu:
        try:
            delegate_gpu = mpt.BaseOptions.Delegate.GPU
            return mpt.BaseOptions(model_asset_path=model_path, delegate=delegate_gpu)
        except Exception:
            pass
    return mpt.BaseOptions(model_asset_path=model_path)


def create_hand_landmarker(prefer_gpu: bool = True) -> mpt.vision.HandLandmarker:
    try:
        hand_options = mpt.vision.HandLandmarkerOptions(
            base_options=_make_base_options("hand_landmarker.task", prefer_gpu=prefer_gpu),
            running_mode=mpt.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mpt.vision.HandLandmarker.create_from_options(hand_options)
    except Exception:
        hand_options = mpt.vision.HandLandmarkerOptions(
            base_options=_make_base_options("hand_landmarker.task", prefer_gpu=False),
            running_mode=mpt.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mpt.vision.HandLandmarker.create_from_options(hand_options)


def create_face_detector(prefer_gpu: bool = True) -> mpt.vision.FaceDetector:
    try:
        face_options = mpt.vision.FaceDetectorOptions(
            base_options=_make_base_options("face_detector.tflite", prefer_gpu=prefer_gpu),
            running_mode=mpt.vision.RunningMode.VIDEO,
            min_detection_confidence=0.55,
        )
        return mpt.vision.FaceDetector.create_from_options(face_options)
    except Exception:
        face_options = mpt.vision.FaceDetectorOptions(
            base_options=_make_base_options("face_detector.tflite", prefer_gpu=False),
            running_mode=mpt.vision.RunningMode.VIDEO,
            min_detection_confidence=0.55,
        )
        return mpt.vision.FaceDetector.create_from_options(face_options)


def create_image_segmenter() -> mpt.vision.ImageSegmenter:
    """Selfie Segmentation : masque de confiance personne/fond en temps reel.

    Utilise pour incruster la pluie de code DERRIERE la personne. Renvoie des
    masques de confiance (float 0..1) qui donnent des bords plus doux qu'un
    masque de categorie binaire.

    Force le delegate CPU : le delegate GPU de MediaPipe (Metal sur macOS) fait
    planter la conversion vers texture GPU par un CHECK C++ qui abort le process
    (non rattrapable en Python, donc le fallback try/except ne sert a rien ici).
    Le CPU est le seul chemin portable Linux / Windows / macOS.
    """
    segmenter_options = mpt.vision.ImageSegmenterOptions(
        base_options=_make_base_options("selfie_segmenter.tflite", prefer_gpu=False),
        running_mode=mpt.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    return mpt.vision.ImageSegmenter.create_from_options(segmenter_options)


def create_pose_landmarker(prefer_gpu: bool = True) -> mpt.vision.PoseLandmarker:
    try:
        pose_options = mpt.vision.PoseLandmarkerOptions(
            base_options=_make_base_options("pose_landmarker_lite.task", prefer_gpu=prefer_gpu),
            running_mode=mpt.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mpt.vision.PoseLandmarker.create_from_options(pose_options)
    except Exception:
        pose_options = mpt.vision.PoseLandmarkerOptions(
            base_options=_make_base_options("pose_landmarker_lite.task", prefer_gpu=False),
            running_mode=mpt.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mpt.vision.PoseLandmarker.create_from_options(pose_options)
