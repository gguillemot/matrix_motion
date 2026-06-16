from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import mediapipe.tasks as mpt
import numpy as np

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


def create_image_segmenter() -> mpt.vision.ImageSegmenter:
    """Selfie Segmentation : masque de confiance personne/fond en temps reel.

    Utilise pour incruster la pluie de code DERRIERE la personne. Renvoie des
    masques de confiance (float 0..1) qui donnent des bords plus doux qu'un
    masque de categorie binaire.
    """
    segmenter_options = mpt.vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_model("selfie_segmenter.tflite")),
        running_mode=mpt.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    return mpt.vision.ImageSegmenter.create_from_options(segmenter_options)


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
