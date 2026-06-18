#!/usr/bin/env python3
"""Headless cross-platform smoke test for Matrix Motion.

Loads every MediaPipe model the game uses and runs ONE inference on a synthetic
frame -- no camera, no display. Its only job is to prove the inference pipeline
SURVIVES on the current machine (Linux / Windows / macOS).

Why this matters: a bad GPU delegate (e.g. Metal on macOS for the segmenter)
crashes via a C++ ``abort()`` that Python cannot catch -- it kills the whole
process. So the reliable signal is the EXIT CODE of this script:

    exit 0  -> all models load and infer on this OS (you are sure)
    exit !=0 (incl. crash) -> this machine cannot run the pipeline as configured

Run it on each target OS:

    uv run python scripts/smoke_test.py          # portable CPU path (what --mp-cpu uses)
    uv run python scripts/smoke_test.py --gpu     # probe this machine's GPU delegate

The landmarkers default to the CPU delegate here (the cross-platform safe path);
pass --gpu to check whether the GPU delegate survives on this machine. The
segmenter is always CPU (its GPU path is broken on Metal).
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

# Allow running as `python scripts/smoke_test.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.tracking import (  # noqa: E402
    create_face_detector,
    create_hand_landmarker,
    create_image_segmenter,
    create_pose_landmarker,
    mp,
)


def _synthetic_image() -> "mp.Image":
    """A 720p SRGB frame, same format the game feeds the models."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="probe the GPU delegate for the landmarkers (default: CPU)",
    )
    args = parser.parse_args()
    prefer_gpu = args.gpu

    backend = "GPU-preferred" if prefer_gpu else "CPU"
    print(
        f"[SMOKE] platform={platform.system()} python={platform.python_version()} "
        f"landmarkers={backend} (segmenter always CPU)"
    )

    try:
        image = _synthetic_image()
    except Exception as exc:  # pragma: no cover - environment issue
        print(f"[SMOKE] FAIL building synthetic image: {exc}")
        return 1

    # (label, factory, inference callable taking the created model)
    tasks = [
        ("hand", lambda: create_hand_landmarker(prefer_gpu=prefer_gpu),
         lambda m, ts: m.detect_for_video(image, ts)),
        ("face", lambda: create_face_detector(prefer_gpu=prefer_gpu),
         lambda m, ts: m.detect_for_video(image, ts)),
        ("pose", lambda: create_pose_landmarker(prefer_gpu=prefer_gpu),
         lambda m, ts: m.detect_for_video(image, ts)),
        ("segmenter", create_image_segmenter,
         lambda m, ts: m.segment_for_video(image, ts)),
    ]

    ts = 0
    for label, factory, infer in tasks:
        try:
            model = factory()
        except Exception as exc:
            # Likely a model download failure (offline) -- not a pipeline bug.
            print(f"[SMOKE] SKIP {label}: could not create model ({exc})")
            print("[SMOKE] (offline? models download on first run) -- inconclusive")
            return 2
        try:
            ts += 33
            result = infer(model, ts)
        except Exception as exc:
            print(f"[SMOKE] FAIL {label} inference: {exc}")
            return 1
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()

        # Sanity check the segmenter actually produced a mask.
        if label == "segmenter":
            masks = getattr(result, "confidence_masks", None)
            if not masks:
                print("[SMOKE] FAIL segmenter returned no confidence mask")
                return 1
            shape = np.asarray(masks[0].numpy_view()).shape
            print(f"[SMOKE] segmenter OK (mask {shape[1]}x{shape[0]})")
        else:
            print(f"[SMOKE] {label} OK")

    print(f"[SMOKE] all models OK on {platform.system()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
