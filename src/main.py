from __future__ import annotations

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from src.config import PROJECT_ROOT, AppConfig, parse_args
from src.challenges import observe_hands, observe_pose
from src.game_engine import ATTRACT, COUNTDOWN, IN_ROUND, SCORE, GameEngine
from src.mqtt_client import MQTTConfig, MQTTPublisher
from src.rendering import (
    MatrixRain,
    draw_agent_glasses,
    draw_attract_screen,
    draw_celebration,
    draw_countdown,
    draw_face_detections,
    draw_flash,
    draw_hand_detections,
    draw_hud,
    draw_pills,
    draw_pose_skeleton,
    draw_round_overlay,
    draw_score_screen,
    draw_spoon,
    draw_yolo_detections,
)
from src.tracking import YoloWorker, create_face_detector, create_hand_landmarker, create_pose_landmarker


def run(cfg: AppConfig) -> None:
    print("[INFO] Starting Matrix Motion")
    if not cfg.mqtt_disable and "CHANGE_" in cfg.mqtt_topic:
        print("[INFO] MQTT topic/token still on placeholders. Set --mqtt-topic and --mqtt-token.")

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try --camera-index 1 or verify camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    if not cfg.windowed:
        cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    publisher = MQTTPublisher(
        MQTTConfig(
            enabled=not cfg.mqtt_disable,
            host=cfg.mqtt_host,
            port=cfg.mqtt_port,
            topic=cfg.mqtt_topic,
            token=cfg.mqtt_token,
            client_id=cfg.mqtt_client_id,
        )
    )

    yolo_worker: YoloWorker | None = None
    model: YOLO | None = None
    hand_landmarker = None
    face_detector = None
    pose_landmarker = None
    rain = None
    engine = GameEngine(
        round_duration=cfg.round_duration,
        countdown_duration=cfg.countdown_duration,
        victory_pill=cfg.victory_pill,
        best_score_path=PROJECT_ROOT / "highscore.json",
    )

    try:
        if not cfg.disable_yolo:
            try:
                model = YOLO(cfg.model)
                yolo_worker = YoloWorker(model, cfg.imgsz, cfg.conf)
                print(f"[YOLO] loaded {cfg.model} (background thread, imgsz={cfg.imgsz})")
            except Exception as exc:
                print(f"[YOLO] disabled (load failed): {exc}")

        hand_landmarker = create_hand_landmarker()
        face_detector = create_face_detector()
        pose_landmarker = create_pose_landmarker()
        video_start_ts = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera opened but no frame received.")

        frame = cv2.flip(frame, 1)
        rain = MatrixRain(frame.shape[1], frame.shape[0])

        frame_idx = 0
        last_tick = time.perf_counter()
        fps = 0.0

        # Bullet-time : on garde les ~24 dernieres frames propres pour les
        # rejouer au ralenti quand le Neo Dodge est reussi.
        frame_buffer: deque[np.ndarray] = deque(maxlen=24)
        replay_frames: list[np.ndarray] = []
        replay_pos = 0.0
        bullet_time_active = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            frame_buffer.append(frame.copy())

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.perf_counter() - video_start_ts) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_results = face_detector.detect_for_video(mp_image, timestamp_ms)
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            h_frame, w_frame = frame.shape[:2]

            if yolo_worker is not None and frame_idx % cfg.yolo_stride == 0:
                yolo_worker.submit(frame.copy())

            hands = observe_hands(hand_results)
            pose_observation = observe_pose(pose_results)

            now = time.monotonic()
            event = engine.update(hands, pose_observation, now, publisher.publish_pill)
            if event.published:
                print(f"[MQTT] pill published: {event.chosen_pill or 'default'}")

            # Bullet-time : au debut de la celebration du dodge, on fige le
            # buffer ; tant qu'il reste des frames, on les rejoue ~2.5x plus
            # lentement pendant que camera et moteur continuent de tourner.
            if event.celebration_key == "neo_dodge":
                if not bullet_time_active:
                    bullet_time_active = True
                    replay_frames = list(frame_buffer)
                    replay_pos = 0.0
            else:
                bullet_time_active = False
                replay_frames = []

            replay_frame = None
            if bullet_time_active and int(replay_pos) < len(replay_frames):
                replay_frame = replay_frames[int(replay_pos)].copy()
                replay_pos += 0.4

            cached_boxes = yolo_worker.get_boxes() if yolo_worker is not None else []

            if replay_frame is not None:
                # Pendant le rejeu : pas d'overlays de detection ni d'ecran de
                # round, juste l'effet bullet-time sur les frames du passe.
                display = replay_frame
                persons = draw_yolo_detections(display, cached_boxes, getattr(model, "names", {}), draw=False)
                faces = len(face_results.detections or [])
                poses = len(pose_results.pose_landmarks or [])
            else:
                display = frame
                # Les boites YOLO ne sont dessinees qu'en attract : pendant la
                # partie, elles parasitent la lecture des figures.
                persons = draw_yolo_detections(
                    display, cached_boxes, getattr(model, "names", {}), draw=event.phase == ATTRACT
                )

                scanning = event.phase == IN_ROUND and event.challenge_kind == "scan"
                if scanning:
                    faces = draw_agent_glasses(display, face_results.detections, event.figure_progress)
                else:
                    faces = draw_face_detections(display, face_results.detections)

                draw_hand_detections(display, hand_results, w_frame, h_frame)
                poses = draw_pose_skeleton(display, pose_results, w_frame, h_frame)

                if event.phase == ATTRACT:
                    draw_attract_screen(display, event)
                elif event.phase == COUNTDOWN:
                    draw_countdown(display, event)
                elif event.phase == IN_ROUND:
                    draw_round_overlay(display, event)
                    if event.challenge_kind == "pill":
                        draw_pills(display, event)
                    elif event.challenge_kind == "spoon":
                        draw_spoon(display, event)
                elif event.phase == SCORE:
                    draw_score_screen(display, event)

            draw_celebration(display, event)
            rain.draw(display, boost=event.rain_boost, white=event.celebration_key == "white_rabbit")
            draw_flash(display, event)

            tick = time.perf_counter()
            dt = tick - last_tick
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)
            last_tick = tick

            draw_hud(display, fps=fps, persons=persons, faces=faces, poses=poses, event=event)

            cv2.imshow(cfg.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                current = cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN)
                if current == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    finally:
        publisher.close()
        if yolo_worker is not None:
            yolo_worker.close()
        if hand_landmarker is not None:
            hand_landmarker.close()
        if face_detector is not None:
            face_detector.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
