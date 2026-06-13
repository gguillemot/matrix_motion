from __future__ import annotations

import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from src.config import (
    BULLET_STOP_VIDEO_END,
    BULLET_STOP_VIDEO_START,
    BULLET_TIME_BUFFER_FRAMES,
    BULLET_TIME_PLAYBACK_FPS,
    MASK_BLUR_KSIZE,
    MASK_INVERT,
    MASK_SMOOTHING,
    MASK_THRESHOLD,
    PROJECT_ROOT,
    SEGMENTATION_ENABLED,
    AppConfig,
    parse_args,
)
from src.challenges import observe_hands, observe_pose
from src.game_engine import ATTRACT, BLUE_ENDING, COUNTDOWN, IN_ROUND, INTRO, PILL_CHOICE, SCORE, GameEngine
from src.mqtt_client import MQTTConfig, MQTTPublisher
from src.rendering import (
    MatrixRain,
    compose_matrix_scene,
    draw_attract_screen,
    draw_blue_ending,
    draw_celebration,
    draw_countdown,
    draw_face_detections,
    draw_flash,
    draw_hand_detections,
    draw_hud,
    draw_intro_hint,
    draw_pill_choice,
    draw_pose_skeleton,
    draw_ready_card,
    draw_round_overlay,
    draw_bullet_stop,
    draw_score_screen,
    draw_spoon,
    draw_yolo_detections,
)
from src.tracking import (
    PersonMaskTracker,
    YoloWorker,
    create_face_detector,
    create_hand_landmarker,
    create_image_segmenter,
    create_pose_landmarker,
)

_FFPLAY = shutil.which("ffplay")


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
    segmenter = None
    mask_tracker: PersonMaskTracker | None = None
    rain = None
    intro_path = Path(cfg.intro_video)
    has_intro = intro_path.exists()
    if has_intro:
        print(f"[INTRO] clip found: {intro_path} ({cfg.intro_start:.0f}s -> {cfg.intro_end:.0f}s, SPACE to skip)")
    else:
        print("[INTRO] no clip at assets/intro.mp4, intro skipped")

    engine = GameEngine(
        round_duration=cfg.round_duration,
        countdown_duration=cfg.countdown_duration,
        victory_pill=cfg.victory_pill,
        best_score_path=PROJECT_ROOT / "highscore.json",
        has_intro=has_intro,
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

        # Segmentation pour le fond "code rain" derriere la personne. Si le
        # modele ou la lib echoue, on retombe proprement sur l'ancienne pluie
        # par-dessus la camera (mask_tracker reste None).
        if SEGMENTATION_ENABLED:
            try:
                segmenter = create_image_segmenter()
                mask_tracker = PersonMaskTracker(
                    segmenter,
                    smoothing=MASK_SMOOTHING,
                    threshold=MASK_THRESHOLD,
                    blur_ksize=MASK_BLUR_KSIZE,
                    invert=MASK_INVERT,
                )
                print("[SEG] selfie segmenter loaded (code rain behind person)")
            except Exception as exc:
                print(f"[SEG] disabled (load failed): {exc} -- fallback rain overlay")
                segmenter = None
                mask_tracker = None

        video_start_ts = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera opened but no frame received.")

        frame = cv2.flip(frame, 1)
        rain = MatrixRain(frame.shape[1], frame.shape[0])

        frame_idx = 0
        last_tick = time.perf_counter()
        fps = 0.0

        # Bullet-time : on garde les dernieres frames propres pour les
        # rejouer au ralenti quand le Neo Dodge est reussi.
        frame_buffer: deque[np.ndarray] = deque(maxlen=BULLET_TIME_BUFFER_FRAMES)
        replay_frames: list[np.ndarray] = []
        replay_started_at = 0.0
        bullet_time_active = False

        # Lecteur du clip d'intro (phase INTRO), pilote par l'horloge murale.
        intro_cap: cv2.VideoCapture | None = None
        intro_started = 0.0
        intro_last_frame: np.ndarray | None = None
        _audio_proc: subprocess.Popen | None = None

        # Lecteur du clip "Neo Stops The Bullets" (celebration bullet_stop).
        bullet_stop_path = PROJECT_ROOT / "assets" / "stop_bullet.mp4"
        has_bullet_stop_video = bullet_stop_path.exists()
        bullet_stop_cap: cv2.VideoCapture | None = None
        bullet_stop_last_frame: np.ndarray | None = None
        bullet_stop_video_active = False
        bullet_stop_video_started = 0.0
        _bullet_stop_audio: subprocess.Popen | None = None

        def start_intro_audio() -> None:
            nonlocal _audio_proc
            if _FFPLAY is None:
                print("[AUDIO] ffplay not found – intro plays without audio")
                return
            dur = cfg.intro_end - cfg.intro_start
            _audio_proc = subprocess.Popen(
                [
                    _FFPLAY,
                    "-nodisp", "-autoexit",
                    "-ss", str(cfg.intro_start),
                    "-t",  str(dur),
                    str(intro_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def close_intro() -> None:
            nonlocal intro_cap, intro_last_frame, _audio_proc
            if _audio_proc is not None:
                _audio_proc.terminate()
                _audio_proc = None
            if intro_cap is not None:
                intro_cap.release()
            intro_cap = None
            intro_last_frame = None

        def start_bullet_stop_video(start_now: float) -> None:
            nonlocal bullet_stop_cap, bullet_stop_video_active, _bullet_stop_audio, bullet_stop_video_started
            if not has_bullet_stop_video:
                return
            bullet_stop_video_started = start_now
            bullet_stop_cap = cv2.VideoCapture(str(bullet_stop_path))
            bullet_stop_cap.set(cv2.CAP_PROP_POS_MSEC, BULLET_STOP_VIDEO_START * 1000.0)
            bullet_stop_video_active = True
            if _FFPLAY is not None:
                dur = BULLET_STOP_VIDEO_END - BULLET_STOP_VIDEO_START
                _bullet_stop_audio = subprocess.Popen(
                    [_FFPLAY, "-nodisp", "-autoexit",
                     "-ss", str(BULLET_STOP_VIDEO_START), "-t", str(dur),
                     str(bullet_stop_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        def close_bullet_stop_video() -> None:
            nonlocal bullet_stop_cap, bullet_stop_last_frame, bullet_stop_video_active, _bullet_stop_audio
            if _bullet_stop_audio is not None:
                _bullet_stop_audio.terminate()
                _bullet_stop_audio = None
            if bullet_stop_cap is not None:
                bullet_stop_cap.release()
            bullet_stop_cap = None
            bullet_stop_last_frame = None
            bullet_stop_video_active = False

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
            # buffer et on note l'instant de depart pour piloter la lecture
            # par l'horloge (cadence stable quel que soit le framerate).
            if event.celebration_key == "neo_dodge":
                if not bullet_time_active:
                    bullet_time_active = True
                    replay_frames = list(frame_buffer)
                    replay_started_at = now
            else:
                bullet_time_active = False
                replay_frames = []

            replay_frame = None
            if bullet_time_active and replay_frames:
                idx = int((now - replay_started_at) * BULLET_TIME_PLAYBACK_FPS)
                idx = min(idx, len(replay_frames) - 1)   # fige sur la derniere frame
                replay_frame = replay_frames[idx].copy()

            # Clip "Neo Stops The Bullets" : demarre a la premiere frame de
            # celebration et lit jusqu'a la fin de la fenetre (10 s).
            if event.celebration_key == "bullet_stop":
                if not bullet_stop_video_active:
                    start_bullet_stop_video(now)
                if bullet_stop_cap is not None:
                    elapsed = now - bullet_stop_video_started
                    target_ms = (BULLET_STOP_VIDEO_START + elapsed) * 1000.0
                    while bullet_stop_cap.get(cv2.CAP_PROP_POS_MSEC) < target_ms:
                        ok_v, vframe = bullet_stop_cap.read()
                        if not ok_v:
                            break
                        bullet_stop_last_frame = vframe
            else:
                if bullet_stop_video_active:
                    close_bullet_stop_video()

            cached_boxes = yolo_worker.get_boxes() if yolo_worker is not None else []

            # Masque personne pour le fond code rain. Calcule uniquement dans
            # les phases "Matrix" en direct (pas l'intro, pas la fin bleue, pas
            # le rejeu bullet-time qui rejoue des frames passees non segmentees).
            white_rain = event.celebration_key == "white_rabbit"
            matrix_scene = event.phase not in (INTRO, BLUE_ENDING) and replay_frame is None
            mask = None
            if mask_tracker is not None and matrix_scene:
                mask = mask_tracker.update(mp_image, timestamp_ms, (h_frame, w_frame))

            if event.phase == INTRO:
                # Clip d'intro : la camera et le moteur continuent de tourner,
                # seul l'affichage est remplace par la video (sans audio).
                if intro_cap is None:
                    intro_cap = cv2.VideoCapture(str(intro_path))
                    intro_cap.set(cv2.CAP_PROP_POS_MSEC, cfg.intro_start * 1000.0)
                    intro_started = now
                    start_intro_audio()

                elapsed = now - intro_started
                intro_over = elapsed >= max(0.5, cfg.intro_end - cfg.intro_start)
                target_ms = (cfg.intro_start + elapsed) * 1000.0
                while not intro_over and intro_cap.get(cv2.CAP_PROP_POS_MSEC) < target_ms:
                    ok_video, video_frame = intro_cap.read()
                    if not ok_video:
                        intro_over = True
                        break
                    intro_last_frame = video_frame

                if intro_over:
                    close_intro()
                    engine.finish_intro(now)

                display = np.zeros_like(frame)
                if intro_last_frame is not None:
                    vh, vw = intro_last_frame.shape[:2]
                    scale = min(w_frame / vw, h_frame / vh)
                    new_w, new_h = int(vw * scale), int(vh * scale)
                    resized = cv2.resize(intro_last_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    off_x, off_y = (w_frame - new_w) // 2, (h_frame - new_h) // 2
                    display[off_y : off_y + new_h, off_x : off_x + new_w] = resized
                draw_intro_hint(display)
                persons = faces = poses = 0

            elif event.phase == BLUE_ENDING:
                # Pilule bleue : camera brute, aucun effet Matrix, ni pluie ni
                # HUD ni squelettes. Juste l'invite a rejouer.
                display = frame
                draw_blue_ending(display, event)
                persons = faces = poses = 0

            elif replay_frame is not None:
                # Pendant le rejeu : pas d'overlays de detection ni d'ecran de
                # round, juste l'effet bullet-time sur les frames du passe.
                display = replay_frame
                persons = draw_yolo_detections(display, cached_boxes, getattr(model, "names", {}), draw=False)
                faces = len(face_results.detections or [])
                poses = len(pose_results.pose_landmarks or [])

            elif bullet_stop_last_frame is not None:
                # Clip "Neo stops the bullets" : remplace la camera pendant la celebration.
                display = np.zeros_like(frame)
                vh, vw = bullet_stop_last_frame.shape[:2]
                scale = min(w_frame / vw, h_frame / vh)
                new_w, new_h = int(vw * scale), int(vh * scale)
                resized = cv2.resize(bullet_stop_last_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                off_x = (w_frame - new_w) // 2
                off_y = (h_frame - new_h) // 2
                display[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                persons = draw_yolo_detections(display, cached_boxes, getattr(model, "names", {}), draw=False)
                faces = len(face_results.detections or [])
                poses = len(pose_results.pose_landmarks or [])

            else:
                # Scene signature : la pluie de code tombe DERRIERE la personne
                # segmentee. Si le masque est indisponible, on garde la frame
                # brute et la pluie sera dessinee par-dessus plus bas (fallback).
                if mask is not None:
                    display = compose_matrix_scene(frame, mask, rain, boost=event.rain_boost, white=white_rain)
                else:
                    display = frame
                # Les boites YOLO ne sont dessinees qu'en attract : pendant la
                # partie, elles parasitent la lecture des figures.
                persons = draw_yolo_detections(
                    display, cached_boxes, getattr(model, "names", {}), draw=event.phase == ATTRACT
                )

                faces = draw_face_detections(display, face_results.detections)

                draw_hand_detections(display, hand_results, w_frame, h_frame)
                poses = draw_pose_skeleton(display, pose_results, w_frame, h_frame)

                if event.phase == ATTRACT:
                    draw_attract_screen(display, event)
                elif event.phase == PILL_CHOICE:
                    draw_pill_choice(display, event)
                elif event.phase == COUNTDOWN:
                    draw_countdown(display, event)
                elif event.phase == IN_ROUND:
                    if event.celebration_key:
                        pass  # celebration en cours : draw_celebration gere seul, pas d'overlay figure
                    elif not event.figure_active:
                        draw_ready_card(display, event)  # ecran "prepare-toi" avant la figure
                    else:
                        draw_round_overlay(display, event)
                        if event.challenge_kind == "spoon":
                            draw_spoon(display, event)
                        elif event.challenge_kind == "bullet_stop":
                            draw_bullet_stop(display, event)
                elif event.phase == SCORE:
                    draw_score_screen(display, event)

            if event.phase not in (INTRO, BLUE_ENDING):
                draw_celebration(display, event)
                # Si le masque a deja place la pluie en fond, ne pas la
                # redessiner par-dessus (sinon on noie la personne).
                if mask is None:
                    rain.draw(display, boost=event.rain_boost, white=white_rain)
                draw_flash(display, event)

            tick = time.perf_counter()
            dt = tick - last_tick
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0 else (0.9 * fps + 0.1 * instant_fps)
            last_tick = tick

            if event.phase not in (INTRO, BLUE_ENDING):
                draw_hud(display, fps=fps, persons=persons, faces=faces, poses=poses, event=event)

            cv2.imshow(cfg.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == 32 and event.phase == INTRO:  # ESPACE : passer l'intro
                close_intro()
                engine.finish_intro(time.monotonic())
            if key == ord("f"):
                current = cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN)
                if current == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    finally:
        publisher.close()
        try:
            close_intro()
            close_bullet_stop_video()
        except NameError:
            pass  # boucle jamais atteinte
        if yolo_worker is not None:
            yolo_worker.close()
        if hand_landmarker is not None:
            hand_landmarker.close()
        if face_detector is not None:
            face_detector.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        if segmenter is not None:
            segmenter.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
