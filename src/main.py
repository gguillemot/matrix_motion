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
    TRINITY_VIDEO_END,
    TRINITY_VIDEO_START,
    AppConfig,
    parse_args,
)
from src.challenges import observe_hands, observe_pose
from src.game_engine import (
    ATTRACT,
    BLUE_ENDING,
    COUNTDOWN,
    IN_ROUND,
    INTRO,
    PILL_CHOICE,
    SCORE,
    TRINITY_OUTRO,
    GameEngine,
)
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
    draw_quiz,
    draw_ready_card,
    draw_round_overlay,
    draw_bullet_stop,
    draw_outro_qr,
    draw_score_screen,
    draw_spoon,
    draw_trinity_outro_text,
    draw_yolo_detections,
)
from src.tracking import (
    MediaPipeInferenceCache,
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
        print(
            "[INFO] MQTT topic/token still on placeholders. Set --mqtt-topic and --mqtt-token."
        )

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera. Try --camera-index 1 or verify camera permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    if not cfg.windowed:
        cv2.setWindowProperty(
            cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

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
    if cfg.benchmark_seconds > 0:
        has_intro = False
        print(
            f"[BENCH] benchmark mode enabled for {cfg.benchmark_seconds:.1f}s (intro disabled)"
        )
    if has_intro:
        print(
            f"[INTRO] clip found: {intro_path} ({cfg.intro_start:.0f}s -> {cfg.intro_end:.0f}s, SPACE to skip)"
        )
    else:
        if cfg.benchmark_seconds > 0:
            print("[INTRO] skipped during benchmark")
        else:
            print("[INTRO] no clip at assets/intro.mp4, intro skipped")

    engine = GameEngine(
        round_duration=cfg.round_duration,
        countdown_duration=cfg.countdown_duration,
        victory_pill=cfg.victory_pill,
        best_score_path=PROJECT_ROOT / "highscore.json",
        has_intro=has_intro,
    )

    # Resolve runtime performance knobs from presets or keep manual values.
    resolved_mp_scale = cfg.mp_scale
    resolved_mp_hand_stride = cfg.mp_hand_stride
    resolved_mp_face_stride = cfg.mp_face_stride
    resolved_mp_pose_stride = cfg.mp_pose_stride
    resolved_mp_seg_stride = cfg.mp_seg_stride
    resolved_yolo_stride = cfg.yolo_stride
    resolved_yolo_device = cfg.yolo_device
    resolved_yolo_half = cfg.yolo_half

    detected_gpu = False
    try:
        import torch

        detected_gpu = torch.cuda.is_available()
    except Exception:
        detected_gpu = False

    if cfg.perf_mode != "manual":
        use_gpu = detected_gpu
        if cfg.perf_target == "cpu":
            use_gpu = False
        elif cfg.perf_target == "gpu" and not detected_gpu:
            print("[OPTIM] perf-target=gpu requested but no CUDA/ROCm backend detected; fallback to CPU preset")
            use_gpu = False

        if use_gpu:
            presets = {
                "quality": (0.95, 1, 1, 1, 1, 2, "cuda:0", False),
                "balanced": (0.85, 1, 1, 1, 2, 3, "cuda:0", True),
                "fast": (0.75, 1, 1, 2, 2, 4, "cuda:0", True),
            }
            hw = "gpu"
        else:
            presets = {
                "quality": (0.80, 2, 2, 2, 3, 3, "cpu", False),
                "balanced": (0.70, 2, 2, 3, 3, 4, "cpu", False),
                "fast": (0.60, 3, 3, 3, 4, 5, "cpu", False),
            }
            hw = "cpu"

        (
            resolved_mp_scale,
            resolved_mp_hand_stride,
            resolved_mp_face_stride,
            resolved_mp_pose_stride,
            resolved_mp_seg_stride,
            resolved_yolo_stride,
            resolved_yolo_device,
            resolved_yolo_half,
        ) = presets[cfg.perf_mode]
        print(
            "[OPTIM] preset="
            f"{cfg.perf_mode}/{hw} "
            f"mp_scale={resolved_mp_scale:.2f} "
            f"strides(h/f/p/s)={resolved_mp_hand_stride}/{resolved_mp_face_stride}/{resolved_mp_pose_stride}/{resolved_mp_seg_stride} "
            f"yolo_stride={resolved_yolo_stride} yolo_device={resolved_yolo_device} yolo_half={resolved_yolo_half}"
        )

    try:
        if not cfg.disable_yolo:
            try:
                model = YOLO(cfg.model)
                yolo_device = resolved_yolo_device
                if yolo_device == "auto":
                    yolo_device = "cpu"
                    try:
                        import torch

                        if torch.cuda.is_available():
                            yolo_device = "cuda:0"
                    except Exception:
                        yolo_device = "cpu"

                try:
                    model.to(yolo_device)
                except Exception as exc:
                    print(f"[YOLO] requested device '{yolo_device}' failed ({exc}), fallback to cpu")
                    yolo_device = "cpu"
                    model.to(yolo_device)

                if resolved_yolo_half and yolo_device.startswith("cuda"):
                    try:
                        raw_model = getattr(model, "model", None)
                        half_fn = getattr(raw_model, "half", None)
                        if callable(half_fn):
                            half_fn()
                            print("[YOLO] FP16 enabled")
                        else:
                            print("[YOLO] FP16 disabled (model backend has no half())")
                    except Exception as exc:
                        print(f"[YOLO] FP16 disabled (not supported): {exc}")

                yolo_worker = YoloWorker(model, cfg.imgsz, cfg.conf)
                print(
                    f"[YOLO] loaded {cfg.model} on {yolo_device} (background thread, imgsz={cfg.imgsz})"
                )
            except Exception as exc:
                print(f"[YOLO] disabled (load failed): {exc}")

        hand_landmarker = create_hand_landmarker()
        print("[HAND] landmarker loaded (GPU delegate if available)")
        face_detector = create_face_detector()
        print("[FACE] detector loaded (GPU delegate if available)")
        pose_landmarker = create_pose_landmarker()
        print("[POSE] landmarker loaded (GPU delegate if available)")

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
                print("[SEG] selfie segmenter loaded (GPU delegate if available, code rain behind person)")
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

        bench_enabled = cfg.benchmark_seconds > 0
        bench_target_runs = cfg.benchmark_runs
        bench_current_run = 1
        bench_started_at = 0.0
        bench_frames = 0
        bench_sum_fps = 0.0
        bench_min_fps = float("inf")
        bench_max_fps = 0.0
        bench_hand_runs = 0
        bench_face_runs = 0
        bench_pose_runs = 0
        bench_seg_runs = 0
        bench_reported = False
        bench_run_avgs: list[float] = []

        def emit_bench_summary(reason: str) -> None:
            nonlocal bench_reported
            if not bench_enabled or bench_reported:
                return
            bench_reported = True
            elapsed = max(0.001, time.perf_counter() - bench_started_at) if bench_started_at > 0 else 0.0
            avg_fps = bench_sum_fps / max(1, bench_frames)
            min_fps = bench_min_fps if bench_frames > 0 else 0.0
            max_fps = bench_max_fps if bench_frames > 0 else 0.0
            bench_run_avgs.append(avg_fps)
            hand_rate = bench_hand_runs / max(1, bench_frames)
            face_rate = bench_face_runs / max(1, bench_frames)
            pose_rate = bench_pose_runs / max(1, bench_frames)
            seg_rate = bench_seg_runs / max(1, bench_frames)
            print(f"[BENCH] run {bench_current_run}/{bench_target_runs} done ({reason})")
            print(
                "[BENCH] "
                f"elapsed={elapsed:.1f}s frames={bench_frames} "
                f"fps_avg={avg_fps:.1f} fps_min={min_fps:.1f} fps_max={max_fps:.1f}"
            )
            print(
                "[BENCH] infer/frame "
                f"hand={hand_rate:.2f} face={face_rate:.2f} pose={pose_rate:.2f} seg={seg_rate:.2f}"
            )

        def reset_bench_run(now_tick: float) -> None:
            nonlocal bench_started_at, bench_frames, bench_sum_fps, bench_min_fps, bench_max_fps
            nonlocal bench_hand_runs, bench_face_runs, bench_pose_runs, bench_seg_runs, bench_reported
            bench_started_at = now_tick
            bench_frames = 0
            bench_sum_fps = 0.0
            bench_min_fps = float("inf")
            bench_max_fps = 0.0
            bench_hand_runs = 0
            bench_face_runs = 0
            bench_pose_runs = 0
            bench_seg_runs = 0
            bench_reported = False

        hand_cache = MediaPipeInferenceCache(resolved_mp_hand_stride)
        face_cache = MediaPipeInferenceCache(resolved_mp_face_stride)
        pose_cache = MediaPipeInferenceCache(resolved_mp_pose_stride)
        seg_cache = MediaPipeInferenceCache(resolved_mp_seg_stride)
        print(
            "[OPTIM] MediaPipe: "
            f"scale={resolved_mp_scale:.2f} "
            f"strides(hand/face/pose/seg)="
            f"{resolved_mp_hand_stride}/{resolved_mp_face_stride}/{resolved_mp_pose_stride}/{resolved_mp_seg_stride}"
        )

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

        # Lecteur du clip outro "Trinity" (phase TRINITY_OUTRO, gele sur 1:44).
        trinity_path = PROJECT_ROOT / "assets" / "trinity.mp4"
        has_trinity = trinity_path.exists()
        trinity_cap: cv2.VideoCapture | None = None
        trinity_last_frame: np.ndarray | None = None
        trinity_started = 0.0
        _trinity_audio: subprocess.Popen | None = None
        trinity_clip_dur = TRINITY_VIDEO_END - TRINITY_VIDEO_START
        # Theme de fond joue pendant toute la scene outro (en plus de l'audio natif
        # du clip), coupe au retour en attract.
        trinity_theme_path = PROJECT_ROOT / "assets" / "trinity_theme.m4a"
        has_trinity_theme = trinity_theme_path.exists()
        _trinity_theme: subprocess.Popen | None = None

        def start_intro_audio() -> None:
            nonlocal _audio_proc
            if _FFPLAY is None:
                print("[AUDIO] ffplay not found – intro plays without audio")
                return
            dur = cfg.intro_end - cfg.intro_start
            _audio_proc = subprocess.Popen(
                [
                    _FFPLAY,
                    "-nodisp",
                    "-autoexit",
                    "-ss",
                    str(cfg.intro_start),
                    "-t",
                    str(dur),
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
            nonlocal \
                bullet_stop_cap, \
                bullet_stop_video_active, \
                _bullet_stop_audio, \
                bullet_stop_video_started
            if not has_bullet_stop_video:
                return
            bullet_stop_video_started = start_now
            bullet_stop_cap = cv2.VideoCapture(str(bullet_stop_path))
            bullet_stop_cap.set(cv2.CAP_PROP_POS_MSEC, BULLET_STOP_VIDEO_START * 1000.0)
            bullet_stop_video_active = True
            if _FFPLAY is not None:
                dur = BULLET_STOP_VIDEO_END - BULLET_STOP_VIDEO_START
                _bullet_stop_audio = subprocess.Popen(
                    [
                        _FFPLAY,
                        "-nodisp",
                        "-autoexit",
                        "-ss",
                        str(BULLET_STOP_VIDEO_START),
                        "-t",
                        str(dur),
                        str(bullet_stop_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        def close_bullet_stop_video() -> None:
            nonlocal \
                bullet_stop_cap, \
                bullet_stop_last_frame, \
                bullet_stop_video_active, \
                _bullet_stop_audio
            if _bullet_stop_audio is not None:
                _bullet_stop_audio.terminate()
                _bullet_stop_audio = None
            if bullet_stop_cap is not None:
                bullet_stop_cap.release()
            bullet_stop_cap = None
            bullet_stop_last_frame = None
            bullet_stop_video_active = False

        def start_trinity_outro(start_now: float) -> None:
            nonlocal trinity_cap, trinity_started, _trinity_audio, _trinity_theme
            if not has_trinity:
                return
            trinity_started = start_now
            trinity_cap = cv2.VideoCapture(str(trinity_path))
            trinity_cap.set(cv2.CAP_PROP_POS_MSEC, TRINITY_VIDEO_START * 1000.0)
            if _FFPLAY is not None:
                # Audio natif du clip (whoosh bullet-time, ~3 s).
                _trinity_audio = subprocess.Popen(
                    [
                        _FFPLAY,
                        "-nodisp",
                        "-autoexit",
                        "-ss",
                        str(TRINITY_VIDEO_START),
                        "-t",
                        str(trinity_clip_dur),
                        str(trinity_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Theme de fond, joue en entier (coupe au retour en attract).
                if has_trinity_theme:
                    _trinity_theme = subprocess.Popen(
                        [
                            _FFPLAY,
                            "-nodisp",
                            "-autoexit",
                            str(trinity_theme_path),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

        def close_trinity_outro() -> None:
            nonlocal trinity_cap, trinity_last_frame, _trinity_audio, _trinity_theme
            if _trinity_audio is not None:
                _trinity_audio.terminate()
                _trinity_audio = None
            if _trinity_theme is not None:
                _trinity_theme.terminate()
                _trinity_theme = None
            if trinity_cap is not None:
                trinity_cap.release()
            trinity_cap = None
            trinity_last_frame = None

        while True:
            ok, frame = cap.read()
            if not ok:
                emit_bench_summary("camera")
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.perf_counter() - video_start_ts) * 1000)
            if resolved_mp_scale < 1.0:
                rgb_infer = cv2.resize(
                    rgb,
                    None,
                    fx=resolved_mp_scale,
                    fy=resolved_mp_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                rgb_infer = rgb

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_infer)

            if hand_cache.should_run() or hand_cache.get() is None:
                hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_cache.set(hand_results)
                bench_hand_runs += 1
            else:
                hand_results = hand_cache.get()
            if hand_results is None:
                hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_cache.set(hand_results)
                bench_hand_runs += 1

            if face_cache.should_run() or face_cache.get() is None:
                face_results = face_detector.detect_for_video(mp_image, timestamp_ms)
                face_cache.set(face_results)
                bench_face_runs += 1
            else:
                face_results = face_cache.get()
            if face_results is None:
                face_results = face_detector.detect_for_video(mp_image, timestamp_ms)
                face_cache.set(face_results)
                bench_face_runs += 1

            if pose_cache.should_run() or pose_cache.get() is None:
                pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                pose_cache.set(pose_results)
                bench_pose_runs += 1
            else:
                pose_results = pose_cache.get()
            if pose_results is None:
                pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                pose_cache.set(pose_results)
                bench_pose_runs += 1
            h_frame, w_frame = frame.shape[:2]

            if yolo_worker is not None and frame_idx % resolved_yolo_stride == 0:
                yolo_worker.submit(frame.copy())

            hands = observe_hands(hand_results)
            pose_observation = observe_pose(pose_results)

            now = time.monotonic()
            event = engine.update(hands, pose_observation, now, publisher.publish_pill)
            if event.published:
                print(f"[MQTT] pill published: {event.chosen_pill or 'default'}")

            # Keep bullet-time source frames only when Neo Dodge can use them.
            should_buffer_for_bullet_time = (
                event.phase == IN_ROUND
                and (
                    event.challenge_kind == "neo_dodge"
                    or event.celebration_key == "neo_dodge"
                    or bullet_time_active
                )
            )
            if should_buffer_for_bullet_time:
                frame_buffer.append(frame.copy())

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
                idx = min(idx, len(replay_frames) - 1)  # fige sur la derniere frame
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

            # Outro Trinity : libere le lecteur des qu'on quitte la phase.
            if event.phase != TRINITY_OUTRO and trinity_cap is not None:
                close_trinity_outro()

            cached_boxes = yolo_worker.get_boxes() if yolo_worker is not None else []

            # Masque personne pour le fond code rain. Calcule uniquement dans
            # les phases "Matrix" en direct (pas l'intro, pas la fin bleue, pas
            # le rejeu bullet-time qui rejoue des frames passees non segmentees).
            white_rain = event.celebration_key == "white_rabbit"
            matrix_scene = (
                event.phase not in (INTRO, BLUE_ENDING, TRINITY_OUTRO)
                and replay_frame is None
            )
            mask = None
            if mask_tracker is not None and matrix_scene:
                if seg_cache.should_run() or seg_cache.get() is None:
                    mask = mask_tracker.update(mp_image, timestamp_ms, (h_frame, w_frame))
                    seg_cache.set(mask)
                    bench_seg_runs += 1
                else:
                    mask = seg_cache.get()

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
                while (
                    not intro_over and intro_cap.get(cv2.CAP_PROP_POS_MSEC) < target_ms
                ):
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
                    resized = cv2.resize(
                        intro_last_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )
                    off_x, off_y = (w_frame - new_w) // 2, (h_frame - new_h) // 2
                    display[off_y : off_y + new_h, off_x : off_x + new_w] = resized
                draw_intro_hint(display)
                persons = faces = poses = 0

            elif event.phase == TRINITY_OUTRO:
                # Clip outro "Trinity" : joue 1:41 -> 1:44, gele sur la derniere
                # frame, puis affiche le texte. Pilote par l'horloge murale.
                if trinity_cap is None:
                    start_trinity_outro(now)

                elapsed = now - trinity_started
                if trinity_cap is not None:
                    target_ms = (
                        TRINITY_VIDEO_START + min(elapsed, trinity_clip_dur)
                    ) * 1000.0
                    while trinity_cap.get(cv2.CAP_PROP_POS_MSEC) < target_ms:
                        ok_video, video_frame = trinity_cap.read()
                        if not ok_video:
                            break
                        trinity_last_frame = video_frame

                display = np.zeros_like(frame)
                if trinity_last_frame is not None:
                    vh, vw = trinity_last_frame.shape[:2]
                    scale = min(w_frame / vw, h_frame / vh)
                    new_w, new_h = int(vw * scale), int(vh * scale)
                    resized = cv2.resize(
                        trinity_last_frame,
                        (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    off_x, off_y = (w_frame - new_w) // 2, (h_frame - new_h) // 2
                    display[off_y : off_y + new_h, off_x : off_x + new_w] = resized
                # Texte + QR affiches une fois la frame gelee (clip termine).
                if elapsed >= trinity_clip_dur:
                    draw_trinity_outro_text(display)
                    draw_outro_qr(display)
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
                persons = draw_yolo_detections(
                    display, cached_boxes, getattr(model, "names", {}), draw=False
                )
                faces = len(face_results.detections or [])
                poses = len(pose_results.pose_landmarks or [])

            elif bullet_stop_last_frame is not None:
                # Clip "Neo stops the bullets" : remplace la camera pendant la celebration.
                display = np.zeros_like(frame)
                vh, vw = bullet_stop_last_frame.shape[:2]
                scale = min(w_frame / vw, h_frame / vh)
                new_w, new_h = int(vw * scale), int(vh * scale)
                resized = cv2.resize(
                    bullet_stop_last_frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                off_x = (w_frame - new_w) // 2
                off_y = (h_frame - new_h) // 2
                display[off_y : off_y + new_h, off_x : off_x + new_w] = resized
                persons = draw_yolo_detections(
                    display, cached_boxes, getattr(model, "names", {}), draw=False
                )
                faces = len(face_results.detections or [])
                poses = len(pose_results.pose_landmarks or [])

            else:
                # Scene signature : la pluie de code tombe DERRIERE la personne
                # segmentee. Si le masque est indisponible, on garde la frame
                # brute et la pluie sera dessinee par-dessus plus bas (fallback).
                if mask is not None:
                    display = compose_matrix_scene(
                        frame, mask, rain, boost=event.rain_boost, white=white_rain
                    )
                else:
                    display = frame
                # Les boites YOLO ne sont dessinees qu'en attract : pendant la
                # partie, elles parasitent la lecture des figures.
                persons = draw_yolo_detections(
                    display,
                    cached_boxes,
                    getattr(model, "names", {}),
                    draw=event.phase == ATTRACT,
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
                        draw_ready_card(
                            display, event
                        )  # ecran "prepare-toi" avant la figure
                    elif event.challenge_kind == "quiz":
                        draw_quiz(display, event)
                    else:
                        draw_round_overlay(display, event)
                        if event.challenge_kind == "spoon":
                            draw_spoon(display, event)
                        elif event.challenge_kind == "bullet_stop":
                            draw_bullet_stop(display, event)
                elif event.phase == SCORE:
                    draw_score_screen(display, event)

            if event.phase not in (INTRO, BLUE_ENDING, TRINITY_OUTRO):
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

            if bench_enabled:
                if bench_started_at == 0.0:
                    bench_started_at = tick
                bench_frames += 1
                bench_sum_fps += fps
                bench_min_fps = min(bench_min_fps, fps)
                bench_max_fps = max(bench_max_fps, fps)
                if (tick - bench_started_at) >= cfg.benchmark_seconds:
                    emit_bench_summary("timeout")
                    if bench_current_run >= bench_target_runs:
                        if bench_run_avgs:
                            avg_over_runs = sum(bench_run_avgs) / len(bench_run_avgs)
                            print(
                                "[BENCH] all-runs "
                                f"runs={len(bench_run_avgs)} fps_avg_mean={avg_over_runs:.1f} "
                                f"fps_avg_best={max(bench_run_avgs):.1f} fps_avg_worst={min(bench_run_avgs):.1f}"
                            )
                        break

                    bench_current_run += 1
                    print(f"[BENCH] starting run {bench_current_run}/{bench_target_runs}")
                    reset_bench_run(tick)

            if event.phase not in (INTRO, BLUE_ENDING, TRINITY_OUTRO):
                draw_hud(
                    display,
                    fps=fps,
                    persons=persons,
                    faces=faces,
                    poses=poses,
                    event=event,
                )

            cv2.imshow(cfg.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                emit_bench_summary("user")
                break
            if key == 32 and event.phase == INTRO:  # ESPACE : passer l'intro
                close_intro()
                engine.finish_intro(time.monotonic())
            if key == 32 and event.phase == TRINITY_OUTRO:  # ESPACE : sortir de l'outro
                close_trinity_outro()
                engine.skip_outro(time.monotonic())
            if key == ord("f"):
                current = cv2.getWindowProperty(
                    cfg.window_name, cv2.WND_PROP_FULLSCREEN
                )
                if current == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(
                        cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                else:
                    cv2.setWindowProperty(
                        cfg.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
    finally:
        publisher.close()
        try:
            close_intro()
            close_bullet_stop_video()
            close_trinity_outro()
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
