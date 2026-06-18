from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Segmentation "code rain" (feature signature)
# ---------------------------------------------------------------------------
# La personne est segmentee (MediaPipe ImageSegmenter / Selfie Segmentation)
# et la pluie de caracteres tombe DERRIERE elle. Toutes les constantes de
# reglage du rendu sont ici pour pouvoir ajuster vite sur le stand.

# Active le fond code rain par segmentation. Si False (ou si le modele/lib
# manque), on retombe sur l'ancienne pluie dessinee par-dessus la camera.
SEGMENTATION_ENABLED = True

# Lissage temporel du masque : nouveau_masque = a*courant + (1-a)*precedent.
# Plus c'est bas, plus c'est stable (mais moins reactif). 0.5 = bon compromis.
MASK_SMOOTHING = 0.5

# Seuil de decision personne/fond avant adoucissement des bords (0..1).
MASK_THRESHOLD = 0.35

# Rayon du flou gaussien applique au masque pour des bords doux (impair).
MASK_BLUR_KSIZE = 21

# Si le modele renvoie le masque inverse (fond=1, personne=0), passer a True.
MASK_INVERT = False

# Teinte verte appliquee a la personne au premier plan (0 = couleur brute,
# 1 = tres vert Matrix). Donne l'impression d'etre "dans" la Matrix.
FOREGROUND_GREEN_TINT = 0.22

# Fond de la pluie : noir tres legerement teinte vert (BGR), plus cinematique
# qu'un noir pur derriere les caracteres.
RAIN_BACKGROUND_COLOR = (8, 14, 8)

# ---------------------------------------------------------------------------
# Bullet-time : replay au ralenti apres le Neo Dodge reussi
# ---------------------------------------------------------------------------
# Duree de la fenetre bullet-time = duree de la celebration ET de la pause
# avant la figure suivante, UNIQUEMENT pour le Neo Dodge (les autres figures
# gardent ROUND_TRANSITION_SEC). Laisse le temps de re-regarder l'esquive.
BULLET_TIME_REPLAY_SEC = 4.5

# Nombre de frames de camera gardees avant le succes pour etre rejouees.
# ~48 frames = ~2 s de footage selon le framerate de la boucle.
BULLET_TIME_BUFFER_FRAMES = 48

# Cadence de relecture du buffer, en frames/seconde (horloge murale). Basse =
# tres ralenti. 11 fps sur ~2 s de footage capture a ~22 fps => ~2x slow motion.
BULLET_TIME_PLAYBACK_FPS = 11.0

# ---------------------------------------------------------------------------
# "Neo Stops The Bullets" : clip video joue a la reussite de la figure
# ---------------------------------------------------------------------------
# Duree de la fenetre celebration + transition = duree du clip (10 s).
BULLET_STOP_VIDEO_SEC = 10.0
BULLET_STOP_VIDEO_START = 130.0  # 2:10 dans le fichier source
BULLET_STOP_VIDEO_END = 140.0  # 2:20

# ---------------------------------------------------------------------------
# "Trinity" : clip outro joue 4 s apres l'ecran de score, gele sur la derniere
# frame (1:44) avec son natif en guise d'effet bullet-time sci-fi, puis texte.
# ---------------------------------------------------------------------------
TRINITY_VIDEO_START = 101.0  # 1:41
TRINITY_VIDEO_END = 104.0  # 1:44
TRINITY_FREEZE_HOLD_SEC = 10.0  # duree du gel + texte avant retour auto a l'attract

# ---------------------------------------------------------------------------
# Quiz Matrix : defis QCM repondus au geste (pointage + maintien)
# ---------------------------------------------------------------------------
# Un defi "quiz" pose une question sur l'univers Matrix ; le joueur pointe sa
# reponse avec l'index (landmark 8) et la maintient QUIZ_DWELL_SEC pour valider.
# Le quiz alterne strictement avec les figures de geste (figure, quiz, figure,
# quiz...). Toutes les constantes sont ici pour ajuster vite sur le stand.

# Active l'insertion de defis quiz dans la campagne. Si False, partie 100 %
# figures comme avant.
QUIZ_ENABLED = True

# Banque de questions externalisee (editable sans toucher au code). Si le
# fichier manque ou est invalide, un fallback integre prend le relais.
QUIZ_QUESTIONS_PATH = PROJECT_ROOT / "assets" / "quiz" / "questions.json"

# Temps imparti par defaut pour repondre (une question peut le surcharger via
# son champ "duration_s").
QUIZ_DEFAULT_DURATION_S = 10.0

# Duree de l'ecran d'intro (question affichee, chrono fige) avant le decompte.
QUIZ_INTRO_SEC = 1.5

# Duree de l'ecran de revelation (bonne/mauvaise reponse + explication).
QUIZ_REVEAL_SEC = 2.5

# Maintien du doigt sur une reponse pour la valider (dwell).
QUIZ_DWELL_SEC = 1.2

# Grace en debut de phase active : on ignore le pointage le temps que la main
# arrive, pour eviter une validation involontaire.
QUIZ_GRACE_SEC = 0.3

# Nombre maximum de reponses gerees par question (layout 2x2).
QUIZ_MAX_ANSWERS = 4

# Nombre maximum de quiz inseres dans une partie (cap pour garder une session
# de stand courte malgre l'alternance stricte).
QUIZ_MAX_PER_GAME = 4

# Marge anti-bord (fraction de la carte) : la main doit etre franchement dans
# la zone pour compter, pas a cheval sur le bord.
QUIZ_ZONE_MARGIN = 0.06

# Mecanique de selection : "A" = pointage + maintien (seule implementee).
# "B" (comptage de doigts) et "C" (inclinaison) sont reserves pour plus tard.
QUIZ_SELECTION_MODE = "A"

# Sons de succes/echec du quiz : non cables pour l'instant (feedback visuel).
QUIZ_SOUND_ENABLED = False


@dataclass(slots=True)
class AppConfig:
    camera_index: int
    width: int
    height: int
    model: str
    yolo_device: str
    yolo_half: bool
    perf_mode: str
    perf_target: str
    conf: float
    imgsz: int
    yolo_stride: int
    mp_scale: float
    mp_hand_stride: int
    mp_face_stride: int
    mp_pose_stride: int
    mp_seg_stride: int
    disable_yolo: bool
    window_name: str
    windowed: bool
    mqtt_disable: bool
    mqtt_host: str
    mqtt_port: int
    mqtt_topic: str
    mqtt_token: str
    mqtt_client_id: str
    sequence_length: int
    victory_pill: str
    round_duration: float
    countdown_duration: float
    intro_video: str
    intro_start: float
    intro_end: float
    benchmark_seconds: float
    benchmark_runs: int
    badge_scan_first: bool = False
    mp_cpu: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matrix-style vision demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "yolov8n.pt"))
    parser.add_argument(
        "--yolo-device",
        type=str,
        default="auto",
        help="YOLO device: auto, cpu, cuda, cuda:0 ...",
    )
    parser.add_argument(
        "--yolo-half",
        action="store_true",
        help="Enable FP16 for YOLO when running on CUDA/ROCm",
    )
    parser.add_argument(
        "--perf-mode",
        type=str,
        choices=("manual", "quality", "balanced", "fast"),
        default="quality",
        help="Performance preset. 'manual' keeps explicit mp/yolo tuning flags.",
    )
    parser.add_argument(
        "--perf-target",
        type=str,
        choices=("auto", "cpu", "gpu"),
        default="gpu",
        help="Preset target hardware. 'auto' detects GPU availability.",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--yolo-stride", type=int, default=3)
    parser.add_argument(
        "--mp-scale",
        type=float,
        default=0.75,
        help="MediaPipe inference scale in (0,1], lower is faster",
    )
    parser.add_argument(
        "--mp-hand-stride",
        type=int,
        default=2,
        help="Run hand inference every N frames",
    )
    parser.add_argument(
        "--mp-face-stride",
        type=int,
        default=2,
        help="Run face inference every N frames",
    )
    parser.add_argument(
        "--mp-pose-stride",
        type=int,
        default=2,
        help="Run pose inference every N frames",
    )
    parser.add_argument(
        "--mp-seg-stride",
        type=int,
        default=3,
        help="Run segmentation every N frames",
    )
    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument(
        "--mp-cpu",
        action="store_true",
        help=(
            "force tous les modeles MediaPipe (main/visage/pose) sur le delegate "
            "CPU. A utiliser quand le delegate GPU plante (ex: Metal sur macOS). "
            "Le segmenter est deja toujours en CPU."
        ),
    )
    parser.add_argument("--window-name", type=str, default="THE MATRIX")
    parser.add_argument("--windowed", action="store_true")

    parser.add_argument("--mqtt-disable", action="store_true")
    parser.add_argument("--mqtt-host", type=str, default="broker.hivemq.com")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", type=str, default="")
    parser.add_argument("--mqtt-token", type=str, default="")
    parser.add_argument(
        "--mqtt-client-id",
        type=str,
        default=f"matrix-motion-{random.randint(1000, 9999)}",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=(5, 10),
        default=5,
        help="(legacy) ignore par le mode jeu BreizhCamp, qui joue toujours les 5 figures",
    )
    parser.add_argument(
        "--victory-pill",
        type=str,
        choices=("auto", "red", "blue"),
        default="auto",
        help="pilule MQTT par defaut si la figure pilule n'a pas ete reussie",
    )
    parser.add_argument(
        "--round-duration",
        type=float,
        default=8.0,
        help="duree en secondes laissee pour reussir chaque figure",
    )
    parser.add_argument(
        "--countdown-duration",
        type=float,
        default=3.0,
        help="duree du decompte 3-2-1 avant la premiere figure",
    )
    parser.add_argument(
        "--intro-video",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "intro.mp4"),
        help="clip video local joue avant le choix des pilules (saute si absent)",
    )
    parser.add_argument(
        "--intro-start",
        type=float,
        default=164.0,
        help="debut du segment du clip d'intro, en secondes (2:44)",
    )
    parser.add_argument(
        "--intro-end",
        type=float,
        default=178.0,
        help="fin du segment du clip d'intro, en secondes (2:58)",
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=0.0,
        help="run a timed benchmark and exit automatically (0 disables benchmark)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="number of benchmark runs to execute (used with --benchmark-seconds)",
    )
    parser.add_argument(
        "--badge-scan-first",
        action="store_true",
        help="demarre directement sur l'ecran de scan de badge (test)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        model=args.model,
        yolo_device=args.yolo_device,
        yolo_half=args.yolo_half,
        perf_mode=args.perf_mode,
        perf_target=args.perf_target,
        conf=args.conf,
        imgsz=args.imgsz,
        yolo_stride=max(1, args.yolo_stride),
        mp_scale=max(0.1, min(1.0, args.mp_scale)),
        mp_hand_stride=max(1, args.mp_hand_stride),
        mp_face_stride=max(1, args.mp_face_stride),
        mp_pose_stride=max(1, args.mp_pose_stride),
        mp_seg_stride=max(1, args.mp_seg_stride),
        disable_yolo=args.disable_yolo,
        window_name=args.window_name,
        windowed=args.windowed,
        mqtt_disable=args.mqtt_disable,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_token=args.mqtt_token,
        mqtt_client_id=args.mqtt_client_id,
        sequence_length=args.sequence_length,
        victory_pill=args.victory_pill,
        round_duration=args.round_duration,
        countdown_duration=args.countdown_duration,
        intro_video=args.intro_video,
        intro_start=args.intro_start,
        intro_end=args.intro_end,
        benchmark_seconds=max(0.0, args.benchmark_seconds),
        benchmark_runs=max(1, args.benchmark_runs),
        badge_scan_first=args.badge_scan_first,
        mp_cpu=args.mp_cpu,
    )
