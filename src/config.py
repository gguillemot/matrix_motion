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
BULLET_STOP_VIDEO_SEC   = 10.0
BULLET_STOP_VIDEO_START = 130.0  # 2:10 dans le fichier source
BULLET_STOP_VIDEO_END   = 140.0  # 2:20


@dataclass(slots=True)
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
    sequence_length: int
    victory_pill: str
    round_duration: float
    countdown_duration: float
    intro_video: str
    intro_start: float
    intro_end: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matrix-style vision demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "yolov8n.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--yolo-stride", type=int, default=3)
    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument("--window-name", type=str, default="THE MATRIX")
    parser.add_argument("--windowed", action="store_true")

    parser.add_argument("--mqtt-disable", action="store_true")
    parser.add_argument("--mqtt-host", type=str, default="broker.hivemq.com")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", type=str, default="")
    parser.add_argument("--mqtt-token", type=str, default="")
    parser.add_argument("--mqtt-client-id", type=str, default=f"matrix-motion-{random.randint(1000, 9999)}")
    parser.add_argument("--sequence-length", type=int, choices=(5, 10), default=5,
                        help="(legacy) ignore par le mode jeu BreizhCamp, qui joue toujours les 5 figures")
    parser.add_argument("--victory-pill", type=str, choices=("auto", "red", "blue"), default="auto",
                        help="pilule MQTT par defaut si la figure pilule n'a pas ete reussie")
    parser.add_argument("--round-duration", type=float, default=8.0,
                        help="duree en secondes laissee pour reussir chaque figure")
    parser.add_argument("--countdown-duration", type=float, default=3.0,
                        help="duree du decompte 3-2-1 avant la premiere figure")
    parser.add_argument("--intro-video", type=str, default=str(PROJECT_ROOT / "assets" / "intro.mp4"),
                        help="clip video local joue avant le choix des pilules (saute si absent)")
    parser.add_argument("--intro-start", type=float, default=164.0,
                        help="debut du segment du clip d'intro, en secondes (2:44)")
    parser.add_argument("--intro-end", type=float, default=178.0,
                        help="fin du segment du clip d'intro, en secondes (2:58)")
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
        sequence_length=args.sequence_length,
        victory_pill=args.victory_pill,
        round_duration=args.round_duration,
        countdown_duration=args.countdown_duration,
        intro_video=args.intro_video,
        intro_start=args.intro_start,
        intro_end=args.intro_end,
    )
