from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    )
