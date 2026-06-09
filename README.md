# Matrix Vision (Python + uv)

Real-time demo with a camera in Matrix style:
- Object/person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand gesture recognition (MediaPipe Hands)
- Fullscreen neon green HUD + Matrix rain effect
- Optional MQTT trigger to your NodeMCU servo project

## 1) Install with uv

From repository root:

```bash
uv sync --project matrix_motion
```

## 2) Run

Minimal run (MQTT optional):

```bash
uv run --project matrix_motion python main.py
```

Run with MQTT enabled (recommended for NodeMCU trigger):

```bash
uv run --project matrix_motion python main.py \
  --mqtt-host broker.hivemq.com \
  --mqtt-port 1883 \
  --mqtt-topic thematrix/pill/CHANGE_TO_UNIQUE_ID \
  --mqtt-token CHANGE_ME_TO_A_LONG_RANDOM_SECRET
```

## Controls

- `q` or `Esc`: quit
- `f`: toggle fullscreen

## Gestures

- `thumbs_up`: sends `blue` pill over MQTT (if MQTT is enabled)
- `ok_sign`: sends `red` pill over MQTT (if MQTT is enabled)
- `open_palm`: toggles Matrix boost visual mode

Payload published to MQTT:

```json
{"token":"<token>","pill":"red|blue"}
```

Compatible with your NodeMCU firmware payload format.

## Useful options

```bash
# Disable YOLO object detection (keep face + hand)
uv run --project matrix_motion python main.py --disable-yolo

# Disable MQTT completely
uv run --project matrix_motion python main.py --mqtt-disable

# Use another camera index
uv run --project matrix_motion python main.py --camera-index 1

# Windowed mode instead of fullscreen
uv run --project matrix_motion python main.py --windowed
```

## Raspberry Pi notes

- Start with lower resolution for better FPS:

```bash
uv run --project matrix_motion python main.py --width 960 --height 540 --imgsz 480
```

- If performance is low, increase `--yolo-stride` (ex: `3` or `4`) or use `--disable-yolo`.
