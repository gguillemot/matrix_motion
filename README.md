# Matrix Vision (Python + uv)

Real-time demo with a camera in Matrix style:
- Object/person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand gesture recognition (MediaPipe Hands)
- Fullscreen neon green HUD + Matrix rain effect
- Optional MQTT trigger to your NodeMCU servo project

## Structure

- `src/main.py`: entry point that wires the app together
- `src/config.py`: CLI and environment config
- `src/tracking.py`: MediaPipe and YOLO setup
- `src/game_engine.py`: campaign state machine and round progression
- `src/challenges.py`: gesture recognition and campaign definitions
- `src/rendering.py`: HUD, overlays, challenge cards, and visual effects
- `src/mqtt_client.py`: MQTT payload and publisher
- `tests/`: unit tests for gestures, transitions, and MQTT payloads

## Prerequisite

Install uv : 
 - Official doc : https://docs.astral.sh/uv/
 - Asdf Plugin : https://docs.astral.sh/uv/
 - Mise backend/plugin : https://mise.jdx.dev/mise-cookbook/python.html

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

You can also launch the modular entry point directly:

```bash
uv run --project matrix_motion python -m src.main
```

Run with MQTT enabled (recommended for NodeMCU trigger):

```bash
uv run --project matrix_motion python main.py \
  --mqtt-host broker.hivemq.com \
  --mqtt-port 1883 \
  --mqtt-topic thematrix/pill/CHANGE_TO_UNIQUE_ID \
  --mqtt-token CHANGE_ME_TO_A_LONG_RANDOM_SECRET
```

Default campaign uses 5 challenges. You can switch to 10 with:

```bash
uv run --project matrix_motion python main.py --sequence-length 10
```

If you want to force the victory MQTT color instead of using the automatic final-step color, pass `--victory-pill red` or `--victory-pill blue`.

## Controls

- `q` or `Esc`: quit
- `f`: toggle fullscreen
- Two hands open, held 1 second: start or restart the campaign

## Gestures

- `point`: useful for the red pill challenge
- `open_palm`: useful for the stop-bullets / bunny-ears challenges
- `fist`: useful for dodge / kung-fu challenges

MQTT is published only once, on final victory, with a payload such as:

Payload published to MQTT:

```json
{"token":"<token>","pill":"red|blue"}
```

Compatible with your NodeMCU firmware payload format.

Victory uses the last challenge's pill color in `auto` mode. In the default 5-step campaign it is `blue`; in the 10-step campaign it is `red`.

## Tests

```bash
uv run --project matrix_motion python -m unittest discover -s tests
```

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

# Switch to 10 challenges
uv run --project matrix_motion python main.py --sequence-length 10
```

## Raspberry Pi notes

- Start with lower resolution for better FPS:

```bash
uv run --project matrix_motion python main.py --width 960 --height 540 --imgsz 480
```

- If performance is low, increase `--yolo-stride` (ex: `3` or `4`) or use `--disable-yolo`.

## Assets

Drop optional PNG backgrounds in `assets/` using the challenge filenames shown in `src/challenges.py`. If an image is missing, the app falls back to an OpenCV-generated Matrix panel automatically.
