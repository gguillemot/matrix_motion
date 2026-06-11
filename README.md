# Matrix Vision (Python + uv)

Real-time demo with a camera in Matrix style:
- Arcade mini-game for the BreizhCamp 2026 booth (30-45 s sessions, see below)
- Object/person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand gesture recognition (MediaPipe Hands)
- Fullscreen neon green HUD + Matrix rain effect
- Optional MQTT trigger to your NodeMCU servo project

## Mode jeu BreizhCamp 2026

State machine : `ATTRACT` (ecran d'accueil) -> `COUNTDOWN` (3, 2, 1) -> `IN_ROUND` (5 figures, ordre aleatoire) -> `SCORE`.

Demarrage : montrer **2 paumes ouvertes pendant 1 s** face a la camera (zero calibration). Une figure ratee (temps ecoule) passe simplement a la suivante avec 0 point : la partie se termine toujours sur l'ecran de score. Scoring : **100 pts par figure + 10 pts par seconde restante**.

Les 5 figures (seuils geometriques documentes dans `src/challenges.py`) :

| Figure | Action joueur | Detection |
|---|---|---|
| The Neo Dodge | Pencher fortement buste + tete sur le cote | Offset nez / centre des epaules >= 30 % de l'envergure d'epaules |
| Red Pill / Blue Pill | Attraper une des 2 pilules affichees avec la paume ouverte | Paume dans la hitbox 0.4 s — **la pilule choisie part en MQTT** |
| Follow the White Rabbit | Oreilles de lapin (index + majeur) au-dessus de la tete | 2 mains `bunny_ears` au-dessus du nez, 0.4 s |
| There Is No Spoon | Tordre par "telekinesie" la cuillere geante au centre de l'ecran : pincer pouce-index et tourner la main | Pince pouce-index + rotation cumulee >= 45° |
| Agent Smith | Ne plus bouger du tout : les lunettes d'agent apparaissent en fondu avec l'immobilite, opacite totale = gagne | Nez immobile (< 8 % de l'envergure d'epaules) pendant 2 s |

La detection de main ouverte est independante du miroir camera et accepte paume comme revers (test geometrique du pouce par rapport a la direction de la main, voir `finger_states` dans `src/challenges.py`).

Reglages : `--round-duration 8.0` (temps par figure), `--countdown-duration 3.0`. Sur le stand, lancer avec `--disable-yolo` pour maximiser les FPS.

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

A game always plays the 5 BreizhCamp figures in random order. To force the fallback MQTT pill color (used when the pill figure was missed), pass `--victory-pill red` or `--victory-pill blue`.

## Controls

- `q` or `Esc`: quit
- `f`: toggle fullscreen
- Two hands open, held 1 second: start or restart a game

MQTT is published only once, when the score screen is reached, with the pill chosen by the player during the Red Pill / Blue Pill figure (fallback: `--victory-pill`, `blue` in `auto` mode). Payload:

Payload published to MQTT:

```json
{"token":"<token>","pill":"red|blue"}
```

Compatible with your NodeMCU firmware payload format.

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

# Give more time per figure (default 8 s)
uv run --project matrix_motion python main.py --round-duration 10
```

## Raspberry Pi notes

- Start with lower resolution for better FPS:

```bash
uv run --project matrix_motion python main.py --width 960 --height 540 --imgsz 480
```

- If performance is low, increase `--yolo-stride` (ex: `3` or `4`) or use `--disable-yolo`.

## Assets

Drop optional PNG backgrounds in `assets/` using the challenge filenames shown in `src/challenges.py`. If an image is missing, the app falls back to an OpenCV-generated Matrix panel automatically.
