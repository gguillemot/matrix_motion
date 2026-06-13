# Matrix Vision (Python + uv)

Real-time demo with a camera in Matrix style:
- Arcade mini-game for the BreizhCamp 2026 booth (30-45 s sessions, see below)
- Object/person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand gesture recognition (MediaPipe Hands)
- Fullscreen neon green HUD + Matrix rain effect (code rain rendered **behind** the segmented player)
- Optional MQTT trigger to your NodeMCU servo project

## Mode jeu BreizhCamp 2026

State machine :

```
ATTRACT --2 paumes 1s--> INTRO (clip video, si present) --> PILL_CHOICE
PILL_CHOICE --pilule bleue--> BLUE_ENDING (camera normale, sans effet) --2 paumes--> nouvelle partie
PILL_CHOICE --pilule rouge--> COUNTDOWN (3, 2, 1) --> IN_ROUND (4 epreuves, ordre aleatoire) --> SCORE
```

Demarrage : montrer **2 paumes ouvertes pendant 1 s** face a la camera (zero calibration). Le jeu s'ouvre sur le **choix des pilules** : la bleue ramene a la realite (camera brute, aucun effet Matrix, invite a rejouer), la rouge lance les 4 epreuves. La pilule choisie est publiee en MQTT **au moment du choix**. Une epreuve ratee (temps ecoule) passe simplement a la suivante avec 0 point : le parcours rouge se termine toujours sur l'ecran de score.

### Clips video

Les fichiers `.mp4` ne sont pas versionnés (`assets/*.mp4` est dans `.gitignore`) :

- `assets/intro.mp4` — clip d'intro avant le choix des pilules. Segment configurable : `--intro-start 164 --intro-end 178` (defaut 2:44→2:58). Touche **ESPACE** pour passer. Fichier absent : intro sautee.
- `assets/stop_bullet.mp4` — clip "Neo stoppe les balles" joue sur la reussite de la figure *Stop The Bullets* (segment 2:10→2:20, avec audio via `ffplay`). Fichier absent : la figure reste jouable, la celebration visuelle en pur OpenCV prend le relais.

Scoring : `points = (50 + bonus vitesse jusqu'a 50) x combo`. Le **combo** (x1 a x5) compte les figures reussies d'affilee, un echec le casse. Partie moyenne ~150-400 pts, partie parfaite et rapide ~1400 pts. Le **meilleur score du jour** est persiste dans `highscore.json` et affiche sur l'ecran d'accueil (`BEST TODAY`) ; le battre declenche `NEW RECORD !`.

Chaque figure reussie declenche un **effet de recompense** (~3.5 s, 10 s pour "Stop The Bullets") : bullet-time avec rejeu au ralenti et vignette noire (dodge), teinte rouge/bleue plein ecran (pilule), pluie blanche + lapin bondissant (white rabbit), distorsion de la realite (cuillere), clip video Neo stoppe les balles + onde de choc verte (bullet_stop).

Les 5 figures (seuils geometriques documentes dans `src/challenges.py`) :

| Figure | Action joueur | Detection |
|---|---|---|
| Choix des pilules (ouverture) | Attraper une des 2 pilules affichees avec la paume ouverte | Paume dans la hitbox 0.9 s — la pilule choisie part en MQTT, la bleue termine la partie |
| The Neo Dodge | Pencher fortement buste + tete sur le cote, tenir 0.5 s | Offset nez / centre des epaules >= 30 % de l'envergure d'epaules |
| Follow the White Rabbit | Oreilles de lapin (index + majeur) au-dessus de la tete | 2 mains `bunny_ears` au-dessus du nez, 0.9 s |
| There Is No Spoon | Tordre par "telekinesie" la cuillere geante au centre de l'ecran : pincer pouce-index et tourner la main | Pince pouce-index + rotation cumulee >= 45° |
| Stop The Bullets | Lever la paume ouverte au-dessus du milieu de l'image et tenir ~2.5 s — les balles convergent et se figent | Paume ouverte avec `palm_y < 0.60`, maintien 2.5 s (grace 1.5 s en debut de round) |

La detection de main ouverte est independante du miroir camera et accepte paume comme revers (test geometrique du pouce par rapport a la direction de la main, voir `finger_states` dans `src/challenges.py`).

Reglages : `--round-duration 8.0` (temps par figure), `--countdown-duration 3.0`. Sur le stand, lancer avec `--disable-yolo` pour maximiser les FPS.

## Fond "code rain" par segmentation

La pluie de caracteres Matrix tombe **derriere** la personne (effet signature : on est "dans" la Matrice). La personne est segmentee a chaque frame avec MediaPipe `ImageSegmenter` (modele `selfie_segmenter.tflite`, telecharge automatiquement au 1er lancement) ; le masque est seuille, ses bords adoucis au flou gaussien, puis **lisse temporellement** (blend avec la frame precedente) pour eviter le scintillement. La personne reste au premier plan avec un leger virage vert neon.

Tout est reglable dans `src/config.py` (section *Segmentation "code rain"*) :

| Constante | Role |
|---|---|
| `SEGMENTATION_ENABLED` | Active le fond par segmentation. `False` (ou modele/lib absents) -> ancienne pluie dessinee par-dessus la camera |
| `MASK_SMOOTHING` | Lissage temporel du masque (bas = plus stable, moins reactif) |
| `MASK_THRESHOLD` | Seuil de decision personne/fond avant adoucissement |
| `MASK_BLUR_KSIZE` | Rayon du flou des bords du masque (impair) |
| `MASK_INVERT` | A passer a `True` si fond et personne sont inverses a l'ecran |
| `FOREGROUND_GREEN_TINT` | Intensite du virage vert applique a la personne (0 = couleur brute) |
| `RAIN_BACKGROUND_COLOR` | Couleur du fond derriere les caracteres (BGR) |

Degradation propre : si le modele ou MediaPipe echoue, le jeu retombe automatiquement sur l'ancienne pluie en overlay (log `[SEG] disabled ... fallback`).

## Structure

- `src/main.py`: entry point that wires the app together
- `src/config.py`: CLI and environment config
- `src/tracking.py`: MediaPipe and YOLO setup, plus `PersonMaskTracker` for code-rain segmentation
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
