# Matrix Vision (Python + uv)

Real-time demo with a camera in Matrix style:
- Arcade mini-game for the BreizhCamp 2026 booth (30-45 s sessions, see below)
- Object/person detection (YOLOv8)
- Face detection (MediaPipe)
- Hand gesture recognition (MediaPipe Hands)
- Fullscreen neon green HUD + Matrix rain effect (code rain rendered **behind** the segmented player)

## Mode jeu BreizhCamp 2026

State machine :

```
ATTRACT --2 paumes 1s--> INTRO (clip video, si present) --> PILL_CHOICE
PILL_CHOICE --pilule bleue--> BLUE_ENDING (camera normale, sans effet) --2 paumes--> nouvelle partie
PILL_CHOICE --pilule rouge--> COUNTDOWN (3, 2, 1) --> IN_ROUND (4 epreuves + quiz QCM intercalaires) --> SCORE
SCORE -->= 3s apres la celebration finale--> TRINITY_OUTRO (clip 1:41->1:44 gele + texte + QR) --ESPACE / ~10s--> ATTRACT
```

Demarrage : montrer **2 paumes ouvertes pendant 1 s** face a la camera (zero calibration). Le jeu s'ouvre sur le **choix des pilules** : la bleue ramene a la realite (camera brute, aucun effet Matrix, invite a rejouer), la rouge lance les 4 epreuves. Une epreuve ratee (temps ecoule) passe simplement a la suivante avec 0 point : le parcours rouge se termine toujours sur l'ecran de score.

### Clips video

Les fichiers `.mp4` ne sont pas versionnés (`assets/*.mp4` est dans `.gitignore`) :

- `assets/intro.mp4` — clip d'intro avant le choix des pilules. Segment configurable : `--intro-start 164 --intro-end 178` (defaut 2:44→2:58). Touche **ESPACE** pour passer. Fichier absent : intro sautee.
- `assets/stop_bullet.mp4` — clip "Neo stoppe les balles" joue sur la reussite de la figure *Stop The Bullets* (segment 2:10→2:20, avec audio via `ffplay`). Fichier absent : la figure reste jouable, la celebration visuelle en pur OpenCV prend le relais.
- `assets/trinity.mp4` — clip outro de fin de partie (voir ci-dessous). Fichier absent : l'outro est sautee, le jeu revient directement a l'attract depuis l'ecran de score.

### Outro Trinity (apres l'ecran de score)

A la fin d'une partie pilule rouge, l'**ecran de score reste visible au moins 3 s** (ce delai ne demarre qu'**apres la fin de la celebration** de la derniere figure, pour qu'il soit toujours lisible meme apres une figure a longue celebration comme *Stop The Bullets*). Puis demarre la phase **`TRINITY_OUTRO`** :

- le clip `assets/trinity.mp4` joue de **1:41 → 1:44** (segment `TRINITY_VIDEO_START/END` dans `src/config.py`), puis **gele sur la derniere frame** ;
- l'audio est double : le **son natif du clip** (~3 s, effet bullet-time) **+** le **theme de fond** `assets/trinity_theme.m4a` (joue en entier via `ffplay`, coupe au retour en attract) ;
- sur la frame gelee s'affichent, via **Pillow** (texte accentue propre, hors HERSHEY) : le **message d'invitation** (centre, bas) et le **QR code de contact** `assets/QRCode pour Formulaire de contact rapide.png` (incruste a droite, sur un panneau clair + liseré vert pour rester scannable) ;
- retour a l'attract sur **ESPACE** ou automatiquement apres ~10 s de gel (`TRINITY_FREEZE_HOLD_SEC`).

Contrairement aux `.mp4`, les assets `trinity_theme.m4a` et le PNG du QR **sont versionnes**. Chacun est optionnel a l'execution : le code retombe proprement si un fichier manque.

Scoring : `points = (50 + bonus vitesse jusqu'a 50) x combo`. Le **combo** (x1 a x5) compte les figures reussies d'affilee, un echec le casse. Partie moyenne ~150-400 pts, partie parfaite et rapide ~1400 pts. Le **meilleur score du jour** est persiste dans `highscore.json` et affiche sur l'ecran d'accueil (`BEST TODAY`) ; le battre declenche `NEW RECORD !`.

Chaque figure reussie declenche un **effet de recompense** : bullet-time avec rejeu au ralenti et vignette noire (dodge), teinte rouge/bleue plein ecran (pilule), pluie blanche + lapin bondissant (white rabbit), distorsion de la realite (cuillere), clip video Neo stoppe les balles + onde de choc verte (bullet_stop).

Le passage d'une figure a la suivante se fait en deux temps, sans chevauchement : (1) la celebration joue **seule** (3 s pour les figures standard, 4.5 s de replay bullet-time pour le dodge, 10 s de clip pour Stop The Bullets), puis (2) un ecran **"PREPARE-TOI"** annonce la figure suivante (titre + consigne) pendant 2 s, chrono fige, avant que la detection ne demarre. De quoi profiter de la celebration et comprendre l'epreuve a venir.

### Quiz Matrix (QCM)

La campagne pilule rouge peut maintenant intercaler des mini-challenges **QCM Matrix** entre les figures physiques.

- Chaque quiz affiche une question + 2 a 4 reponses (zones A/B/C/D).
- La selection se fait en pointant avec l'index (landmark 8 MediaPipe) et en tenant le curseur dans une zone pendant ~1.2 s (dwell).
- Une courte phase d'intro precede la question, puis une phase de reveal affiche la bonne reponse et l'explication.
- Si le temps expire, le quiz est marque *timeout* et la partie continue.

Scoring quiz : meme formule que les figures (base + bonus vitesse, puis multiplicateur combo). Une bonne reponse conserve/augmente le combo, une mauvaise reponse (ou timeout) casse le combo.

Reglages quiz (dans `src/config.py`) : `QUIZ_ENABLED`, `QUIZ_MAX_PER_GAME`, `QUIZ_DEFAULT_DURATION_S`, `QUIZ_DWELL_SEC`, `QUIZ_INTRO_SEC`, `QUIZ_REVEAL_SEC`, `QUIZ_GRACE_SEC`, `QUIZ_ZONE_MARGIN`.

Banque de questions : `assets/quiz/questions.json` (fallback integre si fichier manquant ou invalide).

Les 5 figures physiques (seuils geometriques documentes dans `src/challenges.py`) :

| Figure | Action joueur | Detection |
|---|---|---|
| Choix des pilules (ouverture) | Attraper une des 2 pilules affichees avec la paume ouverte | Paume dans la hitbox 0.9 s — la bleue termine la partie |
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
- `tests/`: unit tests for gestures and state-machine transitions

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

Minimal run:

```bash
uv run --project matrix_motion python main.py
```

You can also launch the modular entry point directly:

```bash
uv run --project matrix_motion python -m src.main
```

A game always plays the 5 BreizhCamp figures in random order.

## Controls

- `q` or `Esc`: quit
- `f`: toggle fullscreen
- Two hands open, held 1 second: start or restart a game

## Tests

```bash
uv run --project matrix_motion python -m unittest discover -s tests
```

## Cross-platform check (Linux / Windows / macOS)

Before an event, verify the inference pipeline survives on the target machine. The
smoke test loads every model and runs one inference on a synthetic frame (no camera,
no display). **Exit code 0 = this machine can run the pipeline.** A bad GPU delegate
(e.g. Metal on macOS) crashes via a C++ `abort()` that Python cannot catch, so the
exit code is the reliable signal.

```bash
# Portable CPU path (the safe default, == running the app with --mp-cpu)
uv run --project matrix_motion python scripts/smoke_test.py

# Probe whether THIS machine's GPU delegate survives
uv run --project matrix_motion python scripts/smoke_test.py --gpu
```

The MediaPipe **selfie segmenter always runs on CPU** (its GPU/Metal path aborts).
The hand/face/pose landmarkers prefer the GPU delegate by default; if that crashes on
your machine (notably macOS Metal), run the app with `--mp-cpu` to force them onto CPU:

```bash
# macOS / any machine whose MediaPipe GPU delegate crashes
uv run --project matrix_motion python main.py --mp-cpu
```

## Useful options

```bash
# Force all MediaPipe models onto CPU (use when the GPU delegate crashes, e.g. macOS Metal)
uv run --project matrix_motion python main.py --mp-cpu

# Disable YOLO object detection (keep face + hand)
uv run --project matrix_motion python main.py --disable-yolo


# Use another camera index
uv run --project matrix_motion python main.py --camera-index 1

# Windowed mode instead of fullscreen
uv run --project matrix_motion python main.py --windowed

# Give more time per figure (default 8 s)
uv run --project matrix_motion python main.py --round-duration 10
```

## Performance (CPU/GPU)

Le projet inclut une optimisation runtime pour MediaPipe et YOLO, avec fallback propre si un backend GPU n'est pas disponible.

Fonctionnalites actuellement implementees:
- Selection explicite du device YOLO avec fallback CPU.
- Tentative delegate GPU MediaPipe puis fallback CPU.
- Frame skipping MediaPipe configurable (hand, face, pose, segmentation).
- Reduction de resolution d'inference MediaPipe configurable.
- Flou du masque de segmentation optimise (flou sur masque reduit puis re-echantillonnage).

Options CLI utiles:
- `--yolo-device` : `auto`, `cpu`, `cuda`, `cuda:0`, ...
- `--yolo-half` : active FP16 quand backend CUDA/ROCm disponible.
- `--mp-scale` : echelle d'inference MediaPipe (defaut 0.75).
- `--mp-hand-stride`, `--mp-face-stride`, `--mp-pose-stride`, `--mp-seg-stride` : inference tous les N frames.
- `--perf-mode` : `manual`, `quality`, `balanced`, `fast`.
- `--perf-target` : `auto`, `cpu`, `gpu`.
- `--benchmark-seconds` : lance un benchmark chronometre et quitte automatiquement.
- `--benchmark-runs` : nombre de runs benchmark consecutifs (moyenne finale affichee).

Par defaut, l'application utilise `--perf-mode quality --perf-target gpu`.
Si aucun backend GPU CUDA/ROCm n'est detecte, le preset bascule automatiquement vers CPU.

Exemples:

```bash
# Preset par defaut (quality + gpu, fallback CPU automatique)
uv run --project matrix_motion python main.py --camera-index 0

# Preset recommande pour machine sans GPU
uv run --project matrix_motion python main.py --camera-index 0 --perf-mode balanced --perf-target cpu

# Preset recommande pour machine avec GPU
uv run --project matrix_motion python main.py --camera-index 0 --perf-mode balanced --perf-target gpu

# Preset auto adapte a la machine
uv run --project matrix_motion python main.py --camera-index 0 --perf-mode balanced --perf-target auto

# Tuning manuel MediaPipe sans YOLO
uv run --project matrix_motion python main.py --camera-index 0 --disable-yolo --mp-scale 0.75 --mp-hand-stride 2 --mp-face-stride 2 --mp-pose-stride 2 --mp-seg-stride 3

# Benchmark CPU (8 secondes, resume FPS et cadence d'inference)
uv run --project matrix_motion python main.py --camera-index 0 --disable-yolo --windowed --benchmark-seconds 8

# Benchmark plus stable sur 3 runs
uv run --project matrix_motion python main.py --camera-index 0 --disable-yolo --windowed --benchmark-seconds 8 --benchmark-runs 3
```

## Raspberry Pi notes

- Start with lower resolution for better FPS:

```bash
uv run --project matrix_motion python main.py --width 960 --height 540 --imgsz 480
```

- If performance is low, increase `--yolo-stride` (ex: `3` or `4`) or use `--disable-yolo`.

## Assets

Drop optional PNG backgrounds in `assets/` using the challenge filenames shown in `src/challenges.py`. If an image is missing, the app falls back to an OpenCV-generated Matrix panel automatically.
