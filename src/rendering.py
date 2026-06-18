from __future__ import annotations

import math
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.challenges import HAND_CONNECTIONS, PILL_ZONES, QUIZ_ACTIVE, QUIZ_REVEAL
from src.config import FOREGROUND_GREEN_TINT, PROJECT_ROOT, RAIN_BACKGROUND_COLOR
from src.game_engine import GameEvent

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
MATRIX_PALE_GREEN = (190, 255, 190)
HUD_BG = (10, 25, 10)
PILL_COLORS = {"red": (60, 60, 255), "blue": (255, 140, 40)}  # BGR

# Texte de l'outro Trinity, affiche sur la frame gelee (1:44). Accents inclus :
# rendu via Pillow car cv2.putText / HERSHEY ne gere pas les caracteres accentues.
TRINITY_OUTRO_TEXT = (
    "Attends Neo, avant de te retourner, passe saluer Juliette à côté "
    "et mesure toi à elle pour savoir si tu es prêt à rejoindre nos équipes..."
)

# Polices TrueType candidates (macOS) pour le texte accentue ; fallback bitmap.
_PIL_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
_pil_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

# QR code "formulaire de contact" incruste sur la droite de la frame gelee de
# l'outro. Charge/redimensionne une seule fois (cache par taille cible).
_QR_PATH = PROJECT_ROOT / "assets" / "QRCode pour Formulaire de contact rapide.png"
_qr_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (15, 17),
    (17, 19),
    (19, 21),
    (16, 18),
    (18, 20),
    (20, 22),
]

POSE_BODY_LANDMARKS = range(11, 33)


class MatrixRain:
    def __init__(
        self, width: int, height: int, spacing: int = 12, trail_length: int = 4
    ) -> None:
        self.width = width
        self.height = height
        self.spacing = spacing
        self.trail_length = trail_length
        self.columns = max(1, width // spacing)
        self.drops = [random.randint(-height, 0) for _ in range(self.columns)]
        # Chiffres + lettres ASCII pour la base.
        self.charset = (
            string.digits + string.ascii_uppercase + "#$%&*+-=<>?@"
        )
        self.ascii_font_scale = max(0.42, min(0.56, self.spacing / 22.0))
        self._tile_size = max(12, self.spacing + 8)
        self._glyph_cache: dict[tuple[str, int, bool], np.ndarray] = {}

    def _glyph_tile(self, char: str, brightness: int, white: bool) -> np.ndarray:
        key = (char, brightness, white)
        cached = self._glyph_cache.get(key)
        if cached is not None:
            return cached

        tile = np.zeros((self._tile_size, self._tile_size, 3), dtype=np.uint8)
        color = (
            (brightness, brightness, brightness)
            if white
            else (35, brightness, 70)
        )
        cv2.putText(
            tile,
            char,
            (2, self._tile_size - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.ascii_font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        self._glyph_cache[key] = tile
        return tile

    def _render(
        self, canvas: np.ndarray, boost: bool = False, white: bool = False
    ) -> None:
        """Dessine la pluie sur `canvas` et avance les gouttes d'une frame.

        Mutualise par `draw` (overlay par-dessus la camera, mode legacy/fallback)
        et `render_background` (image de fond plein cadre, mode segmentation).
        """
        speed = 25 if boost else 14

        for col in range(self.columns):
            x = col * self.spacing
            y = self.drops[col]
            for trail_idx in range(self.trail_length):
                trail_y = y - trail_idx * self.spacing
                if trail_y < 0:
                    continue

                char = random.choice(self.charset)
                brightness = max(95, 255 - trail_idx * 35)
                tile = self._glyph_tile(char, brightness, white)
                y0 = trail_y - (self._tile_size - 3)
                if y0 >= self.height or x >= self.width:
                    continue
                x1 = max(0, x)
                y1 = max(0, y0)
                x2 = min(self.width, x + self._tile_size)
                y2 = min(self.height, y0 + self._tile_size)
                if x2 <= x1 or y2 <= y1:
                    continue

                tx1 = x1 - x
                ty1 = y1 - y0
                tx2 = tx1 + (x2 - x1)
                ty2 = ty1 + (y2 - y1)
                src = tile[ty1:ty2, tx1:tx2]
                dst = canvas[y1:y2, x1:x2]
                # Lighten only where glyph pixels are present.
                np.maximum(dst, src, out=dst)

            self.drops[col] += speed
            if self.drops[col] > self.height + random.randint(0, 150):
                self.drops[col] = random.randint(-140, 0)

    def draw(self, frame: np.ndarray, boost: bool = False, white: bool = False) -> None:
        """Mode legacy/fallback : pluie semi-transparente par-dessus la camera."""
        overlay = np.zeros_like(frame)
        self._render(overlay, boost=boost, white=white)
        cv2.addWeighted(overlay, 0.35, frame, 1.0, 0, frame)

    def render_background(self, boost: bool = False, white: bool = False) -> np.ndarray:
        """Mode segmentation : image plein cadre de la pluie sur fond noir teinte.

        Sert de calque de fond derriere la personne segmentee : les caracteres
        sont a pleine luminosite (vrai look Matrix), pas attenues comme en
        overlay.
        """
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = RAIN_BACKGROUND_COLOR
        self._render(canvas, boost=boost, white=white)
        return canvas


def compose_matrix_scene(
    frame: np.ndarray,
    mask: np.ndarray | None,
    rain: MatrixRain,
    boost: bool = False,
    white: bool = False,
) -> np.ndarray:
    """Compose la scene signature : personne au premier plan, pluie en fond.

    `mask` est un masque de presence personne lisse (float HxW, 0..1). Si
    `mask` est None (segmentation indisponible), on retombe proprement sur
    l'ancien rendu (camera + pluie par-dessus).
    """
    rain_bg = rain.render_background(boost=boost, white=white)

    if mask is None:
        display = frame.copy()
        rain.draw(display, boost=boost, white=white)
        return display

    # La personne reste en couleur, avec un leger virage vert pour l'integrer
    # dans la Matrice (on pousse le canal vert, on attenue rouge/bleu).
    foreground = frame.astype(np.float32)
    if FOREGROUND_GREEN_TINT > 0:
        tint = FOREGROUND_GREEN_TINT
        foreground[:, :, 0] *= 1.0 - 0.5 * tint  # bleu
        foreground[:, :, 1] = np.minimum(
            255.0, foreground[:, :, 1] * (1.0 + tint)
        )  # vert
        foreground[:, :, 2] *= 1.0 - 0.4 * tint  # rouge

    m = mask[:, :, None]
    composed = foreground * m + rain_bg.astype(np.float32) * (1.0 - m)
    return composed.astype(np.uint8)


def add_scanlines(frame: np.ndarray, step: int = 4) -> None:
    frame[::step, :] = (frame[::step, :] * 0.55).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers texte / overlay
# ---------------------------------------------------------------------------


def _put_centered(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    glow: bool = False,
) -> None:
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (frame.shape[1] - text_w) // 2)
    if glow:
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            MATRIX_DARK_GREEN,
            thickness + 4,
            cv2.LINE_AA,
        )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _darken(frame: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(np.zeros_like(frame), alpha, frame, 1.0 - alpha, 0, frame)


def _pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Charge (et met en cache) une police TrueType pour le texte accentue.

    Retombe sur la police bitmap par defaut de Pillow si aucune TTF n'est
    disponible (le rendu reste correct, juste plus petit).
    """
    cached = _pil_font_cache.get(size)
    if cached is not None:
        return cached
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for candidate in _PIL_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                font = ImageFont.truetype(candidate, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _pil_font_cache[size] = font
    return font


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Coupe `text` en lignes ne depassant pas `max_width` pixels (word-wrap)."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _put_paragraph_pil(
    frame: np.ndarray,
    text: str,
    center_y: int,
    font_size: int = 30,
    color: tuple[int, int, int] = MATRIX_GREEN,
    width_ratio: float = 0.82,
) -> None:
    """Dessine un paragraphe accentue centre via Pillow (gere é, à, ô, ê...).

    `color` est en BGR (convention OpenCV) ; on convertit en RGB pour PIL.
    `frame` est modifie en place. Un contour sombre ameliore la lisibilite.
    """
    h, w = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _pil_font(font_size)

    max_width = int(w * width_ratio)
    lines = _wrap_text(draw, text, font, max_width)
    line_height = int(font_size * 1.35)
    total_height = line_height * len(lines)
    y = center_y - total_height // 2

    rgb = (color[2], color[1], color[0])
    outline = (10, 40, 18)
    for line in lines:
        line_w = draw.textlength(line, font=font)
        x = (w - line_w) / 2
        draw.text(
            (x, y), line, font=font, fill=rgb, stroke_width=3, stroke_fill=outline
        )
        y += line_height

    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_trinity_outro_text(frame: np.ndarray) -> None:
    """Texte de l'outro Trinity sur la frame gelee (1:44)."""
    _darken(frame, 0.35)
    _put_paragraph_pil(frame, TRINITY_OUTRO_TEXT, center_y=int(frame.shape[0] * 0.78))


def _load_qr(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Charge le QR redimensionne en `size`x`size` -> (BGR, alpha 0..1). Cache."""
    cached = _qr_cache.get(size)
    if cached is not None:
        return cached
    if not _QR_PATH.exists():
        return None
    raw = cv2.imread(str(_QR_PATH), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    if raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr = raw[:, :, :3]
        alpha = np.ones((size, size), dtype=np.float32)
    result = (bgr, alpha)
    _qr_cache[size] = result
    return result


def draw_outro_qr(frame: np.ndarray) -> None:
    """Incruste le QR code "contact" sur la droite de la frame gelee de l'outro.

    Pose un panneau clair semi-opaque (zone calme + contraste pour rester
    scannable sur la video sombre), un liseré vert Matrix, puis composite le QR
    via son canal alpha. Place en haut/milieu a droite pour ne pas chevaucher le
    texte (bande basse ~0.78).
    """
    h, w = frame.shape[:2]
    qr_size = int(h * 0.34)
    loaded = _load_qr(qr_size)
    if loaded is None:
        return
    qr_bgr, qr_alpha = loaded

    pad = max(10, qr_size // 12)
    margin = 40
    x0 = w - qr_size - pad - margin
    y0 = int(h * 0.16)
    x1, y1 = x0 + qr_size + 2 * pad, y0 + qr_size + 2 * pad
    if x0 < 0 or y1 > h:
        return

    # Panneau clair semi-opaque (fond + zone calme), liseré vert.
    panel = frame[y0:y1, x0:x1]
    white = np.full_like(panel, 235)
    cv2.addWeighted(white, 0.85, panel, 0.15, 0, panel)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), MATRIX_GREEN, 2, cv2.LINE_AA)

    # Composite alpha du QR par-dessus le panneau.
    qx, qy = x0 + pad, y0 + pad
    roi = frame[qy : qy + qr_size, qx : qx + qr_size].astype(np.float32)
    a = qr_alpha[:, :, None]
    blended = qr_bgr.astype(np.float32) * a + roi * (1.0 - a)
    frame[qy : qy + qr_size, qx : qx + qr_size] = blended.astype(np.uint8)


def _blink(period: float = 0.8) -> bool:
    return int(time.monotonic() / period) % 2 == 0


def fit_to_window(frame: np.ndarray, window_name: str) -> np.ndarray:
    """Adapte `frame` a la taille reelle de la fenetre en PRESERVANT le ratio
    (letterbox / pillarbox avec bandes noires).

    En plein ecran, OpenCV etire sinon l'image 16:9 pour remplir un ecran de
    ratio different : la question et les cartes du quiz se retrouvent deformees.
    Ici on redimensionne au plus grand format qui tient dans la fenetre sans
    deformation, puis on centre sur un canvas noir a la taille de la fenetre.

    Le pointage n'est pas affecte : il travaille en coordonnees normalisees de
    la camera, independantes de l'affichage."""
    if not hasattr(cv2, "getWindowImageRect"):
        return frame
    try:
        rect = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame
    win_w, win_h = int(rect[2]), int(rect[3])
    if win_w <= 0 or win_h <= 0:
        return frame  # fenetre minimisee / taille indisponible

    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    # Deja a la bonne taille : rien a faire (cas fenetre = image native).
    if (win_w, win_h) == (w, h):
        return frame

    scale = min(win_w / w, win_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((win_h, win_w, 3), dtype=frame.dtype)
    x0 = (win_w - new_w) // 2
    y0 = (win_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: tuple[int, int, int] = MATRIX_GREEN,
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), MATRIX_DARK_GREEN, 1)
    fill = int(width * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)


# ---------------------------------------------------------------------------
# Detections (debug / feedback visuel)
# ---------------------------------------------------------------------------


def draw_yolo_detections(
    frame: np.ndarray, cached_boxes: list, model_names, draw: bool = True
) -> int:
    persons = 0

    for box in cached_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = (
            model_names.get(cls_id, str(cls_id))
            if hasattr(model_names, "get")
            else str(cls_id)
        )

        if label == "person":
            persons += 1

        if not draw:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), MATRIX_DARK_GREEN, 1)
        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            MATRIX_GREEN,
            2,
            cv2.LINE_AA,
        )

    return persons


def draw_face_detections(frame: np.ndarray, detections) -> int:
    faces = 0

    if detections:
        for det in detections:
            faces += 1
            bb = det.bounding_box
            x, y, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (80, 255, 140), 2)

    return faces


def draw_pose_skeleton(
    frame: np.ndarray, pose_results, frame_width: int, frame_height: int
) -> int:
    poses = 0

    if pose_results.pose_landmarks:
        for pose_landmarks in pose_results.pose_landmarks:
            poses += 1

            for start, end in POSE_CONNECTIONS:
                x1 = int(pose_landmarks[start].x * frame_width)
                y1 = int(pose_landmarks[start].y * frame_height)
                x2 = int(pose_landmarks[end].x * frame_width)
                y2 = int(pose_landmarks[end].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)

            for index in POSE_BODY_LANDMARKS:
                landmark = pose_landmarks[index]
                cx = int(landmark.x * frame_width)
                cy = int(landmark.y * frame_height)
                cv2.circle(frame, (cx, cy), 3, (120, 255, 150), -1)

    return poses


def draw_hand_detections(
    frame: np.ndarray, hand_results, frame_width: int, frame_height: int
) -> int:
    hands = 0

    if hand_results.hand_landmarks:
        for lm_list in hand_results.hand_landmarks:
            hands += 1
            for start, end in HAND_CONNECTIONS:
                x1 = int(lm_list[start].x * frame_width)
                y1 = int(lm_list[start].y * frame_height)
                x2 = int(lm_list[end].x * frame_width)
                y2 = int(lm_list[end].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), MATRIX_DARK_GREEN, 2)

            for lm in lm_list:
                cx = int(lm.x * frame_width)
                cy = int(lm.y * frame_height)
                cv2.circle(frame, (cx, cy), 4, MATRIX_GREEN, -1)

    return hands


# ---------------------------------------------------------------------------
# Ecrans du jeu
# ---------------------------------------------------------------------------


def draw_attract_screen(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.45)

    _put_centered(
        frame, "ENTER THE MATRIX", height // 2 - 70, 2.2, MATRIX_GREEN, 4, glow=True
    )
    if _blink(1.4):
        _put_centered(
            frame,
            "WAKE UP, NEO...",
            height // 2,
            1.0,
            MATRIX_PALE_GREEN,
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )
    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(
            frame,
            "MONTRE TES 2 PAUMES POUR COMMENCER",
            height // 2 + 80,
            0.95,
            MATRIX_GREEN,
            2,
        )

    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(
            frame,
            (frame.shape[1] - bar_width) // 2,
            height // 2 + 110,
            bar_width,
            14,
            event.start_hold_progress,
        )

    # L'accroche competitive du stand
    if event.best_score > 0:
        _put_centered(
            frame,
            f"BEST TODAY : {event.best_score}",
            height // 2 + 170,
            1.0,
            MATRIX_PALE_GREEN,
            2,
        )


def draw_countdown(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.35)
    value = max(1, event.countdown_value)
    # Pulsation : le chiffre grossit a mesure que la seconde s'ecoule.
    fraction = time.monotonic() % 1.0
    scale = 5.0 + 1.5 * fraction
    _put_centered(frame, "GET READY", height // 2 - 130, 1.1, MATRIX_PALE_GREEN, 2)
    _put_centered(
        frame, str(value), height // 2 + 70, scale, MATRIX_GREEN, 10, glow=True
    )


def draw_ready_card(frame: np.ndarray, event: GameEvent) -> None:
    """Ecran "prepare-toi" entre la celebration et l'activation de la figure :
    annonce la figure suivante (titre + consigne) chrono fige, pour laisser le
    temps de comprendre l'epreuve avant qu'elle ne demarre."""
    height, _ = frame.shape[:2]
    _darken(frame, 0.45)

    _put_centered(frame, "PREPARE-TOI", height // 2 - 150, 1.1, MATRIX_PALE_GREEN, 2)
    _put_centered(
        frame,
        event.challenge_title.upper(),
        height // 2 - 60,
        1.7,
        MATRIX_GREEN,
        3,
        glow=True,
    )
    _put_centered(
        frame,
        event.challenge_prompt,
        height // 2 + 20,
        0.9,
        MATRIX_PALE_GREEN,
        2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    )
    countdown = max(1, math.ceil(event.round_starts_in))
    _put_centered(
        frame, str(countdown), height // 2 + 140, 2.5, MATRIX_GREEN, 6, glow=True
    )


def draw_round_overlay(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    # Bandeau figure (sous le HUD)
    round_label = f"ROUND {event.round_index}/{event.round_total}"
    cv2.putText(
        frame,
        round_label,
        (24, 116),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        event.challenge_title.upper(),
        (24, 158),
        cv2.FONT_HERSHEY_DUPLEX,
        1.15,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        event.challenge_prompt,
        (24, 196),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        MATRIX_PALE_GREEN,
        2,
        cv2.LINE_AA,
    )

    # Progression de la figure (maintien pilule/bunny, cuillere, scan)
    if event.figure_progress > 0:
        _draw_progress_bar(frame, 24, 212, 280, 12, event.figure_progress)

    # Timer : barre + texte. Verte, puis rouge sous 2 s.
    ratio = event.timer_left / event.round_duration if event.round_duration > 0 else 0.0
    timer_color = MATRIX_GREEN if event.timer_left > 2.0 else (60, 60, 255)
    bar_margin = 24
    bar_y = height - 46
    _draw_progress_bar(
        frame, bar_margin, bar_y, width - 2 * bar_margin, 16, ratio, timer_color
    )
    cv2.putText(
        frame,
        f"{event.timer_left:0.1f}s",
        (bar_margin, bar_y - 12),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        timer_color,
        2,
        cv2.LINE_AA,
    )


def draw_pills(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    for name, (zone_x, zone_y) in PILL_ZONES.items():
        cx, cy = int(zone_x * width), int(zone_y * height)
        color = PILL_COLORS[name]
        hovered = event.pill_hover == name

        # Halo
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 70 if hovered else 58, color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Capsule + reflet
        axes = (52, 26)
        cv2.ellipse(frame, (cx, cy), axes, 25, 0, 360, color, -1)
        cv2.ellipse(frame, (cx, cy), axes, 25, 0, 360, (255, 255, 255), 2)
        cv2.ellipse(frame, (cx - 12, cy - 10), (16, 6), 25, 0, 360, (255, 255, 255), -1)

        if hovered:
            cv2.circle(frame, (cx, cy), 80, (255, 255, 255), 2)

        label = name.upper()
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(
            frame,
            label,
            (cx - text_w // 2, cy + 78),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_pill_choice(frame: np.ndarray, event: GameEvent) -> None:
    """Phase d'ouverture : le joueur choisit sa pilule avec la paume ouverte."""
    height, width = frame.shape[:2]

    _put_centered(frame, "CHOISIS TA PILULE", 140, 1.4, MATRIX_GREEN, 3, glow=True)
    draw_pills(frame, event)

    legends = {"red": "ENTRE DANS LA MATRICE", "blue": "RETOUR A LA REALITE"}
    for name, (zone_x, zone_y) in PILL_ZONES.items():
        label = legends[name]
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cx, cy = int(zone_x * width), int(zone_y * height)
        cv2.putText(
            frame,
            label,
            (cx - text_w // 2, cy + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            MATRIX_PALE_GREEN,
            2,
            cv2.LINE_AA,
        )

    if event.figure_progress > 0:
        bar_width = 320
        _draw_progress_bar(
            frame,
            (width - bar_width) // 2,
            height - 70,
            bar_width,
            14,
            event.figure_progress,
        )
    elif _blink(0.8):
        _put_centered(
            frame,
            "PAUME OUVERTE SUR UNE PILULE",
            height - 64,
            0.85,
            MATRIX_PALE_GREEN,
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )


def draw_badge_scan_screen(
    frame: np.ndarray,
    event: GameEvent,
    qr_result: str | None,
    result_at: float,
) -> None:
    """Écran intermédiaire (30 s) entre le score et l'outro Trinity.

    Affiche la caméra live (scène Matrix déjà composée) avec une invitation
    bien visible à scanner son badge QR, un timer et le résultat décodé.
    """
    h, w = frame.shape[:2]
    now = time.monotonic()

    # --- Bandeau titre haut (sous le HUD) ----------------------------------
    by = 80
    cv2.rectangle(frame, (0, by), (w, by + 52), (5, 18, 5), -1)
    cv2.rectangle(frame, (0, by + 52), (w, by + 54), MATRIX_DARK_GREEN, -1)
    _put_centered(
        frame, "SCANNEZ VOTRE BADGE", by + 36, 1.3, MATRIX_GREEN, 3, glow=True
    )

    # --- Zone de scan centrale ---------------------------------------------
    zone_w, zone_h = 360, 270
    zx = (w - zone_w) // 2
    zy = by + 70
    zx2, zy2 = zx + zone_w, zy + zone_h

    cv2.rectangle(frame, (zx, zy), (zx2 - 1, zy2 - 1), MATRIX_GREEN, 2, cv2.LINE_AA)

    has_result = (
        qr_result is not None
        and result_at > 0
        and (now - result_at) < _QR_DISPLAY_SEC
    )

    if has_result:
        cv2.putText(
            frame, "BADGE LU !",
            (zx + 20, zy + 52),
            cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 255, 80), 2, cv2.LINE_AA,
        )
        assert qr_result is not None
        line_len = 30
        text = qr_result if len(qr_result) <= 120 else qr_result[:117] + "..."
        lines = [text[i : i + line_len] for i in range(0, len(text), line_len)][:5]
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (zx + 20, zy + 96 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, MATRIX_PALE_GREEN, 1, cv2.LINE_AA,
            )
    else:
        # Quatre coins clignotants style viseur
        cl = 20
        pad = 24
        sx, sy = zx + pad, zy + pad
        ex, ey = zx2 - pad, zy2 - pad
        col = MATRIX_GREEN if _blink(0.5) else MATRIX_DARK_GREEN
        for (ax, ay), (bx, by_) in [
            ((sx, sy), (sx + cl, sy)), ((sx, sy), (sx, sy + cl)),
            ((ex, sy), (ex - cl, sy)), ((ex, sy), (ex, sy + cl)),
            ((sx, ey), (sx + cl, ey)), ((sx, ey), (sx, ey - cl)),
            ((ex, ey), (ex - cl, ey)), ((ex, ey), (ex, ey - cl)),
        ]:
            cv2.line(frame, (ax, ay), (bx, by_), col, 3)
        # Ligne de scan animée
        scan_y = int(sy + (ey - sy) * ((now % 1.8) / 1.8))
        cv2.line(frame, (sx + 6, scan_y), (ex - 6, scan_y), MATRIX_GREEN, 2, cv2.LINE_AA)
        # Invite
        _put_centered(
            frame, "Pointez votre badge vers la camera",
            zy + zone_h // 2 + 20, 0.72, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX,
        )

    # --- Timer en bas de la zone -------------------------------------------
    time_left = event.badge_scan_time_left
    ratio = time_left / 30.0
    bar_x, bar_w = zx, zone_w
    bar_y = zy2 + 10
    _draw_progress_bar(frame, bar_x, bar_y, bar_w, 10, ratio, MATRIX_GREEN)
    cv2.putText(
        frame,
        f"{int(time_left) + 1}s",
        (zx + bar_w + 6, bar_y + 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, MATRIX_GREEN, 1, cv2.LINE_AA,
    )

    # --- Invite rejouer (bas d'écran) --------------------------------------
    if _blink(0.9) or event.start_hold_progress > 0:
        _put_centered(
            frame, "2 PAUMES OUVERTES POUR REJOUER",
            h - 70, 0.9, MATRIX_GREEN, 2,
        )
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(
            frame,
            (w - bar_width) // 2,
            h - 50,
            bar_width,
            14,
            event.start_hold_progress,
        )


def draw_blue_ending(frame: np.ndarray, event: GameEvent) -> None:
    """Pilule bleue : camera normale, aucun effet Matrix. Texte discret et
    invite a rejouer (gris neutres, surtout pas de vert Matrix)."""
    height, width = frame.shape[:2]

    _put_centered(frame, "RETOUR A LA REALITE", 90, 1.1, (235, 235, 235), 2)
    _put_centered(
        frame,
        "L'HISTOIRE S'ARRETE ICI...",
        135,
        0.75,
        (180, 180, 180),
        2,
        cv2.FONT_HERSHEY_SIMPLEX,
    )

    if _blink(1.0) or event.start_hold_progress > 0:
        _put_centered(
            frame,
            "2 PAUMES OUVERTES POUR RETOURNER DANS LA MATRICE",
            height - 64,
            0.8,
            (210, 210, 210),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(
            frame,
            (width - bar_width) // 2,
            height - 44,
            bar_width,
            12,
            event.start_hold_progress,
            (220, 220, 220),
        )


def draw_intro_hint(frame: np.ndarray) -> None:
    if _blink(1.2):
        cv2.putText(
            frame,
            "ESPACE POUR PASSER",
            (24, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def draw_spoon(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    # Cuillere geante fixe au centre : le joueur la tord "par telekinesie" en
    # pincant et tournant sa main, comme l'enfant de l'Oracle. Toujours
    # visible pendant la figure, lisible de loin sur le stand.
    length = int(height * 0.42)
    base_x = width // 2
    base_y = int(height * 0.80)  # bas du manche, fixe
    bend = event.figure_progress
    steps = 16

    points: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        # Le bas du manche reste droit, la moitie haute se courbe de plus en
        # plus (offset quadratique, max ~35 % de la longueur).
        curve = max(0.0, t - 0.45) / 0.55
        offset = bend * 0.35 * length * curve * curve
        points.append((int(base_x + offset), int(base_y - length * t)))

    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], False, (60, 60, 70), 22, cv2.LINE_AA)  # contour sombre
    cv2.polylines(frame, [pts], False, (210, 210, 220), 14, cv2.LINE_AA)  # corps metal
    cv2.polylines(frame, [pts], False, (255, 255, 255), 4, cv2.LINE_AA)  # reflet

    # Cuilleron au sommet, oriente selon la torsion du bout du manche
    tip_x, tip_y = points[-1]
    prev_x, prev_y = points[-2]
    tip_angle = math.degrees(math.atan2(tip_y - prev_y, tip_x - prev_x)) + 90
    bowl_axes = (int(height * 0.06), int(height * 0.085))
    cv2.ellipse(frame, (tip_x, tip_y), bowl_axes, tip_angle, 0, 360, (60, 60, 70), -1)
    cv2.ellipse(
        frame,
        (tip_x, tip_y),
        (bowl_axes[0] - 5, bowl_axes[1] - 5),
        tip_angle,
        0,
        360,
        (225, 225, 235),
        -1,
    )
    cv2.ellipse(
        frame,
        (tip_x - bowl_axes[0] // 3, tip_y - bowl_axes[1] // 3),
        (bowl_axes[0] // 3, bowl_axes[1] // 4),
        tip_angle,
        0,
        360,
        (255, 255, 255),
        -1,
    )

    if event.spoon_anchor is None:
        if _blink(0.8):
            _put_centered(
                frame,
                "PINCE POUCE + INDEX ET TOURNE LA MAIN",
                height - 90,
                0.85,
                MATRIX_PALE_GREEN,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )
        return

    # Rayon de telekinesie : main pincee -> cuillere
    hand_x = int(event.spoon_anchor[0] * width)
    hand_y = int(event.spoon_anchor[1] * height)
    mid_y = base_y - length // 2
    cv2.line(frame, (hand_x, hand_y), (base_x, mid_y), MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.circle(frame, (hand_x, hand_y), 12, MATRIX_GREEN, 2)
    cv2.putText(
        frame,
        f"{int(bend * 100)}%",
        (base_x + int(height * 0.09), mid_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )


def draw_bullet_stop(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]
    progress = event.figure_progress

    if event.palm_anchor is None:
        if _blink(0.7):
            _put_centered(
                frame,
                "RAISE YOUR HAND",
                height - 90,
                0.95,
                MATRIX_PALE_GREEN,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )
        return

    palm_x = int(event.palm_anchor[0] * width)
    palm_y = int(event.palm_anchor[1] * height)

    # Trajectoires deterministes : N balles depuis les bords vers la paume.
    # t = progression d'avance vers la paume (0 = bord, 1 = sur la paume).
    N = 14
    radius_out = int(math.hypot(width, height) * 0.65)
    t = min(1.0, progress * 1.15)

    overlay = frame.copy()
    for i in range(N):
        theta = 2 * math.pi * i / N
        src_x = int(width / 2 + radius_out * math.cos(theta))
        src_y = int(height / 2 + radius_out * math.sin(theta))

        bx = int(src_x + (palm_x - src_x) * t)
        by = int(src_y + (palm_y - src_y) * t)

        # Trainee : longueur diminue quand t augmente (effet de freinage)
        trail_len = max(1.0 - t, 0.05)
        tail_x = int(src_x + (palm_x - src_x) * max(0.0, t - trail_len * 0.35))
        tail_y = int(src_y + (palm_y - src_y) * max(0.0, t - trail_len * 0.35))
        cv2.line(overlay, (tail_x, tail_y), (bx, by), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.circle(overlay, (bx, by), 4, (235, 235, 235), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Anneau de "champ de force" autour de la paume quand les balles se figent
    if progress > 0.70:
        alpha_ring = (progress - 0.70) / 0.30
        ring_r = int(40 + 20 * alpha_ring)
        cv2.circle(frame, (palm_x, palm_y), ring_r, MATRIX_GREEN, 2, cv2.LINE_AA)
        cv2.circle(
            frame, (palm_x, palm_y), ring_r + 8, MATRIX_DARK_GREEN, 1, cv2.LINE_AA
        )

    # Jauge de progression en bas
    _draw_progress_bar(
        frame, (width - 240) // 2, height - 30, 240, 10, progress, MATRIX_GREEN
    )


# ---------------------------------------------------------------------------
# Effets de recompense (celebration de ~2.4 s apres chaque figure reussie)
# ---------------------------------------------------------------------------

_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}
_WAVE_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _vignette_mask(height: int, width: int) -> np.ndarray:
    mask = _VIGNETTE_CACHE.get((height, width))
    if mask is None:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        dist = np.sqrt(xs * xs + ys * ys)
        mask = np.clip(1.15 - dist, 0.0, 1.0)[..., None]
        _VIGNETTE_CACHE[(height, width)] = mask
    return mask


def apply_vignette(frame: np.ndarray, strength: float) -> None:
    """Assombrit les bords (strength 0 = rien, 1 = bords noirs)."""
    if strength <= 0.0:
        return
    mask = _vignette_mask(*frame.shape[:2])
    scale = 1.0 - strength * (1.0 - mask)
    frame[:] = (frame * scale).astype(np.uint8)


# ---------------------------------------------------------------------------
# Quiz Matrix : ecran de question (cartes + pointage + reveal)
# ---------------------------------------------------------------------------

# Couleurs RGB (Pillow) du texte quiz ; le rendu cv2 utilise les constantes BGR.
_QUIZ_RGB_GREEN = (90, 255, 130)
_QUIZ_RGB_PALE = (190, 255, 190)
_QUIZ_RGB_RED = (255, 80, 80)
_QUIZ_TEXT_OUTLINE = (8, 32, 14)
_QUIZ_WRONG_BGR = (60, 60, 255)
_QUIZ_LETTERS = "ABCD"


def _draw_quiz_text(
    frame: np.ndarray, event: GameEvent, reveal: bool, card_top: float
) -> None:
    """Question + libelles des reponses + explication/indice, en une seule passe
    Pillow (gere les accents et limite le cout par frame).

    `card_top` (normalise 0..1) = haut des cartes : la question est centree
    verticalement dans l'espace AU-DESSUS, pour rester visible quel que soit le
    nombre de reponses, la resolution ou le ratio d'ecran."""
    h, w = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # Region disponible pour la question : du haut de l'ecran jusqu'au-dessus des
    # cartes, en reservant une bande (info + barre de temps) juste au-dessus
    # des cartes.
    region_top = int(h * 0.045)
    reserved = int(h * 0.11)  # bande info + timer au-dessus des cartes
    region_bottom = max(region_top + int(h * 0.10), int(card_top * h) - reserved)

    # Question sur un panneau sombre translucide borde de vert (lisible par-dessus
    # la camera + la pluie de code). Police bornee pour tenir meme en 720p.
    q_font = _pil_font(min(int(h * 0.070), int((region_bottom - region_top) * 0.42)))
    q_size = int(getattr(q_font, "size", int(h * 0.046)))
    q_lh = int(q_size * 1.28)
    q_lines = _wrap_text(draw, event.quiz_question, q_font, int(w * 0.80))
    text_h = q_lh * len(q_lines)
    pad_x, pad_y = int(w * 0.030), int(h * 0.022)
    panel_h = text_h + 2 * pad_y
    # Centre le panneau verticalement dans la region (descend la question vers le
    # milieu quand il y a peu de reponses).
    panel_top = region_top + max(0, ((region_bottom - region_top) - panel_h) // 2)
    panel_bottom = panel_top + panel_h
    max_lw = max((draw.textlength(line, font=q_font) for line in q_lines), default=0)
    panel_w = min(w - 2 * int(w * 0.04), int(max_lw) + 2 * pad_x)
    panel_x0 = (w - panel_w) // 2
    panel_x1 = panel_x0 + panel_w

    # Panneau translucide (overlay RGBA composite pour un fond presque opaque).
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [panel_x0, panel_top, panel_x1, panel_bottom],
        radius=20,
        fill=(6, 20, 9, 215),
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [panel_x0, panel_top, panel_x1, panel_bottom],
        radius=20,
        outline=_QUIZ_RGB_GREEN,
        width=3,
    )

    y = panel_top + pad_y
    for line in q_lines:
        lw = draw.textlength(line, font=q_font)
        draw.text(
            ((w - lw) / 2, y),
            line,
            font=q_font,
            fill=_QUIZ_RGB_GREEN,
            stroke_width=2,
            stroke_fill=_QUIZ_TEXT_OUTLINE,
        )
        y += q_lh

    # Libelles des reponses, centres dans chaque carte (apres la lettre A/B/C/D).
    a_font = _pil_font(int(h * 0.040))
    a_lh = int(getattr(a_font, "size", int(h * 0.040)) * 1.2)
    for i, zone in enumerate(event.quiz_zones):
        if i >= len(event.quiz_answers):
            break
        x0, y0 = int(zone[0] * w), int(zone[1] * h)
        x1, y1 = int(zone[2] * w), int(zone[3] * h)
        color = _QUIZ_RGB_PALE
        if reveal:
            if i == event.quiz_correct_index:
                color = _QUIZ_RGB_GREEN
            elif i == event.quiz_selected_index:
                color = _QUIZ_RGB_RED
        gutter = int(w * 0.042)
        text_x0 = x0 + gutter
        lines = _wrap_text(
            draw, event.quiz_answers[i], a_font, max(40, (x1 - x0) - gutter - int(w * 0.013))
        )
        total = a_lh * len(lines)
        ty = y0 + ((y1 - y0) - total) // 2
        for line in lines:
            draw.text(
                (text_x0, ty),
                line,
                font=a_font,
                fill=color,
                stroke_width=2,
                stroke_fill=_QUIZ_TEXT_OUTLINE,
            )
            ty += a_lh

    # Bandeau d'info, juste au-dessus des cartes : explication au reveal, sinon
    # indice si aucune main n'est detectee.
    info_text = ""
    info_color = _QUIZ_RGB_PALE
    if reveal and event.quiz_explanation:
        info_text = event.quiz_explanation
    elif event.quiz_substate == QUIZ_ACTIVE and not event.quiz_hand_detected:
        info_text = "Leve la main pour pointer"
    if info_text:
        i_font = _pil_font(int(h * 0.032))
        i_lh = int(getattr(i_font, "size", int(h * 0.032)) * 1.2)
        i_lines = _wrap_text(draw, info_text, i_font, int(w * 0.8))
        # Ancre le bas du bloc juste au-dessus de la barre de temps / des cartes.
        info_anchor = int(card_top * h) - int(h * 0.055)
        iy = info_anchor - i_lh * len(i_lines)
        for line in i_lines:
            lw = draw.textlength(line, font=i_font)
            draw.text(
                ((w - lw) / 2, iy),
                line,
                font=i_font,
                fill=info_color,
                stroke_width=2,
                stroke_fill=_QUIZ_TEXT_OUTLINE,
            )
            iy += i_lh

    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_quiz(frame: np.ndarray, event: GameEvent) -> None:
    """Ecran de quiz : question (centree dans le haut), cartes de reponses
    compactes en bas, curseur main + jauge de dwell, barre de temps, reveal."""
    h, w = frame.shape[:2]
    _darken(frame, 0.5)
    reveal = event.quiz_substate == QUIZ_REVEAL
    active = event.quiz_substate == QUIZ_ACTIVE
    card_top = min((z[1] for z in event.quiz_zones), default=0.6)

    # Barre de temps, juste au-dessus des cartes : verte, puis orange/rouge < 3 s.
    if not reveal:
        total = event.quiz_timer_total if event.quiz_timer_total > 0 else 1.0
        ratio = max(0.0, min(1.0, event.quiz_timer_left / total))
        timer_color = MATRIX_GREEN if event.quiz_timer_left > 3.0 else (40, 140, 255)
        margin = int(w * 0.07)
        bar_y = int(card_top * h) - int(h * 0.04)
        _draw_progress_bar(frame, margin, bar_y, w - 2 * margin, 12, ratio, timer_color)
        cv2.putText(
            frame,
            f"{event.quiz_timer_left:0.1f}s",
            (margin, bar_y - int(h * 0.014)),
            cv2.FONT_HERSHEY_DUPLEX,
            max(0.8, h / 720.0),
            timer_color,
            max(2, int(h / 480)),
            cv2.LINE_AA,
        )

    # Cartes de reponses (fond + bordure + jauge de dwell).
    for i, zone in enumerate(event.quiz_zones):
        x0, y0 = int(zone[0] * w), int(zone[1] * h)
        x1, y1 = int(zone[2] * w), int(zone[3] * h)
        dwell = event.quiz_dwell[i] if i < len(event.quiz_dwell) else 0.0

        border = MATRIX_DARK_GREEN
        thickness = 3
        if reveal:
            if i == event.quiz_correct_index:
                border = MATRIX_GREEN
                pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)
                thickness = 3 + int(4 * pulse)
            elif i == event.quiz_selected_index:
                border = _QUIZ_WRONG_BGR
        elif active and dwell > 0:
            border = MATRIX_GREEN

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 28, 12), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, thickness, cv2.LINE_AA)

        if i < len(_QUIZ_LETTERS):
            cv2.putText(
                frame,
                _QUIZ_LETTERS[i],
                (x0 + int(w * 0.011), y0 + int(h * 0.055)),
                cv2.FONT_HERSHEY_DUPLEX,
                h / 600.0,
                border,
                max(2, int(h / 360)),
                cv2.LINE_AA,
            )

        if active and dwell > 0:
            _draw_progress_bar(
                frame, x0 + 14, y1 - 16, (x1 - x0) - 28, 7, dwell, MATRIX_GREEN
            )

    # Textes (question + reponses + info) en une passe Pillow.
    _draw_quiz_text(frame, event, reveal, card_top)

    # Curseur main facon cible Matrix + anneau de dwell.
    if event.quiz_cursor is not None and not reveal:
        cx = int(event.quiz_cursor[0] * w)
        cy = int(event.quiz_cursor[1] * h)
        cv2.circle(frame, (cx, cy), 18, MATRIX_GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, MATRIX_GREEN, -1, cv2.LINE_AA)
        # Petits traits de visee (haut/bas/gauche/droite).
        cv2.line(frame, (cx - 26, cy), (cx - 20, cy), MATRIX_GREEN, 2)
        cv2.line(frame, (cx + 20, cy), (cx + 26, cy), MATRIX_GREEN, 2)
        cv2.line(frame, (cx, cy - 26), (cx, cy - 20), MATRIX_GREEN, 2)
        cv2.line(frame, (cx, cy + 20), (cx, cy + 26), MATRIX_GREEN, 2)
        dwell = max(event.quiz_dwell) if event.quiz_dwell else 0.0
        if dwell > 0:
            cv2.ellipse(
                frame, (cx, cy), (24, 24), -90, 0, int(360 * dwell), MATRIX_GREEN, 3, cv2.LINE_AA
            )


def draw_celebration(frame: np.ndarray, event: GameEvent) -> None:
    """Effet visuel de recompense, dispatche par figure reussie.

    `celebration_progress` va de 0 a 1 sur la fenetre : les effets demarrent
    forts et s'estompent pour rendre la main a la figure suivante.
    """
    key = event.celebration_key
    if not key:
        return
    progress = event.celebration_progress
    if key == "neo_dodge":
        _celebrate_dodge(frame, progress)
    elif key == "pill_choice":
        _celebrate_pill(frame, progress, event.chosen_pill or "red")
    elif key == "white_rabbit":
        _celebrate_rabbit(frame, progress)
    elif key == "no_spoon":
        _celebrate_spoon(frame, progress)
    elif key == "bullet_stop":
        _celebrate_bullet_stop(frame, progress)


def _celebrate_dodge(frame: np.ndarray, progress: float) -> None:
    # Bullet-time : noir tout autour + desaturation + trainees de balles.
    # Le rejeu au ralenti des dernieres frames est gere dans la boucle
    # principale ; cet effet s'applique aux frames du rejeu.
    height, width = frame.shape[:2]
    fade = 1.0 - progress * progress

    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(gray, 0.45 * fade, frame, 1.0 - 0.45 * fade, 0, frame)
    apply_vignette(frame, 0.9 * fade)

    overlay = frame.copy()
    for _ in range(5):
        start_y = random.randint(int(height * 0.2), int(height * 0.8))
        sweep = random.randint(-40, 40)
        head_x = int(width * (0.1 + 0.8 * random.random()))
        cv2.line(
            overlay,
            (0, start_y),
            (head_x, start_y + sweep),
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (head_x, start_y + sweep), 4, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.5 * fade, frame, 1.0 - 0.5 * fade, 0, frame)

    _put_centered(frame, "BULLET TIME", int(height * 0.18), 1.2, (235, 235, 235), 2)


def _celebrate_pill(frame: np.ndarray, progress: float, pill: str) -> None:
    # Teinte plein ecran de la couleur avalee + anneaux depuis la pilule.
    height, width = frame.shape[:2]
    color = PILL_COLORS.get(pill, PILL_COLORS["red"])
    fade = 1.0 - progress

    tint = np.full_like(frame, color, dtype=np.uint8)
    cv2.addWeighted(tint, 0.45 * fade, frame, 1.0 - 0.45 * fade, 0, frame)

    zone_x, zone_y = PILL_ZONES.get(pill, PILL_ZONES["red"])
    center = (int(zone_x * width), int(zone_y * height))
    for ring in range(3):
        radius = int((progress * 1.2 + ring * 0.25) * width * 0.5)
        if radius > 0:
            cv2.circle(frame, center, radius, color, 3, cv2.LINE_AA)


def _celebrate_rabbit(frame: np.ndarray, progress: float) -> None:
    # Silhouette de lapin blanc qui bondit a travers l'ecran (la pluie passe
    # en blanc via MatrixRain.draw(white=True) dans la boucle principale).
    height, width = frame.shape[:2]
    white = (245, 245, 245)

    rabbit_x = int(progress * (width + 240)) - 120
    hop = int(abs(math.sin(progress * math.pi * 4)) * height * 0.08)
    base_y = int(height * 0.72) - hop

    body_w, body_h = int(height * 0.07), int(height * 0.05)
    head_r = int(height * 0.028)
    cv2.ellipse(frame, (rabbit_x, base_y), (body_w, body_h), 0, 0, 360, white, -1)
    head_x, head_y = rabbit_x + body_w, base_y - body_h
    cv2.circle(frame, (head_x, head_y), head_r, white, -1)
    for ear_dx in (-head_r // 2, head_r // 2):
        cv2.ellipse(
            frame,
            (head_x + ear_dx, head_y - head_r - int(height * 0.035)),
            (max(3, head_r // 3), int(height * 0.04)),
            -10 if ear_dx < 0 else 10,
            0,
            360,
            white,
            -1,
        )
    cv2.circle(
        frame, (rabbit_x - body_w, base_y - body_h // 2), max(4, head_r // 2), white, -1
    )


def _celebrate_spoon(frame: np.ndarray, progress: float) -> None:
    # "La realite se plie" : ondulation de toute l'image, qui s'amortit.
    height, width = frame.shape[:2]
    amplitude = 14.0 * (1.0 - progress)
    if amplitude < 0.5:
        return

    grids = _WAVE_GRID_CACHE.get((height, width))
    if grids is None:
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        grids = np.meshgrid(xs, ys)
        _WAVE_GRID_CACHE[(height, width)] = grids
    grid_x, grid_y = grids

    phase = progress * 18.0
    map_x = grid_x + amplitude * np.sin(grid_y / 28.0 + phase)
    map_y = grid_y + amplitude * 0.6 * np.sin(grid_x / 36.0 + phase)
    frame[:] = cv2.remap(
        frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def _celebrate_bullet_stop(frame: np.ndarray, progress: float) -> None:
    # Onde de choc verte depuis le centre + balles figees qui retombent.
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    fade = max(0.0, 1.0 - progress)

    # Onde de choc : cercle qui s'etend et s'estompe
    shock_r = int(progress * width * 0.65)
    if shock_r > 0:
        cv2.circle(
            frame, (cx, cy), shock_r, MATRIX_GREEN, max(1, int(4 * fade)), cv2.LINE_AA
        )
        if shock_r > 30:
            cv2.circle(frame, (cx, cy), shock_r - 20, MATRIX_DARK_GREEN, 1, cv2.LINE_AA)

    # Balles figees qui retombent (gravite simulee par progress^2)
    N = 14
    drop_y = int(progress * progress * height * 0.6)
    for i in range(N):
        theta = 2 * math.pi * i / N
        bx = int(cx + 60 * math.cos(theta))
        by = int(cy - int(height * 0.25) + drop_y)
        if 0 <= by < height and 0 <= bx < width:
            ball_alpha = max(0.0, fade)
            if ball_alpha > 0.05:
                overlay = frame.copy()
                cv2.circle(overlay, (bx, by), 5, (235, 235, 235), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, ball_alpha, frame, 1.0 - ball_alpha, 0, frame)

    # Texte "YOU ARE THE ONE" qui s'estompe
    if fade > 0.1:
        color = tuple(int(c * fade) for c in MATRIX_PALE_GREEN)
        _put_centered(frame, "YOU ARE THE ONE", int(height * 0.40), 1.4, color, 2)  # type: ignore[arg-type]


def draw_flash(frame: np.ndarray, event: GameEvent) -> None:
    if not event.flash_message:
        return

    height, width = frame.shape[:2]
    band_top, band_bottom = height // 2 - 60, height // 2 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, band_top), (width, band_bottom), (5, 15, 5), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    _put_centered(
        frame, event.flash_message, height // 2, 1.6, MATRIX_GREEN, 3, glow=True
    )


def draw_score_screen(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.55)

    successes = sum(result.success for result in event.round_results)
    headline = "ACCESS GRANTED" if successes else "SYSTEM FAILURE"
    _put_centered(frame, headline, 140, 1.8, MATRIX_GREEN, 3, glow=True)
    _put_centered(frame, f"SCORE  {event.score}", 200, 1.3, MATRIX_PALE_GREEN, 2)
    if event.new_record:
        if _blink(0.5):
            _put_centered(frame, "NEW RECORD !", 245, 1.0, (255, 255, 255), 2)
    elif event.best_score > 0:
        _put_centered(
            frame,
            f"BEST : {event.best_score}",
            245,
            0.8,
            MATRIX_PALE_GREEN,
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )

    y = 290
    for result in event.round_results:
        mark = "[OK]" if result.success else "[--]"
        points = f"+{result.points}" if result.success else "  0"
        color = MATRIX_GREEN if result.success else (90, 130, 95)
        _put_centered(
            frame,
            f"{mark} {result.title.upper()}  {points}",
            y,
            0.75,
            color,
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
        )
        y += 38

    if event.chosen_pill:
        _put_centered(
            frame,
            f"PILL OF CHOICE : {event.chosen_pill.upper()}",
            y + 10,
            0.85,
            PILL_COLORS[event.chosen_pill],
            2,
        )
        y += 48

    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(
            frame, "2 PAUMES OUVERTES POUR REJOUER", height - 70, 0.9, MATRIX_GREEN, 2
        )
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(
            frame,
            (frame.shape[1] - bar_width) // 2,
            height - 50,
            bar_width,
            14,
            event.start_hold_progress,
        )


# ---------------------------------------------------------------------------
# Zone de scan de badge QR (écran de score)
# ---------------------------------------------------------------------------

_QR_ZONE_W = 290
_QR_ZONE_H = 140
_QR_ZONE_MARGIN = 16
_QR_DISPLAY_SEC = 6.0  # durée d'affichage du résultat après scan


def draw_qr_scan_zone(
    frame: np.ndarray,
    qr_result: str | None,
    result_at: float,
    *,
    y_start: int | None = None,
) -> None:
    """Zone en haut à droite permettant aux participants de scanner leur badge QR.

    Affiche une animation de scan quand aucun code n'est lu, puis le contenu
    décodé pendant `_QR_DISPLAY_SEC` secondes.

    `y_start` fixe le bord supérieur de la zone (px). Si None, la zone est
    placée sous le bandeau HUD (comportement par défaut).
    """
    h, w = frame.shape[:2]
    if y_start is None:
        y_start = 80 + _QR_ZONE_MARGIN  # sous le HUD
    zx = w - _QR_ZONE_W - _QR_ZONE_MARGIN
    zy = y_start
    zx2, zy2 = zx + _QR_ZONE_W, zy + _QR_ZONE_H
    if zx < 0 or zy2 > h:
        return

    # Fond semi-opaque
    roi = frame[zy:zy2, zx:zx2]
    dark = np.full_like(roi, (5, 15, 5))
    cv2.addWeighted(dark, 0.80, roi, 0.20, 0, roi)
    frame[zy:zy2, zx:zx2] = roi

    # Bordure verte
    cv2.rectangle(frame, (zx, zy), (zx2 - 1, zy2 - 1), MATRIX_GREEN, 1, cv2.LINE_AA)

    # Titre
    title = "SCANNEZ VOTRE BADGE"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(
        frame, title,
        (zx + (_QR_ZONE_W - tw) // 2, zy + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, MATRIX_GREEN, 1, cv2.LINE_AA,
    )
    cv2.line(frame, (zx + 8, zy + 22), (zx2 - 8, zy + 22), MATRIX_DARK_GREEN, 1)

    now = time.monotonic()
    has_result = (
        qr_result is not None
        and result_at > 0
        and (now - result_at) < _QR_DISPLAY_SEC
    )

    if has_result:
        # Succès : affiche le contenu décodé
        cv2.putText(
            frame, "BADGE LU !",
            (zx + 10, zy + 50),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 255, 80), 2, cv2.LINE_AA,
        )
        assert qr_result is not None  # guaranteed by has_result
        text = qr_result if len(qr_result) <= 34 else qr_result[:31] + "..."
        line_len = 26
        lines = [text[i : i + line_len] for i in range(0, len(text), line_len)][:3]
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (zx + 10, zy + 76 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, MATRIX_PALE_GREEN, 1, cv2.LINE_AA,
            )
    else:
        # Animation de scan : 4 coins + ligne mobile
        cl = 12  # longueur des coins
        pad = 28
        sx, sy = zx + pad, zy + 30
        ex, ey = zx2 - pad, zy2 - 14
        col = MATRIX_GREEN if _blink(0.55) else MATRIX_DARK_GREEN
        cv2.line(frame, (sx, sy), (sx + cl, sy), col, 2)
        cv2.line(frame, (sx, sy), (sx, sy + cl), col, 2)
        cv2.line(frame, (ex, sy), (ex - cl, sy), col, 2)
        cv2.line(frame, (ex, sy), (ex, sy + cl), col, 2)
        cv2.line(frame, (sx, ey), (sx + cl, ey), col, 2)
        cv2.line(frame, (sx, ey), (sx, ey - cl), col, 2)
        cv2.line(frame, (ex, ey), (ex - cl, ey), col, 2)
        cv2.line(frame, (ex, ey), (ex, ey - cl), col, 2)
        # Ligne de scan animée
        scan_y = int(sy + (ey - sy) * ((now % 1.4) / 1.4))
        cv2.line(frame, (sx + 4, scan_y), (ex - 4, scan_y), MATRIX_GREEN, 1, cv2.LINE_AA)
        # Invite textuelle
        hint = "Montrez votre badge"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(
            frame, hint,
            (zx + (_QR_ZONE_W - hw) // 2, zy2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, MATRIX_PALE_GREEN, 1, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def load_challenge_background(
    asset_name: str, frame_shape: tuple[int, int, int]
) -> np.ndarray | None:
    if not asset_name:
        return None

    asset_path = Path(__file__).resolve().parent.parent / "assets" / asset_name
    if not asset_path.exists():
        return None

    background = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
    if background is None:
        return None

    height, width = frame_shape[:2]
    return cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    persons: int,
    faces: int,
    poses: int,
    event: GameEvent,
) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 74), HUD_BG, -1)
    cv2.rectangle(frame, (0, 74), (w, 76), MATRIX_DARK_GREEN, -1)

    cv2.putText(
        frame,
        "MATRIX VISION",
        (20, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MATRIX_GREEN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"PERSONS: {persons}",
        (170, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MATRIX_GREEN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FACES: {faces}",
        (350, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MATRIX_GREEN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"POSES: {poses}",
        (500, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MATRIX_GREEN,
        1,
        cv2.LINE_AA,
    )

    status = f"{event.phase}"
    if event.round_total:
        status += f" | ROUND {event.round_index}/{event.round_total}"
    cv2.putText(
        frame,
        status,
        (w - 440, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        MATRIX_GREEN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SCORE: {event.score}",
        (w - 440, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        MATRIX_GREEN,
        2,
        cv2.LINE_AA,
    )
    if event.best_score > 0:
        cv2.putText(
            frame,
            f"BEST: {event.best_score}",
            (w - 165, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            MATRIX_PALE_GREEN,
            1,
            cv2.LINE_AA,
        )
    if event.combo > 1:
        cv2.putText(
            frame,
            f"COMBO x{event.combo}",
            (w - 165, 58),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
