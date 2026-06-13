from __future__ import annotations

import math
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np

from src.challenges import HAND_CONNECTIONS, PILL_ZONES
from src.config import FOREGROUND_GREEN_TINT, RAIN_BACKGROUND_COLOR
from src.game_engine import GameEvent

MATRIX_GREEN = (40, 255, 90)
MATRIX_DARK_GREEN = (20, 120, 45)
MATRIX_PALE_GREEN = (190, 255, 190)
HUD_BG = (10, 25, 10)
PILL_COLORS = {"red": (60, 60, 255), "blue": (255, 140, 40)}  # BGR

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (17, 19), (19, 21),
    (16, 18), (18, 20), (20, 22),
]

POSE_BODY_LANDMARKS = range(11, 33)


class MatrixRain:
    def __init__(self, width: int, height: int, spacing: int = 12, trail_length: int = 4) -> None:
        self.width = width
        self.height = height
        self.spacing = spacing
        self.trail_length = trail_length
        self.columns = max(1, width // spacing)
        self.drops = [random.randint(-height, 0) for _ in range(self.columns)]
        self.charset = string.digits

    def _render(self, canvas: np.ndarray, boost: bool = False, white: bool = False) -> None:
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
                # Pluie blanche pendant la celebration "Follow the White Rabbit"
                color = (brightness, brightness, brightness) if white else (35, brightness, 70)

                cv2.putText(canvas, char, (x, trail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

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
        foreground[:, :, 1] = np.minimum(255.0, foreground[:, :, 1] * (1.0 + tint))  # vert
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
        cv2.putText(frame, text, (x, y), font, scale, MATRIX_DARK_GREEN, thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _darken(frame: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(np.zeros_like(frame), alpha, frame, 1.0 - alpha, 0, frame)


def _blink(period: float = 0.8) -> bool:
    return int(time.monotonic() / period) % 2 == 0


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


def draw_yolo_detections(frame: np.ndarray, cached_boxes: list, model_names, draw: bool = True) -> int:
    persons = 0

    for box in cached_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_names.get(cls_id, str(cls_id)) if hasattr(model_names, "get") else str(cls_id)

        if label == "person":
            persons += 1

        if not draw:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), MATRIX_GREEN, 2)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), MATRIX_DARK_GREEN, 1)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MATRIX_GREEN, 2, cv2.LINE_AA)

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


def draw_pose_skeleton(frame: np.ndarray, pose_results, frame_width: int, frame_height: int) -> int:
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


def draw_hand_detections(frame: np.ndarray, hand_results, frame_width: int, frame_height: int) -> int:
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

    _put_centered(frame, "ENTER THE MATRIX", height // 2 - 70, 2.2, MATRIX_GREEN, 4, glow=True)
    if _blink(1.4):
        _put_centered(frame, "WAKE UP, NEO...", height // 2, 1.0, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)
    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(frame, "MONTRE TES 2 PAUMES POUR COMMENCER", height // 2 + 80, 0.95, MATRIX_GREEN, 2)

    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (frame.shape[1] - bar_width) // 2, height // 2 + 110, bar_width, 14, event.start_hold_progress)

    # L'accroche competitive du stand
    if event.best_score > 0:
        _put_centered(frame, f"BEST TODAY : {event.best_score}", height // 2 + 170, 1.0, MATRIX_PALE_GREEN, 2)


def draw_countdown(frame: np.ndarray, event: GameEvent) -> None:
    height, _ = frame.shape[:2]
    _darken(frame, 0.35)
    value = max(1, event.countdown_value)
    # Pulsation : le chiffre grossit a mesure que la seconde s'ecoule.
    fraction = time.monotonic() % 1.0
    scale = 5.0 + 1.5 * fraction
    _put_centered(frame, "GET READY", height // 2 - 130, 1.1, MATRIX_PALE_GREEN, 2)
    _put_centered(frame, str(value), height // 2 + 70, scale, MATRIX_GREEN, 10, glow=True)


def draw_ready_card(frame: np.ndarray, event: GameEvent) -> None:
    """Ecran "prepare-toi" entre la celebration et l'activation de la figure :
    annonce la figure suivante (titre + consigne) chrono fige, pour laisser le
    temps de comprendre l'epreuve avant qu'elle ne demarre."""
    height, _ = frame.shape[:2]
    _darken(frame, 0.45)

    _put_centered(frame, "PREPARE-TOI", height // 2 - 150, 1.1, MATRIX_PALE_GREEN, 2)
    _put_centered(
        frame, event.challenge_title.upper(), height // 2 - 60, 1.7, MATRIX_GREEN, 3, glow=True
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
    _put_centered(frame, str(countdown), height // 2 + 140, 2.5, MATRIX_GREEN, 6, glow=True)


def draw_round_overlay(frame: np.ndarray, event: GameEvent) -> None:
    height, width = frame.shape[:2]

    # Bandeau figure (sous le HUD)
    round_label = f"ROUND {event.round_index}/{event.round_total}"
    cv2.putText(frame, round_label, (24, 116), cv2.FONT_HERSHEY_DUPLEX, 0.85, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, event.challenge_title.upper(), (24, 158), cv2.FONT_HERSHEY_DUPLEX, 1.15, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, event.challenge_prompt, (24, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.78, MATRIX_PALE_GREEN, 2, cv2.LINE_AA)

    # Progression de la figure (maintien pilule/bunny, cuillere, scan)
    if event.figure_progress > 0:
        _draw_progress_bar(frame, 24, 212, 280, 12, event.figure_progress)

    # Timer : barre + texte. Verte, puis rouge sous 2 s.
    ratio = event.timer_left / event.round_duration if event.round_duration > 0 else 0.0
    timer_color = MATRIX_GREEN if event.timer_left > 2.0 else (60, 60, 255)
    bar_margin = 24
    bar_y = height - 46
    _draw_progress_bar(frame, bar_margin, bar_y, width - 2 * bar_margin, 16, ratio, timer_color)
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
        cv2.putText(frame, label, (cx - text_w // 2, cy + 78), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)


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
        cv2.putText(frame, label, (cx - text_w // 2, cy + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_PALE_GREEN, 2, cv2.LINE_AA)

    if event.figure_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (width - bar_width) // 2, height - 70, bar_width, 14, event.figure_progress)
    elif _blink(0.8):
        _put_centered(frame, "PAUME OUVERTE SUR UNE PILULE", height - 64, 0.85, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)


def draw_blue_ending(frame: np.ndarray, event: GameEvent) -> None:
    """Pilule bleue : camera normale, aucun effet Matrix. Texte discret et
    invite a rejouer (gris neutres, surtout pas de vert Matrix)."""
    height, width = frame.shape[:2]

    _put_centered(frame, "RETOUR A LA REALITE", 90, 1.1, (235, 235, 235), 2)
    _put_centered(frame, "L'HISTOIRE S'ARRETE ICI...", 135, 0.75, (180, 180, 180), 2, cv2.FONT_HERSHEY_SIMPLEX)

    if _blink(1.0) or event.start_hold_progress > 0:
        _put_centered(frame, "2 PAUMES OUVERTES POUR RETOURNER DANS LA MATRICE", height - 64, 0.8, (210, 210, 210), 2, cv2.FONT_HERSHEY_SIMPLEX)
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (width - bar_width) // 2, height - 44, bar_width, 12, event.start_hold_progress, (220, 220, 220))


def draw_intro_hint(frame: np.ndarray) -> None:
    if _blink(1.2):
        cv2.putText(frame, "ESPACE POUR PASSER", (24, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


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
    cv2.ellipse(frame, (tip_x, tip_y), (bowl_axes[0] - 5, bowl_axes[1] - 5), tip_angle, 0, 360, (225, 225, 235), -1)
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
            _put_centered(frame, "PINCE POUCE + INDEX ET TOURNE LA MAIN", height - 90, 0.85, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)
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
            _put_centered(frame, "RAISE YOUR HAND", height - 90, 0.95, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)
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
        cv2.circle(frame, (palm_x, palm_y), ring_r + 8, MATRIX_DARK_GREEN, 1, cv2.LINE_AA)

    # Jauge de progression en bas
    _draw_progress_bar(frame, (width - 240) // 2, height - 30, 240, 10, progress, MATRIX_GREEN)


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
        cv2.line(overlay, (0, start_y), (head_x, start_y + sweep), (235, 235, 235), 2, cv2.LINE_AA)
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
    cv2.circle(frame, (rabbit_x - body_w, base_y - body_h // 2), max(4, head_r // 2), white, -1)


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
    frame[:] = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _celebrate_bullet_stop(frame: np.ndarray, progress: float) -> None:
    # Onde de choc verte depuis le centre + balles figees qui retombent.
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    fade = max(0.0, 1.0 - progress)

    # Onde de choc : cercle qui s'etend et s'estompe
    shock_r = int(progress * width * 0.65)
    if shock_r > 0:
        cv2.circle(frame, (cx, cy), shock_r, MATRIX_GREEN, max(1, int(4 * fade)), cv2.LINE_AA)
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
    _put_centered(frame, event.flash_message, height // 2, 1.6, MATRIX_GREEN, 3, glow=True)


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
        _put_centered(frame, f"BEST : {event.best_score}", 245, 0.8, MATRIX_PALE_GREEN, 2, cv2.FONT_HERSHEY_SIMPLEX)

    y = 290
    for result in event.round_results:
        mark = "[OK]" if result.success else "[--]"
        points = f"+{result.points}" if result.success else "  0"
        color = MATRIX_GREEN if result.success else (90, 130, 95)
        _put_centered(frame, f"{mark} {result.title.upper()}  {points}", y, 0.75, color, 2, cv2.FONT_HERSHEY_SIMPLEX)
        y += 38

    if event.chosen_pill:
        _put_centered(frame, f"PILL OF CHOICE : {event.chosen_pill.upper()}", y + 10, 0.85, PILL_COLORS[event.chosen_pill], 2)
        y += 48

    if _blink(0.8) or event.start_hold_progress > 0:
        _put_centered(frame, "2 PAUMES OUVERTES POUR REJOUER", height - 70, 0.9, MATRIX_GREEN, 2)
    if event.start_hold_progress > 0:
        bar_width = 320
        _draw_progress_bar(frame, (frame.shape[1] - bar_width) // 2, height - 50, bar_width, 14, event.start_hold_progress)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def load_challenge_background(asset_name: str, frame_shape: tuple[int, int, int]) -> np.ndarray | None:
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


def draw_hud(frame: np.ndarray, fps: float, persons: int, faces: int, poses: int, event: GameEvent) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 74), HUD_BG, -1)
    cv2.rectangle(frame, (0, 74), (w, 76), MATRIX_DARK_GREEN, -1)

    cv2.putText(frame, "MATRIX VISION", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, MATRIX_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PERSONS: {persons}", (170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FACES: {faces}", (350, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"POSES: {poses}", (500, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)

    status = f"{event.phase}"
    if event.round_total:
        status += f" | ROUND {event.round_index}/{event.round_total}"
    cv2.putText(frame, status, (w - 440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MATRIX_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SCORE: {event.score}", (w - 440, 58), cv2.FONT_HERSHEY_DUPLEX, 0.7, MATRIX_GREEN, 2, cv2.LINE_AA)
    if event.best_score > 0:
        cv2.putText(frame, f"BEST: {event.best_score}", (w - 165, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MATRIX_PALE_GREEN, 1, cv2.LINE_AA)
    if event.combo > 1:
        cv2.putText(frame, f"COMBO x{event.combo}", (w - 165, 58), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
