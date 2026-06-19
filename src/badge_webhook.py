"""Extraction de contact badge QR et envoi vers le webhook Airtable.

Ce module est volontairement indépendant des dépendances lourdes (cv2,
MediaPipe, YOLO) pour faciliter les tests unitaires et la maintenance.
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import urllib.error
import urllib.request

AIRTABLE_BADGE_WEBHOOK_URL: str = os.getenv(
    "AIRTABLE_BADGE_WEBHOOK_URL",
    "https://hooks.airtable.com/workflows/v1/genericWebhook/app9r1C7cz6h0hsHV/wflLh1rAewu3gUYEG/wtrkzhCUU42GBIN5S",
)
WEBHOOK_TIMEOUT_SEC = 2.0
WEBHOOK_REPOST_COOLDOWN_SEC = 8.0


def _norm(value: object) -> str:
    return str(value or "").strip()


def _make_payload(
    first_name: str, last_name: str, email: str
) -> dict[str, str] | None:
    email = email.strip().lower()
    if not email:
        return None
    return {
        "prenom": first_name.strip(),
        "nom": last_name.strip(),
        "email": email,
    }


def extract_badge_contact(qr_data: str) -> dict[str, str] | None:
    """Extrait prenom/nom/email depuis un QR badge BreizhCamp (format CSV).

    Format attendu (avec ou sans entête) :
        id,lastname,firstname,email,company,ticketType,...
        1,Baratheon,Robert,robert.baratheon@example.com,...

    Retourne None si les données sont absentes ou si l'email est manquant.
    """
    raw = (qr_data or "").strip()
    if not raw:
        return None

    try:
        reader = csv.reader(io.StringIO(raw))
        rows = [r for r in reader if any(c.strip() for c in r)]
        if not rows:
            return None

        first_lower = [c.strip().lower() for c in rows[0]]
        if any(h in first_lower for h in ("lastname", "firstname", "email", "id")):
            # Entête présent → lecture par nom de colonne
            data_rows = rows[1:]
            col_idx = {name: i for i, name in enumerate(first_lower)}
        else:
            # Pas d'entête → convention positionnelle BreizhCamp
            # id(0), lastname(1), firstname(2), email(3)
            data_rows = rows
            col_idx = {"id": 0, "lastname": 1, "firstname": 2, "email": 3}

        for row in data_rows:
            def _get(key: str, _row: list[str] = row, _idx: dict = col_idx) -> str:
                idx = _idx.get(key)
                return _norm(_row[idx]) if idx is not None and idx < len(_row) else ""

            payload = _make_payload(_get("firstname"), _get("lastname"), _get("email"))
            if payload is not None:
                return payload
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Envoi HTTP
# ---------------------------------------------------------------------------

def post_to_airtable(payload: dict[str, str]) -> None:
    """Envoie le payload JSON vers le webhook Airtable (appel synchrone).

    Appelé depuis un thread daemon via `send_async` pour ne pas bloquer la
    boucle vidéo.
    """
    if not AIRTABLE_BADGE_WEBHOOK_URL:
        return
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        AIRTABLE_BADGE_WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SEC) as response:
            status = response.getcode()
        print(f"[BADGE] webhook sent ({status}) for {payload.get('email', '')}")
    except urllib.error.URLError as exc:
        print(f"[BADGE] webhook failed: {exc}")


def send_async(payload: dict[str, str]) -> None:
    """Envoie le payload dans un thread daemon (non-bloquant)."""
    threading.Thread(target=post_to_airtable, args=(payload,), daemon=True).start()

