from __future__ import annotations

import json
import unittest

from src.mqtt_client import build_pill_payload


class MqttClientTests(unittest.TestCase):
    def test_build_pill_payload_serializes_token_and_pill(self) -> None:
        payload = build_pill_payload("secret-token", "red")

        self.assertEqual(json.loads(payload), {"token": "secret-token", "pill": "red"})
