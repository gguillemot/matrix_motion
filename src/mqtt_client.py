from __future__ import annotations

import json
from dataclasses import dataclass

try:
    import paho.mqtt.client as mqtt
except Exception as exc:  # pragma: no cover - only hit when dependency is missing
    mqtt = None
    _MQTT_IMPORT_ERROR = exc
else:
    _MQTT_IMPORT_ERROR = None


@dataclass(slots=True)
class MQTTConfig:
    enabled: bool
    host: str
    port: int
    topic: str
    token: str
    client_id: str


def build_pill_payload(token: str, pill: str) -> str:
    return json.dumps({"token": token, "pill": pill})


class MQTTPublisher:
    def __init__(self, cfg: MQTTConfig) -> None:
        self.enabled = cfg.enabled
        self.topic = cfg.topic
        self.token = cfg.token
        self.client = None
        self.connected = False

        if not self.enabled:
            return

        if mqtt is None:
            raise RuntimeError(f"paho-mqtt is unavailable: {_MQTT_IMPORT_ERROR}")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cfg.client_id)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        try:
            client.connect(cfg.host, cfg.port, keepalive=60)
            client.loop_start()
            self.client = client
        except Exception as exc:
            print(f"[MQTT] disabled (connect failed): {exc}")
            self.enabled = False

    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties):
        self.connected = not reason_code.is_failure
        if self.connected:
            print("[MQTT] connected")
        else:
            print(f"[MQTT] connect failed rc={reason_code}")

    def _on_disconnect(self, _client, _userdata, _flags, _reason_code, _properties):
        self.connected = False

    def publish_pill(self, pill: str) -> bool:
        if not self.enabled or self.client is None or mqtt is None:
            return False
        if not self.topic:
            print(f"[MQTT] skipped (no topic configured) pill={pill}")
            return False

        payload = build_pill_payload(self.token, pill)
        try:
            info = self.client.publish(self.topic, payload, qos=0, retain=False)
        except ValueError as exc:
            print(f"[MQTT] publish failed (invalid topic '{self.topic}'): {exc}")
            return False
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
