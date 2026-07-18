"""Layer 4 — control plane.

A WebSocket that accepts steering commands (keywords, seeds, state queries) and
applies them to a running engine. Kept separate from the engine so the transport
can change without touching crawl logic.
"""
from __future__ import annotations

import json
import logging
import math

import websockets
from websockets import WebSocketServerProtocol

from .engine import CrawlEngine


logger = logging.getLogger(__name__)


class ControlPlane:
    def __init__(self, engine: CrawlEngine, host: str, port: int) -> None:
        self.engine = engine
        self.host = host
        self.port = port

    async def handle(self, websocket: WebSocketServerProtocol) -> None:
        try:
            raw = await websocket.recv()
            command = json.loads(raw)
            response = self.dispatch(command)
            await websocket.send(json.dumps(response))
        except Exception as error:  # noqa: BLE001 - report any failure to the client
            await websocket.send(json.dumps({"ok": False, "error": str(error)}))

    def dispatch(self, command: dict) -> dict:
        engine = self.engine
        command_type = command.get("type")

        if command_type == "set_keywords":
            keywords = engine.topic_state.set_keywords(command.get("keywords", []))
            engine.add_event("keywords", f"Set keywords: {', '.join(keywords) or 'none'}")
            engine.rescore_frontier()
            engine.seed_from_keywords()
            return {"ok": True, "keywords": keywords, "state": engine.state()}
        if command_type == "add_keywords":
            keywords = engine.topic_state.add_keywords(command.get("keywords", []))
            engine.add_event("keywords", f"Added keywords. Active: {', '.join(keywords) or 'none'}")
            engine.rescore_frontier()
            engine.seed_from_keywords()
            return {"ok": True, "keywords": keywords, "state": engine.state()}
        if command_type == "add_seeds":
            added = [seed for seed in command.get("seeds", []) if engine.add_seed(str(seed))]
            engine.add_event("seeds", f"Added {len(added)} seed(s)")
            return {"ok": True, "added": added, "state": engine.state()}
        if command_type == "set_temperature":
            try:
                temperature = float(command.get("temperature"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "temperature must be a number >= 0"}
            if not math.isfinite(temperature) or temperature < 0:
                return {"ok": False, "error": "temperature must be a finite number >= 0"}
            profile = engine.profile
            span = max(profile.temperature_max - profile.temperature_min, 1e-9)
            exploration = (temperature - profile.temperature_min) / span
            engine.set_exploration(exploration)
            engine.add_event("exploration", f"Mapped legacy temperature to exploration: {engine.exploration:.3f}")
            return {"ok": True, "temperature": engine.temperature, "exploration": engine.exploration, "state": engine.state()}
        if command_type == "set_temperature_mode":
            mode = str(command.get("mode", "static"))
            engine.autopilot.enabled = mode.lower() in {"ou", "adaptive", "autopilot"}
            active = "autopilot" if engine.autopilot.enabled else "static"
            engine.add_event("temperature_mode", f"Set temperature mode: {active}")
            return {"ok": True, "temperature_mode": active, "state": engine.state()}
        if command_type == "set_focus":
            try:
                focus = float(command.get("focus"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "focus must be a number in [0, 1]"}
            if not math.isfinite(focus) or not 0.0 <= focus <= 1.0:
                return {"ok": False, "error": "focus must be a finite number in [0, 1]"}
            engine.set_exploration(1.0 - focus)
            engine.add_event("focus", f"Mapped legacy focus to exploration: {engine.exploration:.3f}")
            return {"ok": True, "focus": focus, "exploration": engine.exploration, "state": engine.state()}
        if command_type == "set_exploration":
            try:
                requested = float(command.get("exploration"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "exploration must be a number in [0, 1]"}
            if not math.isfinite(requested) or not 0.0 <= requested <= 1.0:
                return {"ok": False, "error": "exploration must be a finite number in [0, 1]"}
            exploration = engine.set_exploration(requested)
            engine.add_event("exploration", f"Set exploration: {exploration:.3f}")
            return {"ok": True, "exploration": exploration, "state": engine.state()}
        if command_type == "set_autopilot":
            engine.autopilot.enabled = bool(command.get("enabled"))
            return {"ok": True, "autopilot": engine.autopilot.enabled, "state": engine.state()}
        if command_type == "set_content_policy":
            policy = str(command.get("policy", "broad"))
            if policy not in {"broad", "open-license"}:
                return {"ok": False, "error": "policy must be broad or open-license"}
            if not hasattr(engine.sink, "content_policy"):
                return {"ok": False, "error": "active sink does not support content policies"}
            engine.sink.content_policy = policy
            return {"ok": True, "content_policy": policy, "state": engine.state()}
        if command_type == "get_state":
            return engine.state()
        return {"ok": False, "error": f"Unknown command type: {command_type}"}

    def serve(self):
        return websockets.serve(self.handle, self.host, self.port, close_timeout=1)
