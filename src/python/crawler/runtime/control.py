"""Layer 4 — control plane.

A WebSocket that accepts steering commands (keywords, seeds, state queries) and
applies them to a running engine. Kept separate from the engine so the transport
can change without touching crawl logic.
"""
from __future__ import annotations

import json
import logging

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
                engine.temperature = max(0.0, float(command.get("temperature")))
            except (TypeError, ValueError):
                return {"ok": False, "error": "temperature must be a number >= 0"}
            engine.add_event("temperature", f"Set temperature: {engine.temperature:.3f}")
            return {"ok": True, "temperature": engine.temperature, "state": engine.state()}
        if command_type == "set_temperature_mode":
            from ..core.temperature import make_controller

            mode = str(command.get("mode", "static"))
            profile = engine.profile
            engine.temperature_controller = make_controller(
                mode,
                temperature=engine.temperature,
                low=profile.temperature_min,
                high=profile.temperature_max,
                rng=engine.rng,
            )
            active = getattr(engine.temperature_controller, "mode", "static")
            engine.add_event("temperature_mode", f"Set temperature mode: {active}")
            return {"ok": True, "temperature_mode": active, "state": engine.state()}
        if command_type == "set_focus":
            try:
                engine.score_policy.focus = min(1.0, max(0.0, float(command.get("focus"))))
            except (TypeError, ValueError):
                return {"ok": False, "error": "focus must be a number in [0, 1]"}
            engine.add_event("focus", f"Set focus: {engine.score_policy.focus:.3f}")
            return {"ok": True, "focus": engine.score_policy.focus, "state": engine.state()}
        if command_type == "get_state":
            return engine.state()
        return {"ok": False, "error": f"Unknown command type: {command_type}"}

    def serve(self):
        return websockets.serve(self.handle, self.host, self.port, close_timeout=1)
