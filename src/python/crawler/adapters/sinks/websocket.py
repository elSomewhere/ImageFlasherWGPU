"""Non-blocking rolling artifact broker with independent WebSocket clients."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import websockets
from websockets import WebSocketServerProtocol

from ...core.artifact_protocol import encode_artifact
from ...core.types import Artifact


logger = logging.getLogger(__name__)


@dataclass
class _Client:
    queue: asyncio.Queue[Artifact]
    credits: asyncio.Semaphore
    inflight: set[int]
    dropped: int = 0


class ArtifactBroker:
    """Fixed rolling store and fan-out sink.

    Producers never wait for viewers. Every client receives its own stream; a slow
    client loses its oldest unsent item without affecting crawling or other clients.
    """

    def __init__(
        self,
        host: str,
        port: int,
        capacity: int = 256,
        client_queue_size: int = 32,
        client_inflight: int = 16,
        content_policy: str = "broad",
    ) -> None:
        if capacity < 1 or client_queue_size < 1 or client_inflight < 1:
            raise ValueError("Broker and client capacities must be positive")
        self.host = host
        self.port = port
        self.capacity = capacity
        self.client_queue_size = client_queue_size
        self.client_inflight = client_inflight
        self.content_policy = content_policy
        self.session_id = uuid.uuid4().hex
        self._sequence = 0
        self._ring: deque[Artifact] = deque(maxlen=capacity)
        self._clients: dict[int, _Client] = {}
        self._client_ids = 0
        self._lock = asyncio.Lock()
        self._acks: dict[str, int] = {
            "received": 0,
            "decoded": 0,
            "gpu_uploaded": 0,
            "presented": 0,
            "rejected": 0,
            "skipped": 0,
        }
        self.metrics = {
            "artifacts_published": 0,
            "artifacts_evicted": 0,
            "artifacts_policy_rejected": 0,
            "client_dropped": 0,
            "clients_connected": 0,
            "frames_sent": 0,
        }

    def qsize(self) -> int:
        return len(self._ring)

    def snapshot(self) -> list[Artifact]:
        return list(self._ring)

    def state(self) -> dict:
        return {
            "session_id": self.session_id,
            "broker_resident": len(self._ring),
            "broker_capacity": self.capacity,
            "client_inflight_limit": self.client_inflight,
            "active_clients": len(self._clients),
            "content_policy": self.content_policy,
            "acknowledgements": dict(self._acks),
            **self.metrics,
        }

    async def emit(self, artifact: Artifact) -> bool:
        if self.content_policy == "open-license" and artifact.rights.status != "known":
            self.metrics["artifacts_policy_rejected"] += 1
            return False

        if not artifact.content_hash:
            artifact.content_hash = "sha256:" + hashlib.sha256(artifact.payload).hexdigest()
        if not artifact.acquired_at:
            artifact.acquired_at = datetime.now(timezone.utc).isoformat()
        artifact.session_id = self.session_id

        async with self._lock:
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            if self._sequence == 0:
                self._sequence = 1
            artifact.sequence = self._sequence
            if len(self._ring) == self.capacity:
                self.metrics["artifacts_evicted"] += 1
            self._ring.append(artifact)
            self.metrics["artifacts_published"] += 1

            for client in self._clients.values():
                if client.queue.full():
                    try:
                        client.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    client.dropped += 1
                    self.metrics["client_dropped"] += 1
                client.queue.put_nowait(artifact)
        return True

    async def _register(self) -> tuple[int, _Client, list[Artifact]]:
        async with self._lock:
            self._client_ids += 1
            client_id = self._client_ids
            client = _Client(
                queue=asyncio.Queue(maxsize=self.client_queue_size),
                credits=asyncio.Semaphore(self.client_inflight),
                inflight=set(),
            )
            self._clients[client_id] = client
            self.metrics["clients_connected"] += 1
            return client_id, client, list(self._ring)

    async def _unregister(self, client_id: int) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)

    async def _send(
        self,
        websocket: WebSocketServerProtocol,
        client: _Client,
        artifact: Artifact,
    ) -> None:
        # A client returns this credit only after the renderer uploads the artifact
        # (or explicitly rejects/skips it). This prevents a 256-item reconnect replay
        # from overrunning the much smaller WASM decode queues.
        await client.credits.acquire()
        client.inflight.add(artifact.sequence)
        try:
            await websocket.send(encode_artifact(artifact))
        except BaseException:
            client.inflight.discard(artifact.sequence)
            client.credits.release()
            raise
        self.metrics["frames_sent"] += 1

    async def _sender(
        self,
        websocket: WebSocketServerProtocol,
        client: _Client,
        initial: list[Artifact],
    ) -> None:
        for artifact in initial:
            await self._send(websocket, client, artifact)
        while True:
            await self._send(websocket, client, await client.queue.get())

    async def _receiver(self, websocket: WebSocketServerProtocol, client: _Client) -> None:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "ack":
                continue
            stage = str(event.get("stage", ""))
            if stage in self._acks:
                self._acks[stage] += 1
            if stage not in {"gpu_uploaded", "rejected", "skipped"}:
                continue
            try:
                sequence = int(event.get("sequence"))
            except (TypeError, ValueError):
                continue
            if sequence in client.inflight:
                client.inflight.remove(sequence)
                client.credits.release()

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        client_id, client, initial = await self._register()
        logger.info("Artifact client connected: %s", client_id)
        sender = asyncio.create_task(self._sender(websocket, client, initial))
        receiver = asyncio.create_task(self._receiver(websocket, client))
        try:
            done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            sender.cancel()
            receiver.cancel()
            await self._unregister(client_id)
            logger.info("Artifact client disconnected: %s", client_id)

    def serve(self):
        return websockets.serve(
            self._handler,
            self.host,
            self.port,
            close_timeout=1,
            max_size=None,
        )


class WebSocketImageSink(ArtifactBroker):
    """Compatibility name retained for the existing composition root."""

    def __init__(
        self,
        host: str,
        port: int,
        max_queue_size: int = 256,
        send_delay_seconds: float = 0.0,
        *,
        client_queue_size: int = 32,
        client_inflight: int = 16,
        content_policy: str = "broad",
    ) -> None:
        del send_delay_seconds
        super().__init__(
            host,
            port,
            capacity=max_queue_size,
            client_queue_size=client_queue_size,
            client_inflight=client_inflight,
            content_policy=content_policy,
        )
