"""Layer 3 — WebSocket artifact sink.

Buffers finished Artifact payloads and streams them to a connected renderer over a
WebSocket. Implements the ArtifactSink port; the engine only sees ``emit``.
"""
from __future__ import annotations

import asyncio
import logging

import websockets
from websockets import WebSocketServerProtocol

from ...core.types import Artifact


logger = logging.getLogger(__name__)


class WebSocketImageSink:
    def __init__(self, host: str, port: int, max_queue_size: int, send_delay_seconds: float) -> None:
        self.host = host
        self.port = port
        self.send_delay_seconds = send_delay_seconds
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queue_size)

    def qsize(self) -> int:
        return self.queue.qsize()

    async def emit(self, artifact: Artifact) -> None:
        await self.queue.put(artifact.payload)

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        logger.info("Image client connected")
        try:
            while True:
                image_task = asyncio.create_task(self.queue.get())
                closed_task = asyncio.create_task(websocket.wait_closed())
                done, pending = await asyncio.wait(
                    {image_task, closed_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if closed_task in done:
                    break
                image = image_task.result()
                await websocket.send(image)
                await asyncio.sleep(self.send_delay_seconds)
        except websockets.ConnectionClosed:
            logger.info("Image client disconnected")

    def serve(self):
        """Return the websockets.serve async context manager for this sink."""
        return websockets.serve(self._handler, self.host, self.port, close_timeout=1)
