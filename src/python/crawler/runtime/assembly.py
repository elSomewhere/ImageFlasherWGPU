"""Layer 4 — the composition root.

This is the ONLY place that knows how the layers stack. For the real web it wraps a
raw HttpWorld in the compliance decorators, in the same order the original
SafeFetcher applied them (SSRF first, then robots, then rate limit, then fetch). For
an open/generative world it would swap the inner adapter and drop the wrappers — the
engine it hands back is identical either way.
"""
from __future__ import annotations

import asyncio
import logging

from ..adapters.compliance.backoff import BackoffOnStatus
from ..adapters.compliance.ratelimit import HostRateLimiter, RateLimited
from ..adapters.compliance.robots import RobotsGuard
from ..adapters.compliance.ssrf import SsrfGuard
from ..adapters.extractors.html import HtmlExtractor
from ..adapters.http_world import HttpWorld
from ..adapters.seeds.commons import CommonsImageSource
from ..adapters.seeds.wikipedia_random import WikipediaRandomSeedSource
from ..adapters.sinks.websocket import WebSocketImageSink
from ..ports.world import World
from .control import ControlPlane
from .engine import CrawlEngine
from .profile import Profile


logger = logging.getLogger(__name__)


def build_real_web_worlds(profile: Profile) -> tuple[World, World]:
    """Compose the page/media Worlds for the real internet.

    Order (outer -> inner) reproduces SafeFetcher: SSRF validates the URL first,
    robots gate next, then the per-host politeness delay, then the raw fetch. The
    same shared limiter is used for pages and media so timing is global per host.
    """
    t = profile.transport
    crawl_delays: dict[str, float] = {}
    limiter = HostRateLimiter(t.page_delay_seconds, crawl_delays)
    http_page = HttpWorld(t.user_agent, t.request_timeout, t.max_page_bytes)
    http_media = HttpWorld(t.user_agent, t.request_timeout, t.max_image_bytes)

    if not profile.compliance:
        return http_page, http_media

    # Backoff wraps the raw fetch (retries on 429/503); rate-limit, robots, and SSRF
    # layer outward. RobotsGuard shares crawl_delays with the limiter.
    page_world = SsrfGuard(
        RobotsGuard(
            RateLimited(BackoffOnStatus(http_page), limiter),
            t.user_agent,
            t.request_timeout,
            crawl_delays=crawl_delays,
        )
    )
    media_world = SsrfGuard(RateLimited(BackoffOnStatus(http_media), limiter))
    return page_world, media_world


class Assembly:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        page_world, media_world = build_real_web_worlds(profile)
        t = profile.transport

        self.sink = WebSocketImageSink(
            profile.image_host, profile.image_port, profile.max_queue_size, profile.send_delay_seconds
        )
        seed_sources = []
        if profile.enable_wikipedia_seeds:
            seed_sources.append(WikipediaRandomSeedSource(t.user_agent, t.request_timeout))
        self.engine = CrawlEngine(
            profile,
            page_world=page_world,
            media_world=media_world,
            extractor=HtmlExtractor(),
            sink=self.sink,
            image_source=CommonsImageSource(t.user_agent, t.request_timeout, t.commons_api_limit),
            seed_sources=seed_sources,
        )
        self.control = ControlPlane(self.engine, profile.control_host, profile.control_port)

    async def run(self) -> None:
        workers = [
            asyncio.create_task(self.engine.worker(worker_id))
            for worker_id in range(self.profile.crawler_workers)
        ]
        async with self.sink.serve(), self.control.serve():
            logger.info("Image WebSocket on ws://%s:%s", self.profile.image_host, self.profile.image_port)
            logger.info(
                "Control WebSocket on ws://%s:%s", self.profile.control_host, self.profile.control_port
            )
            self.engine.add_event("startup", "Crawler service started")
            try:
                await asyncio.Future()
            finally:
                for worker in workers:
                    worker.cancel()
