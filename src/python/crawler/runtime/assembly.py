"""Layer 4 — the composition root.

This is the only place that knows how the runtime layers stack. Page, media, robots,
and source requests share one SSRF-safe aiohttp transport and one origin scheduler;
robots policy is applied before every page or media fetch. The engine remains
independent of those concrete adapters.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..adapters.compliance.backoff import BackoffOnStatus
from ..adapters.compliance.ratelimit import OriginScheduler
from ..adapters.compliance.robots import RobotsPolicy
from ..adapters.extractors.html import HtmlExtractor
from ..adapters.http_world import AioHttpTransport, HttpWorld
from ..adapters.processors import ImageProcessor
from ..adapters.seeds.commons import CommonsImageSource
from ..adapters.seeds.static import StaticSeedSource
from ..adapters.seeds.wikipedia_random import WikipediaRandomSeedSource
from ..adapters.seeds.wikidata import WikidataOfficialSeedSource
from ..adapters.sinks.websocket import WebSocketImageSink
from ..ports.world import World
from ..ports.processor import ProcessorRegistry
from .control import ControlPlane
from .engine import CrawlEngine
from .profile import Profile


logger = logging.getLogger(__name__)


@dataclass
class RealWebBundle:
    page_world: World
    media_world: World
    transport: AioHttpTransport
    scheduler: OriginScheduler
    robots: RobotsPolicy


def build_real_web_bundle(profile: Profile) -> RealWebBundle:
    """Compose the page/media Worlds for the real internet.

    The transport validates DNS and connected peers against the SSRF policy. Robots
    gates page and media requests, while the shared scheduler enforces global and
    per-origin concurrency, crawl delay, and circuit breaking across both worlds.
    """
    t = profile.transport
    if not profile.compliance:
        raise ValueError("The real-web transport cannot run with compliance disabled")

    scheduler = OriginScheduler(
        global_concurrency=t.global_concurrency,
        per_origin_concurrency=t.per_origin_concurrency,
        delay_seconds=t.page_delay_seconds,
        failure_threshold=t.circuit_breaker_failures,
        circuit_seconds=t.circuit_breaker_seconds,
    )
    transport = AioHttpTransport(
        t.user_agent,
        connect_timeout=t.connect_timeout,
        request_timeout=t.request_timeout,
    )
    robots_http = HttpWorld(
        transport=transport,
        scheduler=scheduler,
        max_bytes=t.max_robots_bytes,
        max_redirects=t.max_robots_redirects,
    )
    robots = RobotsPolicy(robots_http, t.user_agent, scheduler=scheduler)
    page_http = HttpWorld(
        transport=transport,
        scheduler=scheduler,
        before_request=robots.require_allowed,
        max_bytes=t.max_page_bytes,
        max_redirects=t.max_redirects,
    )
    media_http = HttpWorld(
        transport=transport,
        scheduler=scheduler,
        before_request=robots.require_allowed,
        max_bytes=t.max_image_bytes,
        max_redirects=t.max_redirects,
    )
    return RealWebBundle(
        page_world=BackoffOnStatus(page_http, max_retries=t.max_retries),
        media_world=BackoffOnStatus(media_http, max_retries=t.max_retries),
        transport=transport,
        scheduler=scheduler,
        robots=robots,
    )


def build_real_web_worlds(profile: Profile) -> tuple[World, World]:
    bundle = build_real_web_bundle(profile)
    return bundle.page_world, bundle.media_world


def build_seed_sources(profile: Profile, web: RealWebBundle) -> list:
    """Operator seeds are always the recurring refill source; autonomous seeders
    (Wikipedia/Wikidata) are opt-in plugins named in ``profile.seed_plugins``."""
    t = profile.transport
    plugin_builders = {
        "wikipedia_random": lambda: WikipediaRandomSeedSource(
            t.user_agent, t.request_timeout, world=web.page_world
        ),
        "wikidata_official_sites": lambda: WikidataOfficialSeedSource(web.page_world),
    }
    sources = []
    if profile.operator_seeds:
        sources.append(StaticSeedSource(list(profile.operator_seeds), cycle=True))
    for name in profile.seed_plugins:
        builder = plugin_builders.get(name)
        if builder is None:
            raise ValueError(f"Unknown seed plugin: {name!r}")
        sources.append(builder())
    return sources


class Assembly:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.web = build_real_web_bundle(profile)
        t = profile.transport

        self.sink = WebSocketImageSink(
            profile.image_host,
            profile.image_port,
            profile.max_queue_size,
            profile.send_delay_seconds,
            client_queue_size=profile.client_queue_size,
            client_inflight=profile.client_inflight,
            content_policy=profile.content_policy,
        )
        image_source = (
            CommonsImageSource(
                t.user_agent,
                t.request_timeout,
                t.commons_api_limit,
                world=self.web.page_world,
            )
            if profile.enable_commons
            else None
        )
        self.engine = CrawlEngine(
            profile,
            page_world=self.web.page_world,
            media_world=self.web.media_world,
            extractor=HtmlExtractor(),
            sink=self.sink,
            image_source=image_source,
            seed_sources=build_seed_sources(profile, self.web),
            processors=ProcessorRegistry(
                [
                    ImageProcessor(
                        size=profile.image_size,
                        min_width=profile.min_image_width,
                        min_height=profile.min_image_height,
                        max_pixels=profile.max_image_pixels,
                    )
                ]
            ),
            scheduler=self.web.scheduler,
            robots_policy=self.web.robots,
        )
        self.control = ControlPlane(self.engine, profile.control_host, profile.control_port)

    async def run(self) -> None:
        workers = [
            *(
                asyncio.create_task(self.engine.page_worker(worker_id))
                for worker_id in range(self.profile.page_workers)
            ),
            *(
                asyncio.create_task(self.engine.media_worker(worker_id))
                for worker_id in range(self.profile.media_workers)
            ),
        ]
        if self.engine.image_source is not None:
            workers.append(asyncio.create_task(self.engine.commons_worker()))
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
                await asyncio.gather(*workers, return_exceptions=True)
                await self.web.transport.close()
