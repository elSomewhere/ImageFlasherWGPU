"""Layer 4 — the crawl engine.

The media-agnostic worker loop. It only ever touches ports: a World to fetch, an
Extractor to parse, an ArtifactSink to emit, plus the pure-core frontier and topic
state. It has no idea whether the World is the real internet or an imagined one.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from urllib.parse import urlparse

from ..adapters.image_pipeline import ImageValidationError, normalize_image
from ..core.frontier import FrontierItem, URLFrontier
from ..core.scoring import ScorePolicy, host_of
from ..core.signals.novelty import NoveltyArchive
from ..core.steering import TopicState
from ..core.temperature import make_controller
from ..core.types import Artifact, Link, MediaCandidate
from ..ports.extractor import Extractor
from ..ports.sink import ArtifactSink
from ..ports.world import FetchError, World
from .profile import Profile


logger = logging.getLogger(__name__)


class CrawlEngine:
    def __init__(
        self,
        profile: Profile,
        *,
        page_world: World,
        media_world: World,
        extractor: Extractor,
        sink: ArtifactSink,
        frontier: URLFrontier | None = None,
        topic_state: TopicState | None = None,
        image_source=None,
        seed_sources=None,
    ) -> None:
        self.profile = profile
        self.page_world = page_world
        self.media_world = media_world
        self.extractor = extractor
        self.sink = sink
        self.image_source = image_source
        self.seed_sources = list(seed_sources or [])
        self.topic_state = topic_state or TopicState()
        self.rng = random.Random(profile.random_seed)
        self.frontier = frontier or URLFrontier(max_size=profile.max_frontier_size, rng=self.rng)
        # Live-tunable walk temperature (the control plane can adjust this at runtime).
        self.temperature = profile.temperature
        # Optional self-driving controller. None => manual/static temperature.
        self.temperature_controller = make_controller(
            profile.temperature_mode,
            temperature=profile.temperature,
            low=profile.temperature_min,
            high=profile.temperature_max,
            rng=self.rng,
        )

        # Multi-signal scoring. host_visits drives the host-freshness signal.
        self.host_visits: dict[str, int] = {}
        self.score_policy = ScorePolicy(
            self.topic_state, self.rng, self.host_visits, focus=profile.focus
        )
        # Optional novelty signal (unlike-recent-content) — also de-dups the wall.
        self.novelty = (
            NoveltyArchive(profile.novelty_capacity) if profile.enable_novelty else None
        )
        self.last_novelty = 1.0

        # Per-host in-flight locks: keep concurrent workers off the same host at once
        # (politeness beyond the rate limiter; complements host-stratified sampling).
        self._host_locks: dict[str, asyncio.Lock] = {}

        self.seen_pages: deque[str] = deque(maxlen=profile.max_seen_urls)
        self.seen_page_set: set[str] = set()
        self.seen_images: deque[str] = deque(maxlen=profile.max_seen_urls)
        self.seen_image_set: set[str] = set()
        self.recent_errors: deque[str] = deque(maxlen=20)
        self.recent_events: deque[dict] = deque(maxlen=80)
        self.pending_commons_keywords: deque[str] = deque()
        self.pending_commons_keyword_set: set[str] = set()
        self.stats = {
            "pages_visited": 0,
            "image_candidates": 0,
            "images_accepted": 0,
            "images_rejected": 0,
            "images_duplicate": 0,
            "pages_rejected": 0,
            "commons_api_queries": 0,
        }

    # -- state / bookkeeping ------------------------------------------------
    def state(self) -> dict:
        return {
            "ok": True,
            "keywords": self.topic_state.keywords,
            "temperature": round(self.temperature, 4),
            "temperature_mode": getattr(self.temperature_controller, "mode", "static"),
            "focus": round(self.score_policy.focus, 4),
            "novelty_enabled": self.novelty is not None,
            "distinct_hosts": len(self.host_visits),
            "frontier_size": len(self.frontier),
            "queue_size": self._queue_size(),
            "seen_pages": len(self.seen_page_set),
            "seen_images": len(self.seen_image_set),
            "pending_commons_keywords": len(self.pending_commons_keywords),
            "recent_errors": list(self.recent_errors),
            "recent_events": list(self.recent_events),
            **self.stats,
        }

    def _queue_size(self) -> int:
        qsize = getattr(self.sink, "qsize", None)
        return qsize() if callable(qsize) else 0

    def remember_page(self, url: str) -> bool:
        if url in self.seen_page_set:
            return False
        if len(self.seen_pages) == self.seen_pages.maxlen and self.seen_pages:
            self.seen_page_set.discard(self.seen_pages[0])
        self.seen_pages.append(url)
        self.seen_page_set.add(url)
        return True

    def remember_image(self, url: str) -> bool:
        if url in self.seen_image_set:
            return False
        if len(self.seen_images) == self.seen_images.maxlen and self.seen_images:
            self.seen_image_set.discard(self.seen_images[0])
        self.seen_images.append(url)
        self.seen_image_set.add(url)
        return True

    def add_error(self, message: str) -> None:
        self.recent_errors.append(message[:300])
        self.add_event("error", message)

    def add_event(self, event_type: str, message: str, **details) -> None:
        event = {
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message[:300],
        }
        if details:
            event["details"] = details
        self.recent_events.append(event)
        if event_type == "error":
            logger.warning("%s", message)
        else:
            logger.info("%s", message)

    # -- scoring (delegated to the pure ScorePolicy) ------------------------
    def inherited_link_score(self, referrer: FrontierItem) -> float:
        return referrer.score * 0.15

    def score_frontier_item(self, item: FrontierItem) -> float:
        item.direct_score = self.score_policy.relevance(item.url, item.context)
        if item.source == "seed":
            item.inherited_score = max(item.inherited_score, 1.0)
        return self.score_policy.score_link(item)

    def score_image(self, candidate: MediaCandidate, page_title: str, page_score: float) -> float:
        return self.score_policy.score_image(candidate, page_title, page_score)

    def rescore_frontier(self) -> None:
        self.frontier.rebuild(self.score_frontier_item)

    # -- frontier admission -------------------------------------------------
    def seed_from_keywords(self) -> None:
        for keyword in self.topic_state.keywords:
            if keyword not in self.pending_commons_keyword_set:
                self.pending_commons_keywords.append(keyword)
                self.pending_commons_keyword_set.add(keyword)
                self.add_event("commons_queued", f"Queued Commons API search: {keyword}")

    def add_seed(self, url: str) -> bool:
        from ..adapters.extractors.html import normalize_url

        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            self.add_event("seed_rejected", f"Rejected seed URL: {url}")
            return False
        item = FrontierItem(
            url=normalized,
            depth=0,
            source="seed",
            context=normalized,
            direct_score=self.topic_state.score_text(normalized),
            inherited_score=1.0,
        )
        item.score = self.score_frontier_item(item)
        added = self.frontier.add(item)
        if added:
            self.add_event("seed", f"Queued seed: {normalized}", score=round(item.score, 3))
        return added

    def add_link(self, url: str, context: str, referrer: FrontierItem) -> bool:
        if referrer.depth + 1 > self.profile.max_depth:
            return False
        # Steering biases the score; it no longer gates. Off-topic links stay in the
        # frontier at a lower priority so the walk can still wander into them.
        item = FrontierItem(
            url=url,
            depth=referrer.depth + 1,
            source="link",
            referrer=referrer.url,
            context=context,
            direct_score=self.score_policy.relevance(url, context),
            inherited_score=self.inherited_link_score(referrer),
        )
        item.score = self.score_policy.score_link(item)
        added = self.frontier.add(item)
        if added:
            self.add_event(
                "link", f"Queued link: {url}", score=round(item.score, 3), depth=item.depth
            )
        return added

    # -- media --------------------------------------------------------------
    async def handle_image_candidate(
        self, candidate: MediaCandidate, page_title: str, page_score: float
    ) -> None:
        # Steering biases image ranking (see crawl_once) but does not reject: every
        # candidate that survives dedup is fair game for the avalanche.
        candidate.score = self.score_image(candidate, page_title, page_score)
        if not self.remember_image(candidate.url):
            return

        try:
            resource = await self.media_world.fetch(candidate.url)
            if not resource.content_type.startswith("image/"):
                raise FetchError(f"Not an image: {resource.content_type or 'unknown'}")
            processed = await asyncio.to_thread(
                normalize_image,
                resource.body,
                size=self.profile.image_size,
                min_width=self.profile.min_image_width,
                min_height=self.profile.min_image_height,
            )
            if self.novelty is not None:
                self.last_novelty = self.novelty.novelty(processed.ahash)
                if self.last_novelty < self.profile.novelty_min:
                    self.stats["images_duplicate"] += 1
                    self.add_event(
                        "image_duplicate",
                        f"Skipped near-duplicate: {resource.final_url}",
                        novelty=round(self.last_novelty, 3),
                    )
                    return
                self.novelty.add(processed.ahash)
            await self.sink.emit(
                Artifact(
                    kind="image",
                    payload=processed.data,
                    width=processed.width,
                    height=processed.height,
                    source_url=resource.final_url,
                    page_url=candidate.page_url,
                    score=candidate.score,
                )
            )
            self.stats["images_accepted"] += 1
            self.add_event(
                "image",
                f"Accepted image: {resource.final_url}",
                score=round(candidate.score, 3),
                width=processed.width,
                height=processed.height,
                queue_size=self._queue_size(),
            )
        except (FetchError, ImageValidationError, Exception) as error:
            self.stats["images_rejected"] += 1
            self.add_error(f"image {candidate.url}: {error}")

    def _host_lock(self, host: str) -> asyncio.Lock:
        lock = self._host_locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            self._host_locks[host] = lock
        return lock

    # -- seeding / restart --------------------------------------------------
    def restart_probability(self) -> float:
        """Chance of a teleport this step. Rises with temperature so a hotter walk
        jumps to fresh regions more often (PageRank-style damping / restart)."""
        return min(1.0, self.profile.restart_probability * self.temperature)

    async def inject_seeds(self, count: int = 1) -> int:
        """Pull fresh entry points from the seed sources into the frontier."""
        injected = 0
        for source in self.seed_sources:
            try:
                for seed in await source.poll(count):
                    if self.add_seed(seed.url):
                        injected += 1
            except Exception as error:  # noqa: BLE001 - a flaky source must not kill the loop
                self.add_error(f"seed {getattr(source, 'name', source)}: {error}")
        return injected

    # -- commons ------------------------------------------------------------
    async def process_next_commons_keyword(self) -> bool:
        if not self.pending_commons_keywords or self.image_source is None:
            return False
        keyword = self.pending_commons_keywords.popleft()
        self.pending_commons_keyword_set.discard(keyword)
        self.add_event("commons_fetch", f"Searching Commons API: {keyword}")
        try:
            candidates = await self.image_source.fetch_candidates_async(keyword)
            self.stats["commons_api_queries"] += 1
            self.stats["image_candidates"] += len(candidates)
            self.add_event(
                "commons_results",
                f"Commons API returned {len(candidates)} image candidate(s) for: {keyword}",
            )
            for candidate in candidates:
                await self.handle_image_candidate(candidate, keyword, page_score=1.0)
        except Exception as error:
            self.add_error(str(error))
        return True

    # -- worker loop --------------------------------------------------------
    async def crawl_once(self, worker_id: int) -> None:
        # Self-driving temperature reacts to how novel recent finds have been.
        if self.temperature_controller is not None:
            self.temperature = self.temperature_controller.update(self.last_novelty)

        # Occasional teleport keeps the walk from ossifying and opens new regions.
        if self.seed_sources and self.rng.random() < self.restart_probability():
            await self.inject_seeds(1)

        item = self.frontier.sample(self.temperature, window=self.profile.selection_window)
        if item is None:
            # Never idle when we can generate our own entry points.
            if await self.inject_seeds(1):
                return
            if not await self.process_next_commons_keyword():
                await asyncio.sleep(self.profile.empty_frontier_delay_seconds)
            return
        if not self.remember_page(item.url):
            return
        host = host_of(item.url)
        self.host_visits[host] = self.host_visits.get(host, 0) + 1

        try:
            self.add_event(
                "page_fetch", f"Fetching page: {item.url}", depth=item.depth, score=round(item.score, 3)
            )
            async with self._host_lock(host):
                resource = await self.page_world.fetch(item.url)
            images, links, page_title = self.extractor.extract(resource)
            self.stats["pages_visited"] += 1
            self.stats["image_candidates"] += len(images)
            self.add_event(
                "page",
                f"Crawled page: {resource.final_url}",
                images=len(images),
                links=len(links),
                worker=worker_id,
            )

            ranked_images = sorted(
                images,
                key=lambda candidate: self.score_image(candidate, page_title, item.score),
                reverse=True,
            )
            for candidate in ranked_images[:20]:
                await self.handle_image_candidate(candidate, page_title, item.score)

            for link in links[:100]:
                self.add_link(link.url, link.context, item)
        except Exception as error:
            self.stats["pages_rejected"] += 1
            self.add_error(f"page {item.url}: {error}")

        await asyncio.sleep(self.profile.worker_delay_seconds)

    async def worker(self, worker_id: int) -> None:
        while True:
            await self.crawl_once(worker_id)
