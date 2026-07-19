"""Decoupled page traversal, media processing, and artifact publication engine."""
from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import logging
import random
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from ..core.exploration import ExplorationAutopilot
from ..core.frontier import FrontierItem, URLFrontier
from ..core.scoring import ScorePolicy, registrable_domain
from ..core.seen import BoundedLRUMap, RotatingBloomSet
from ..core.signals.novelty import VisualNoveltyArchive
from ..core.steering import TopicState
from ..core.types import Link, MediaCandidate
from ..core.url_policy import canonicalize_url, rejection_reason
from ..ports.extractor import Extractor
from ..ports.processor import ProcessorRegistry
from ..ports.sink import ArtifactSink
from ..ports.world import FetchError, World
from .profile import Profile


logger = logging.getLogger(__name__)
WIKIMEDIA_THUMB = re.compile(r"(/wikipedia/commons)/thumb(/[^/]+/[^/]+/[^/]+)/(?:[^/]+)$")


class _TTLLRUSet:
    def __init__(self, capacity: int, ttl: float) -> None:
        self.capacity = capacity
        self.ttl = ttl
        self._items: OrderedDict[str, float] = OrderedDict()

    def add(self, value: str) -> bool:
        now = time.monotonic()
        previous = self._items.get(value)
        if previous is not None and now - previous < self.ttl:
            self._items.move_to_end(value)
            return False
        self._items[value] = now
        self._items.move_to_end(value)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return True

    def __contains__(self, value: str) -> bool:
        previous = self._items.get(value)
        if previous is None:
            return False
        if time.monotonic() - previous >= self.ttl:
            self._items.pop(value, None)
            return False
        return True

    def __len__(self) -> int:
        return len(self._items)


@dataclass(order=True)
class _MediaItem:
    priority: float
    sequence: int
    candidate: MediaCandidate = field(compare=False)
    page_title: str = field(compare=False, default="")
    page_score: float = field(compare=False, default=0.0)


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
        processors: ProcessorRegistry | None = None,
        scheduler=None,
        robots_policy=None,
    ) -> None:
        self.profile = profile
        self.page_world = page_world
        self.media_world = media_world
        self.extractor = extractor
        self.sink = sink
        self.image_source = image_source
        self.seed_sources = list(seed_sources or [])
        self.processors = processors or ProcessorRegistry()
        self.scheduler = scheduler
        self.robots_policy = robots_policy
        self.topic_state = topic_state or TopicState()
        self.rng = random.Random(profile.random_seed)
        self.frontier = frontier or URLFrontier(
            max_size=profile.max_frontier_size,
            max_per_domain=profile.max_frontier_per_domain,
            rng=self.rng,
        )

        self.autopilot = ExplorationAutopilot(
            profile.exploration,
            enabled=profile.autopilot,
            window=profile.novelty_window,
            chapter_seconds=profile.chapter_seconds,
            chapter_pages=profile.chapter_pages,
        )
        self.exploration = profile.exploration
        self.temperature = self._temperature()
        self.temperature_controller = None  # legacy state field
        # Bounded so an endless walk does not accumulate one entry per domain forever;
        # losing a stale host's freshness count is semantically fine.
        self.host_visits: dict[str, int] = BoundedLRUMap(50_000)
        self.score_policy = ScorePolicy(
            self.topic_state,
            self.rng,
            self.host_visits,
            focus=profile.focus,
            exploration=profile.exploration,
        )
        self.novelty = VisualNoveltyArchive(profile.novelty_capacity) if profile.enable_novelty else None
        self.last_novelty = 1.0

        # Rotating Bloom sets give a horizon of millions of URLs in fixed memory —
        # the walk's "non-repetition memory". Content hashes stay exact (TTL-LRU):
        # they are bounded and exactness matters more for payload dedup, and the
        # visual novelty archive guards the display layer regardless.
        self.seen_page_set = RotatingBloomSet(profile.seen_capacity, profile.seen_ttl_seconds)
        self.seen_image_set = RotatingBloomSet(profile.seen_capacity, profile.seen_ttl_seconds)
        self._content_hashes = _TTLLRUSet(profile.max_seen_urls, profile.seen_ttl_seconds)
        self.recent_errors: deque[str] = deque(maxlen=20)
        self.recent_events: deque[dict] = deque(maxlen=80)
        self.pending_commons_keywords: deque[str] = deque()
        self.pending_commons_keyword_set: set[str] = set()
        self.media_queue: asyncio.PriorityQueue[_MediaItem] = asyncio.PriorityQueue(
            maxsize=profile.media_queue_size
        )
        self._media_sequence = itertools.count()
        self._seed_cursor = 0
        self._last_seed_page_count = 0
        # Injection failure backoff: when every configured source yields nothing
        # (e.g. robots-blocked APIs), stop hammering them every loop.
        self._seed_backoff_seconds = 0.0
        self._seed_backoff_until = 0.0
        self._no_entry_warned = False
        self.stats = {
            "pages_discovered": 0,
            "pages_visited": 0,
            "pages_rejected": 0,
            "link_candidates": 0,
            "links_admitted": 0,
            "links_rejected": 0,
            "image_candidates": 0,
            "media_enqueued": 0,
            "media_queue_dropped": 0,
            "media_page_cap_dropped": 0,
            "media_fetched": 0,
            "media_processed": 0,
            "images_accepted": 0,
            "images_rejected": 0,
            "images_duplicate": 0,
            "exact_duplicates": 0,
            "commons_api_queries": 0,
            "teleports": 0,
        }

    def _temperature(self) -> float:
        return self.profile.temperature_min + self.exploration * (
            self.profile.temperature_max - self.profile.temperature_min
        )

    def set_exploration(self, value: float) -> float:
        self.exploration = self.autopilot.set_base(value)
        self.score_policy.exploration = self.exploration
        self.score_policy.focus = 1.0 - self.exploration
        self.temperature = self._temperature()
        self.rescore_frontier()
        return self.exploration

    def state(self) -> dict:
        broker_state = getattr(self.sink, "state", lambda: {})()
        scheduler_state = self.scheduler.state() if self.scheduler is not None else {}
        robots_state = self.robots_policy.state() if self.robots_policy is not None else {}
        return {
            "ok": True,
            "keywords": self.topic_state.keywords,
            "exploration": round(self.exploration, 4),
            "autopilot": self.autopilot.enabled,
            "content_policy": broker_state.get(
                "content_policy", getattr(self.sink, "content_policy", "broad")
            ),
            "autopilot_signals": self.autopilot.signals(),
            "temperature": round(self.temperature, 4),
            "temperature_mode": "autopilot" if self.autopilot.enabled else "static",
            "focus": round(1.0 - self.exploration, 4),
            "novelty_enabled": self.novelty is not None,
            "distinct_hosts": len(self.host_visits),
            "frontier_size": len(self.frontier),
            "media_queue_size": self.media_queue.qsize(),
            "queue_size": broker_state.get("broker_resident", self._queue_size()),
            "seen_pages": len(self.seen_page_set),
            "seen_images": len(self.seen_image_set),
            "pending_commons_keywords": len(self.pending_commons_keywords),
            "recent_errors": list(self.recent_errors),
            "recent_events": list(self.recent_events),
            "broker": broker_state,
            "scheduler": scheduler_state,
            "robots": robots_state,
            **self.stats,
        }

    def _queue_size(self) -> int:
        qsize = getattr(self.sink, "qsize", None)
        return qsize() if callable(qsize) else 0

    def remember_page(self, url: str) -> bool:
        return self.seen_page_set.add(url)

    @staticmethod
    def media_identity(url: str) -> str:
        parsed = urlsplit(url)
        match = WIKIMEDIA_THUMB.search(parsed.path)
        path = match.group(1) + match.group(2) if match else parsed.path
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))

    def remember_image(self, url: str) -> bool:
        return self.seen_image_set.add(self.media_identity(url))

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
        elif event_type not in {"link"}:
            logger.info("%s", message)

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

    def seed_from_keywords(self) -> None:
        if self.image_source is None:
            if self.topic_state.keywords:
                self.add_event(
                    "keywords_steering_only",
                    "Keywords steer frontier scoring only (no media source enabled)",
                )
            return
        for keyword in self.topic_state.keywords:
            if keyword not in self.pending_commons_keyword_set:
                self.pending_commons_keywords.append(keyword)
                self.pending_commons_keyword_set.add(keyword)
                self.add_event("commons_queued", f"Queued Commons API search: {keyword}")

    def add_seed(self, url: str) -> bool:
        normalized = canonicalize_url(url)
        reason = rejection_reason(normalized) if normalized else "invalid_url"
        if reason:
            self.add_event("seed_rejected", f"Rejected seed URL: {url}", reason=reason)
            return False
        item = FrontierItem(
            url=normalized,
            depth=0,
            source="seed",
            context=normalized,
            inherited_score=1.0,
        )
        item.score = self.score_frontier_item(item)
        added = self.frontier.add(item)
        if added:
            self.stats["pages_discovered"] += 1
            self._no_entry_warned = False
            self.add_event("seed", f"Queued seed: {normalized}", score=round(item.score, 3))
        return added

    def add_link(self, url: str, context: str, referrer: FrontierItem, *, nofollow: bool = False) -> bool:
        max_depth = self.profile.max_depth
        if nofollow or (max_depth is not None and referrer.depth + 1 > max_depth):
            self.stats["links_rejected"] += 1
            return False
        normalized = canonicalize_url(url, referrer.url)
        reason = rejection_reason(normalized) if normalized else "invalid_url"
        if reason or normalized in self.seen_page_set:
            self.stats["links_rejected"] += 1
            return False
        item = FrontierItem(
            url=normalized,
            depth=referrer.depth + 1,
            source="link",
            referrer=referrer.url,
            context=context,
            inherited_score=self.inherited_link_score(referrer),
        )
        item.score = self.score_frontier_item(item)
        added = self.frontier.add(item)
        if added:
            self.stats["links_admitted"] += 1
            self.stats["pages_discovered"] += 1
        else:
            self.stats["links_rejected"] += 1
        return added

    def _rank_links(self, links: list[Link], referrer: FrontierItem) -> list[Link]:
        current_domain = registrable_domain(referrer.url)
        candidates = [link for link in links if not link.nofollow]
        for link in candidates:
            link.score = self.score_policy.relevance(link.url, link.context)
        external = [link for link in candidates if registrable_domain(link.url) != current_domain]
        local = [link for link in candidates if registrable_domain(link.url) == current_domain]
        self.rng.shuffle(external)
        self.rng.shuffle(local)
        external.sort(key=lambda link: link.score, reverse=True)
        local.sort(key=lambda link: link.score, reverse=True)
        ranked: list[Link] = []
        while len(ranked) < 100 and (external or local):
            prefer_external = self.rng.random() < (0.35 + 0.45 * self.exploration)
            bucket = external if prefer_external and external else local if local else external
            ranked.append(bucket.pop(0))
        return ranked

    def enqueue_media(self, candidate: MediaCandidate, page_title: str, page_score: float) -> bool:
        candidate.score = self.score_image(candidate, page_title, page_score)
        if not self.remember_image(candidate.url):
            self.stats["images_duplicate"] += 1
            return False
        item = _MediaItem(
            priority=-candidate.score,
            sequence=next(self._media_sequence),
            candidate=candidate,
            page_title=page_title,
            page_score=page_score,
        )
        try:
            self.media_queue.put_nowait(item)
        except asyncio.QueueFull:
            self.stats["media_queue_dropped"] += 1
            return False
        self.stats["media_enqueued"] += 1
        return True

    async def handle_media_item(self, item: _MediaItem) -> None:
        candidate = item.candidate
        try:
            resource = await self.media_world.fetch(candidate.url)
            self.stats["media_fetched"] += 1
            if candidate.kind == "image" and not resource.content_type.startswith("image/"):
                raise FetchError(f"Not an image: {resource.content_type or 'unknown'}")
            processor = self.processors.get(candidate.kind)
            artifact = await processor.process(candidate, resource, score=candidate.score)
            digest = hashlib.sha256(artifact.payload).hexdigest()
            if not self._content_hashes.add(digest):
                self.stats["exact_duplicates"] += 1
                self.stats["images_duplicate"] += 1
                return
            if self.novelty is not None and candidate.kind == "image":
                dhash = int(artifact.metadata.get("dhash", 0))
                histogram = tuple(artifact.metadata.get("color_histogram", ()))
                self.last_novelty = self.novelty.novelty(dhash, histogram)
                if self.last_novelty < self.profile.novelty_min:
                    self.stats["images_duplicate"] += 1
                    return
                self.novelty.add(dhash, histogram)
            artifact.novelty = self.last_novelty
            published = await self.sink.emit(artifact)
            self.stats["media_processed"] += 1
            if published is False:
                self.stats["images_rejected"] += 1
                return
            self.stats["images_accepted"] += 1
            domain = registrable_domain(artifact.source_url)
            self.autopilot.record(novelty=artifact.novelty, domain=domain, kind=artifact.kind)
            self.add_event(
                "image",
                f"Accepted image: {artifact.source_url}",
                novelty=round(artifact.novelty, 3),
                broker_resident=self._queue_size(),
            )
        except Exception as error:  # a bad artifact must never kill a worker
            self.stats["images_rejected"] += 1
            self.autopilot.record(
                novelty=0.0,
                domain=registrable_domain(candidate.url),
                success=False,
            )
            self.add_error(f"media {candidate.url}: {error}")

    async def media_once(self, *, block: bool = True) -> bool:
        try:
            item = await self.media_queue.get() if block else self.media_queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        try:
            await self.handle_media_item(item)
        finally:
            self.media_queue.task_done()
        return True

    def restart_probability(self) -> float:
        base = max(0.0, self.profile.restart_probability)
        return min(1.0, base * (0.2 + 3.0 * self.exploration))

    async def inject_seeds(self, count: int = 1) -> int:
        if not self.seed_sources:
            return 0
        now = time.monotonic()
        if now < self._seed_backoff_until:
            return 0
        injected = 0
        polled = 0
        attempts = 0
        while injected < count and attempts < len(self.seed_sources):
            source = self.seed_sources[self._seed_cursor % len(self.seed_sources)]
            self._seed_cursor += 1
            attempts += 1
            try:
                for seed in await source.poll(count - injected):
                    polled += 1
                    if self.add_seed(seed.url):
                        injected += 1
            except Exception as error:
                self.add_error(f"seed {getattr(source, 'name', source)}: {error}")
        if injected:
            self._seed_backoff_seconds = 0.0
            self._seed_backoff_until = 0.0
        else:
            # Cool down either way: failing sources must not be hammered every loop,
            # and re-polling duplicates of a healthy walk is pointless. Only a true
            # failure (nothing polled at all) is worth an event.
            self._seed_backoff_seconds = min(max(self._seed_backoff_seconds * 2, 5.0), 300.0)
            self._seed_backoff_until = time.monotonic() + self._seed_backoff_seconds
            if polled == 0:
                self.add_event(
                    "seed_backoff",
                    f"Seed injection yielded nothing; backing off {self._seed_backoff_seconds:.0f}s",
                )
        return injected

    async def process_next_commons_keyword(self) -> bool:
        if not self.pending_commons_keywords or self.image_source is None:
            return False
        keyword = self.pending_commons_keywords.popleft()
        self.pending_commons_keyword_set.discard(keyword)
        try:
            candidates = await self.image_source.fetch_candidates_async(keyword)
            self.stats["commons_api_queries"] += 1
            self.stats["image_candidates"] += len(candidates)
            for candidate in candidates:
                self.enqueue_media(candidate, keyword, 1.0)
        except Exception as error:
            self.add_error(f"Commons API: {error}")
        return True

    async def crawl_once(self, worker_id: int, *, process_media_inline: bool = True) -> None:
        self.exploration = self.autopilot.effective(self.stats["pages_visited"])
        self.score_policy.exploration = self.exploration
        self.score_policy.focus = 1.0 - self.exploration
        self.temperature = self._temperature()

        frontier_was_empty = len(self.frontier) == 0
        seed_due = (
            len(self.frontier) < self.profile.frontier_seed_threshold
            or self.stats["pages_visited"] - self._last_seed_page_count >= self.profile.seed_interval_pages
        )
        teleport = self.autopilot.consume_teleport() or self.rng.random() < self.restart_probability()
        if seed_due or teleport:
            injected = await self.inject_seeds(1)
            if injected:
                self._last_seed_page_count = self.stats["pages_visited"]
                if teleport:
                    self.stats["teleports"] += 1
                if frontier_was_empty:
                    return

        item = self.frontier.sample(self.temperature, window=self.profile.selection_window)
        if item is None:
            if not self.seed_sources and not self._no_entry_warned:
                self._no_entry_warned = True
                self.add_event(
                    "no_entry_points",
                    "Frontier is empty and no seed sources are configured; "
                    "add seeds via --seed or the control API",
                )
            await asyncio.sleep(self.profile.empty_frontier_delay_seconds)
            return
        if not self.remember_page(item.url):
            return
        domain = registrable_domain(item.url)
        self.host_visits[domain] = self.host_visits.get(domain, 0) + 1

        try:
            resource = await self.page_world.fetch(item.url)
            images, links, page_title = self.extractor.extract(resource)
            self.stats["pages_visited"] += 1
            self.stats["link_candidates"] += len(links)
            self.stats["image_candidates"] += len(images)

            # Traversal is admitted before media work, so a heavy page never stalls
            # discovery of the next region.
            for link in self._rank_links(links, item):
                self.add_link(link.url, link.context, item, nofollow=link.nofollow)

            ranked_images = sorted(
                images,
                key=lambda candidate: self.score_image(candidate, page_title, item.score),
                reverse=True,
            )
            cap = self.profile.media_candidates_per_page
            self.stats["media_page_cap_dropped"] += max(0, len(ranked_images) - cap)
            enqueued = sum(
                self.enqueue_media(candidate, page_title, item.score)
                for candidate in ranked_images[:cap]
            )
            if process_media_inline:
                for _ in range(enqueued):
                    await self.media_once(block=False)
            self.add_event(
                "page",
                f"Crawled page: {resource.final_url}",
                images=len(images),
                links=len(links),
                worker=worker_id,
            )
        except Exception as error:
            self.stats["pages_rejected"] += 1
            self.autopilot.record(novelty=0.0, domain=domain, success=False)
            self.add_error(f"page {item.url}: {error}")

    async def page_worker(self, worker_id: int) -> None:
        while True:
            await self.crawl_once(worker_id, process_media_inline=False)

    async def worker(self, worker_id: int) -> None:
        await self.page_worker(worker_id)

    async def media_worker(self, worker_id: int) -> None:
        del worker_id
        while True:
            await self.media_once(block=True)

    async def commons_worker(self) -> None:
        while True:
            if not await self.process_next_commons_keyword():
                await asyncio.sleep(0.5)
