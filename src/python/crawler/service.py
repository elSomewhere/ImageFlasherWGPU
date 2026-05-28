import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from urllib.parse import urlencode, urlparse

import websockets
from websockets import WebSocketServerProtocol

from .config import CrawlerConfig
from .discovery import ImageCandidate, discover, normalize_url
from .fetching import FetchError, SafeFetcher
from .frontier import FrontierItem, URLFrontier
from .image_pipeline import ImageValidationError, normalize_image
from .topic_state import TopicState


logger = logging.getLogger(__name__)


class CrawlerService:
    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.topic_state = TopicState()
        self.frontier = URLFrontier(max_size=config.max_frontier_size)
        self.fetcher = SafeFetcher(config)
        self.image_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=config.max_queue_size)
        self.seen_pages: deque[str] = deque(maxlen=config.max_seen_urls)
        self.seen_page_set: set[str] = set()
        self.seen_images: deque[str] = deque(maxlen=config.max_seen_urls)
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
            "pages_rejected": 0,
            "commons_api_queries": 0,
        }

    def state(self) -> dict:
        return {
            "ok": True,
            "keywords": self.topic_state.keywords,
            "frontier_size": len(self.frontier),
            "queue_size": self.image_queue.qsize(),
            "seen_pages": len(self.seen_page_set),
            "seen_images": len(self.seen_image_set),
            "pending_commons_keywords": len(self.pending_commons_keywords),
            "recent_errors": list(self.recent_errors),
            "recent_events": list(self.recent_events),
            **self.stats,
        }

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

    def seed_from_keywords(self) -> None:
        for keyword in self.topic_state.keywords:
            if keyword not in self.pending_commons_keyword_set:
                self.pending_commons_keywords.append(keyword)
                self.pending_commons_keyword_set.add(keyword)
                self.add_event("commons_queued", f"Queued Commons API search: {keyword}")

    def direct_link_score(self, url: str, context: str) -> float:
        return self.topic_state.score_text(url, context)

    def inherited_link_score(self, referrer: FrontierItem) -> float:
        return referrer.score * 0.15

    def score_frontier_item(self, item: FrontierItem) -> float:
        item.direct_score = self.direct_link_score(item.url, item.context)
        if item.source == "seed":
            item.inherited_score = max(item.inherited_score, 1.0)
        return item.direct_score + item.inherited_score

    def add_seed(self, url: str) -> bool:
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
        if referrer.depth + 1 > self.config.max_depth:
            return False
        direct_score = self.direct_link_score(url, context)
        if self.topic_state.keywords and direct_score < self.config.min_link_topic_score:
            self.add_event("link_rejected", f"Rejected off-topic link: {url}", direct_score=round(direct_score, 3))
            return False
        inherited_score = self.inherited_link_score(referrer)
        score = direct_score + inherited_score
        added = self.frontier.add(
            FrontierItem(
                url=url,
                score=score,
                depth=referrer.depth + 1,
                source="link",
                referrer=referrer.url,
                context=context,
                direct_score=direct_score,
                inherited_score=inherited_score,
            )
        )
        if added:
            self.add_event("link", f"Queued link: {url}", score=round(score, 3), depth=referrer.depth + 1)
        return added

    def direct_image_score(self, candidate: ImageCandidate, page_title: str) -> float:
        return self.topic_state.score_text(
            candidate.url,
            candidate.alt,
            candidate.context,
            page_title,
        )

    def score_image(self, candidate: ImageCandidate, page_title: str, page_score: float) -> float:
        return self.direct_image_score(candidate, page_title) + (page_score * 0.2)

    def rescore_frontier(self) -> None:
        self.frontier.rebuild(self.score_frontier_item)

    def fetch_commons_candidates(self, keyword: str) -> list[ImageCandidate]:
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": keyword,
                "gsrnamespace": "6",
                "gsrlimit": str(self.config.commons_api_limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "format": "json",
            }
        )
        api_url = f"https://commons.wikimedia.org/w/api.php?{params}"
        request = urllib.request.Request(api_url, headers={"User-Agent": self.config.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise FetchError(f"Commons API failed for '{keyword}': {error}") from error

        pages = payload.get("query", {}).get("pages", {})
        candidates: list[ImageCandidate] = []
        for page in pages.values():
            title = page.get("title", "")
            for image_info in page.get("imageinfo", []):
                image_url = image_info.get("url", "")
                mime = image_info.get("mime", "")
                if not image_url or (mime and not mime.startswith("image/")):
                    continue
                candidates.append(
                    ImageCandidate(
                        url=image_url,
                        page_url=api_url,
                        alt=title.replace("File:", ""),
                        context=f"{keyword} {title}",
                    )
                )
        return candidates

    async def process_next_commons_keyword(self) -> bool:
        if not self.pending_commons_keywords:
            return False

        keyword = self.pending_commons_keywords.popleft()
        self.pending_commons_keyword_set.discard(keyword)
        self.add_event("commons_fetch", f"Searching Commons API: {keyword}")
        try:
            candidates = await asyncio.to_thread(self.fetch_commons_candidates, keyword)
            self.stats["commons_api_queries"] += 1
            self.stats["image_candidates"] += len(candidates)
            self.add_event("commons_results", f"Commons API returned {len(candidates)} image candidate(s) for: {keyword}")
            for candidate in candidates:
                await self.handle_image_candidate(candidate, keyword, page_score=1.0)
        except Exception as error:
            self.add_error(str(error))
        return True

    async def handle_image_candidate(self, candidate: ImageCandidate, page_title: str, page_score: float) -> None:
        direct_score = self.direct_image_score(candidate, page_title)
        if self.topic_state.keywords and direct_score < self.config.min_image_topic_score:
            self.stats["images_rejected"] += 1
            self.add_event("image_rejected", f"Rejected off-topic image: {candidate.url}", direct_score=round(direct_score, 3))
            return
        candidate.score = self.score_image(candidate, page_title, page_score)
        if not self.remember_image(candidate.url):
            return

        try:
            result = await asyncio.to_thread(self.fetcher.fetch_image, candidate.url)
            processed = await asyncio.to_thread(normalize_image, result.content, self.config)
            await self.image_queue.put(processed.data)
            self.stats["images_accepted"] += 1
            self.add_event(
                "image",
                f"Accepted image: {result.url}",
                score=round(candidate.score, 3),
                width=processed.width,
                height=processed.height,
                queue_size=self.image_queue.qsize(),
            )
        except (FetchError, ImageValidationError, Exception) as error:
            self.stats["images_rejected"] += 1
            self.add_error(f"image {candidate.url}: {error}")

    async def crawl_once(self, worker_id: int) -> None:
        item = self.frontier.pop()
        if item is None:
            if not await self.process_next_commons_keyword():
                await asyncio.sleep(self.config.empty_frontier_delay_seconds)
            return
        if not self.remember_page(item.url):
            return

        try:
            self.add_event("page_fetch", f"Fetching page: {item.url}", depth=item.depth, score=round(item.score, 3))
            result = await asyncio.to_thread(self.fetcher.fetch_page, item.url)
            html = result.content.decode("utf-8", errors="replace")
            images, links, page_title = discover(html, result.url)
            self.stats["pages_visited"] += 1
            self.stats["image_candidates"] += len(images)
            self.add_event(
                "page",
                f"Crawled page: {result.url}",
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

        await asyncio.sleep(self.config.worker_delay_seconds)

    async def crawler_worker(self, worker_id: int) -> None:
        while True:
            await self.crawl_once(worker_id)

    async def image_sender(self, websocket: WebSocketServerProtocol) -> None:
        logger.info("Image client connected")
        try:
            while True:
                image_task = asyncio.create_task(self.image_queue.get())
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
                await asyncio.sleep(self.config.send_delay_seconds)
        except websockets.ConnectionClosed:
            logger.info("Image client disconnected")

    async def handle_control(self, websocket: WebSocketServerProtocol) -> None:
        try:
            raw = await websocket.recv()
            command = json.loads(raw)
            command_type = command.get("type")

            if command_type == "set_keywords":
                keywords = self.topic_state.set_keywords(command.get("keywords", []))
                self.add_event("keywords", f"Set keywords: {', '.join(keywords) or 'none'}")
                self.rescore_frontier()
                self.seed_from_keywords()
                response = {"ok": True, "keywords": keywords, "state": self.state()}
            elif command_type == "add_keywords":
                keywords = self.topic_state.add_keywords(command.get("keywords", []))
                self.add_event("keywords", f"Added keywords. Active: {', '.join(keywords) or 'none'}")
                self.rescore_frontier()
                self.seed_from_keywords()
                response = {"ok": True, "keywords": keywords, "state": self.state()}
            elif command_type == "add_seeds":
                added = [seed for seed in command.get("seeds", []) if self.add_seed(str(seed))]
                self.add_event("seeds", f"Added {len(added)} seed(s)")
                response = {"ok": True, "added": added, "state": self.state()}
            elif command_type == "get_state":
                response = self.state()
            else:
                response = {"ok": False, "error": f"Unknown command type: {command_type}"}

            await websocket.send(json.dumps(response))
        except Exception as error:
            await websocket.send(json.dumps({"ok": False, "error": str(error)}))

    async def run(self) -> None:
        workers = [
            asyncio.create_task(self.crawler_worker(worker_id))
            for worker_id in range(self.config.crawler_workers)
        ]
        async with websockets.serve(
            self.image_sender,
            self.config.image_host,
            self.config.image_port,
            close_timeout=1,
        ), websockets.serve(
            self.handle_control,
            self.config.control_host,
            self.config.control_port,
            close_timeout=1,
        ):
            logger.info("Image WebSocket on ws://%s:%s", self.config.image_host, self.config.image_port)
            logger.info("Control WebSocket on ws://%s:%s", self.config.control_host, self.config.control_port)
            self.add_event("startup", "Crawler service started")
            await asyncio.Future()

        for worker in workers:
            worker.cancel()

