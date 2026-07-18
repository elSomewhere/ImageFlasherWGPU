import asyncio
import io
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from crawler.adapters.compliance.robots import RobotsGuard
from crawler.adapters.compliance.ssrf import SsrfGuard
from crawler.adapters.extractors.html import HtmlExtractor, discover, normalize_url, parse_srcset
from crawler.adapters.image_pipeline import normalize_image
from crawler.adapters.processors import ImageProcessor
from crawler.adapters.inmemory_world import InMemoryWorld
from crawler.adapters.seeds.commons import CommonsImageSource
from crawler.core.frontier import FrontierItem, URLFrontier
from crawler.core.steering import TopicState
from crawler.core.types import Artifact
from crawler.ports.world import FetchError, Resource
from crawler.ports.processor import ProcessorRegistry
from crawler.runtime.control import ControlPlane
from crawler.runtime.engine import CrawlEngine
from crawler.runtime.profile import Profile


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _CollectingSink:
    def __init__(self):
        self.artifacts = []

    async def emit(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def qsize(self) -> int:
        return len(self.artifacts)


def build_engine(world, *, image_source=None, profile=None) -> CrawlEngine:
    """Wire an engine against an arbitrary World with NO compliance layers,
    proving the core is transport-agnostic."""
    profile = profile or Profile()
    return CrawlEngine(
        profile,
        page_world=world,
        media_world=world,
        extractor=HtmlExtractor(),
        sink=_CollectingSink(),
        image_source=image_source,
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
    )


def make_png(width=128, height=96, color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class DiscoveryTests(unittest.TestCase):
    def test_discovers_images_and_links(self):
        html = """
        <html>
          <head>
            <title>Brutalist Concrete Archive</title>
            <meta property="og:image" content="/hero.jpg">
          </head>
          <body>
            <a href="/gallery">Gallery</a>
            <img src="photo.jpg" alt="concrete tower">
            <img srcset="small.jpg 480w, large.jpg 960w" data-src="lazy.jpg">
          </body>
        </html>
        """
        images, links, title = discover(html, "https://example.com/index.html")

        self.assertEqual(title, "Brutalist Concrete Archive")
        self.assertIn("https://example.com/hero.jpg", {image.url for image in images})
        self.assertIn("https://example.com/photo.jpg", {image.url for image in images})
        self.assertIn("https://example.com/gallery", {link.url for link in links})

    def test_srcset_and_normalization(self):
        self.assertEqual(list(parse_srcset("a.jpg 1x, b.jpg 2x")), ["a.jpg", "b.jpg"])
        self.assertEqual(
            normalize_url("../img.jpg#section", "https://example.com/a/b/page.html"),
            "https://example.com/a/img.jpg",
        )

    def test_extractor_rejects_non_html(self):
        resource = Resource(final_url="https://x/y", status=200, content_type="application/pdf", body=b"%PDF")
        with self.assertRaises(FetchError):
            HtmlExtractor().extract(resource)


class TopicStateTests(unittest.TestCase):
    def test_scores_keyword_matches(self):
        topic_state = TopicState()
        topic_state.set_keywords(["brutalism", "concrete"])

        self.assertGreater(
            topic_state.score_text("a concrete architecture photo"),
            topic_state.score_text("a forest landscape"),
        )


class ComplianceLayerTests(unittest.TestCase):
    class _DummyInner:
        def __init__(self, resource):
            self.resource = resource
            self.called = False

        async def fetch(self, url):
            self.called = True
            return self.resource

    def test_ssrf_blocks_local_initial_address(self):
        inner = self._DummyInner(None)
        guard = SsrfGuard(inner)
        with self.assertRaises(FetchError):
            run(guard.fetch("http://127.0.0.1:8000/private"))
        self.assertFalse(inner.called)  # rejected before touching transport

    def test_ssrf_blocks_redirect_to_local_address(self):
        redirected = Resource(
            final_url="http://127.0.0.1/private", status=200, content_type="text/html", body=b""
        )
        inner = self._DummyInner(redirected)
        guard = SsrfGuard(inner)
        with self.assertRaises(FetchError):
            run(guard.fetch("http://8.8.8.8/page"))  # literal public IP -> no DNS

    def test_robots_allow_and_deny(self):
        class FakeParser:
            def can_fetch(self, user_agent, url):
                return "allowed" in url

        guard = RobotsGuard(self._DummyInner(None), user_agent="test-agent")
        # Pre-seed the cache (parser, fetched_at) so no robots.txt is actually fetched.
        guard._robots["https://example.com/robots.txt"] = (FakeParser(), time.monotonic())
        self.assertTrue(run(guard._allowed("https://example.com/allowed")))
        self.assertFalse(run(guard._allowed("https://example.com/blocked")))

    def test_robots_missing_file_allows(self):
        guard = RobotsGuard(self._DummyInner(None), user_agent="test-agent")
        guard._robots["https://example.com/robots.txt"] = (None, time.monotonic())  # missing
        self.assertTrue(run(guard._allowed("https://example.com/anything")))

    def test_robots_read_parser_uses_our_user_agent(self):
        guard = RobotsGuard(self._DummyInner(None), user_agent="MyAgent/1.0")
        captured = {}

        class FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                return b"User-agent: *\nDisallow: /private\n"

        def fake_urlopen(request, timeout=None):
            captured["ua"] = request.get_header("User-agent")
            return FakeResp()

        with patch("urllib.request.urlopen", fake_urlopen):
            parser = guard._read_parser("https://example.com/robots.txt")
        self.assertEqual(captured["ua"], "MyAgent/1.0")  # not Python-urllib
        self.assertFalse(parser.can_fetch("MyAgent/1.0", "https://example.com/private"))
        self.assertTrue(parser.can_fetch("MyAgent/1.0", "https://example.com/public"))


class CompliancePolishTests(unittest.TestCase):
    def test_backoff_retries_on_429_then_succeeds(self):
        from crawler.adapters.compliance.backoff import BackoffOnStatus

        class FlakyInner:
            def __init__(self):
                self.calls = 0

            async def fetch(self, url):
                self.calls += 1
                if self.calls == 1:
                    raise FetchError("HTTP 429", status=429, retry_after=0.0)
                return Resource(final_url=url, status=200, content_type="image/png", body=b"ok")

        inner = FlakyInner()
        backoff = BackoffOnStatus(inner, max_retries=2, base_delay=0.0)
        result = run(backoff.fetch("https://x/y"))
        self.assertEqual(result.body, b"ok")
        self.assertEqual(inner.calls, 2)

    def test_backoff_gives_up_and_reraises(self):
        from crawler.adapters.compliance.backoff import BackoffOnStatus

        class AlwaysDown:
            async def fetch(self, url):
                raise FetchError("HTTP 503", status=503, retry_after=0.0)

        with self.assertRaises(FetchError):
            run(BackoffOnStatus(AlwaysDown(), max_retries=1, base_delay=0.0).fetch("https://x"))

    def test_extractor_honors_noindex_nofollow(self):
        html = (
            '<html><head><meta name="robots" content="noindex, nofollow">'
            '<title>t</title></head><body><a href="/l">l</a>'
            '<img src="https://x/i.png"></body></html>'
        )
        resource = Resource(final_url="https://x/p", status=200, content_type="text/html", body=html.encode())
        images, links, _ = HtmlExtractor().extract(resource)
        self.assertEqual(images, [])
        self.assertEqual(links, [])

    def test_extractor_honors_x_robots_tag_header(self):
        html = '<html><title>t</title><body><a href="/l">l</a><img src="https://x/i.png"></body></html>'
        resource = Resource(
            final_url="https://x/p",
            status=200,
            content_type="text/html",
            body=html.encode(),
            headers={"x-robots-tag": "nofollow"},
        )
        images, links, _ = HtmlExtractor().extract(resource)
        self.assertEqual(len(images), 1)  # noindex not set -> images kept
        self.assertEqual(links, [])  # nofollow -> links dropped

    def test_rate_limiter_honors_crawl_delay_override(self):
        from crawler.adapters.compliance.ratelimit import HostRateLimiter

        delays = {"slow.example": 10.0}
        limiter = HostRateLimiter(1.0, delays)
        # effective delay is max(default, crawl-delay); assert the map is consulted.
        self.assertEqual(max(limiter.delay_seconds, limiter.crawl_delays.get("slow.example", 0.0)), 10.0)


class FrontierTests(unittest.TestCase):
    def test_rebuild_rescores_existing_items(self):
        topic_state = TopicState()
        topic_state.set_keywords(["cats"])
        frontier = URLFrontier()
        frontier.add(FrontierItem(url="https://example.com/cats", context="cats", score=1))
        frontier.add(FrontierItem(url="https://example.com/dogs", context="dogs", score=10))

        topic_state.set_keywords(["dogs"])
        frontier.rebuild(lambda item: topic_state.score_text(item.url, item.context))

        self.assertEqual(frontier.pop().url, "https://example.com/dogs")


class SelectionTests(unittest.TestCase):
    def _frontier(self, seed=1):
        import random

        frontier = URLFrontier(rng=random.Random(seed))
        frontier.add(FrontierItem(url="https://a", score=10.0))
        frontier.add(FrontierItem(url="https://b", score=1.0))
        frontier.add(FrontierItem(url="https://c", score=0.5))
        return frontier

    def test_zero_temperature_is_argmax(self):
        frontier = self._frontier()
        # Highest score must always be chosen at T=0, regardless of RNG state.
        self.assertEqual(frontier.sample(0.0).url, "https://a")

    def test_high_temperature_spreads_selection(self):
        import random

        counts = {"https://a": 0, "https://b": 0, "https://c": 0}
        rng = random.Random(7)
        for _ in range(600):
            frontier = URLFrontier(rng=rng)
            frontier.add(FrontierItem(url="https://a", score=10.0))
            frontier.add(FrontierItem(url="https://b", score=1.0))
            frontier.add(FrontierItem(url="https://c", score=0.5))
            counts[frontier.sample(5.0).url] += 1
        # At high T every node gets picked sometimes (no tunneling into the max).
        self.assertTrue(all(count > 0 for count in counts.values()), counts)
        # ...but the best node is still the most frequent.
        self.assertEqual(max(counts, key=counts.get), "https://a")


class FrontierEvictionTests(unittest.TestCase):
    def test_full_frontier_evicts_weakest_for_stronger_newcomer(self):
        frontier = URLFrontier(max_size=2)
        frontier.add(FrontierItem(url="https://x/1", score=1.0))
        frontier.add(FrontierItem(url="https://x/2", score=2.0))
        # Stronger newcomer displaces the weakest (score 1.0), not refused.
        self.assertTrue(frontier.add(FrontierItem(url="https://x/3", score=5.0)))
        self.assertEqual(len(frontier), 2)
        remaining = {frontier.pop().url, frontier.pop().url}
        self.assertEqual(remaining, {"https://x/2", "https://x/3"})

    def test_full_frontier_rejects_weaker_newcomer(self):
        frontier = URLFrontier(max_size=1)
        frontier.add(FrontierItem(url="https://x/1", score=9.0))
        self.assertFalse(frontier.add(FrontierItem(url="https://x/2", score=1.0)))
        self.assertEqual(len(frontier), 1)

    def test_sampling_is_host_stratified(self):
        import random

        # Two hosts: one with many high-scoring pages, one with a single lower page.
        # Host-first sampling must still reach the lone host, unlike per-item sampling
        # which would be swamped by the crowded host.
        picked_hosts = set()
        for seed in range(40):
            frontier = URLFrontier(rng=random.Random(seed))
            for i in range(10):
                frontier.add(FrontierItem(url=f"https://crowded.example/{i}", score=5.0))
            frontier.add(FrontierItem(url="https://lonely.example/1", score=4.0))
            item = frontier.sample(1.0)
            picked_hosts.add(item.url.split("/")[2])
        self.assertIn("lonely.example", picked_hosts)


class CrawlEngineTests(unittest.TestCase):
    def test_steering_biases_but_does_not_gate_links(self):
        # New semantics (Phase 3): off-topic links are admitted, just ranked lower.
        engine = build_engine(InMemoryWorld({}), profile=Profile(random_seed=0, focus=1.0))
        engine.topic_state.set_keywords(["cats"])
        referrer = FrontierItem(url="https://example.com/cats", score=1.0, source="seed")

        self.assertTrue(engine.add_link("https://site.example/cats", "cat photos", referrer))
        self.assertTrue(engine.add_link("https://site.example/dogs", "dog photos", referrer))
        self.assertEqual(len(engine.frontier), 2)  # both admitted, none rejected

        ordered = [engine.frontier.pop().url, engine.frontier.pop().url]
        self.assertEqual(ordered[0], "https://site.example/cats")  # relevant one ranks first

    def test_keyword_update_rescores_frontier(self):
        engine = build_engine(InMemoryWorld({}))
        engine.topic_state.set_keywords(["cats"])
        engine.frontier.add(FrontierItem(url="https://example.com/cats", context="cats", score=1))
        engine.frontier.add(FrontierItem(url="https://example.com/dogs", context="dogs", score=0))

        engine.topic_state.set_keywords(["dogs"])
        engine.rescore_frontier()

        self.assertEqual(engine.frontier.pop().url, "https://example.com/dogs")

    def test_commons_api_payload_becomes_image_candidates(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "query": {
                            "pages": {
                                "1": {
                                    "title": "File:Cat.jpg",
                                    "imageinfo": [
                                        {"url": "https://upload.wikimedia.org/cat.jpg", "mime": "image/jpeg"}
                                    ],
                                }
                            }
                        }
                    }
                ).encode("utf-8")

        source = CommonsImageSource(user_agent="ua", request_timeout=5.0)
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            candidates = source.fetch_candidates("cats")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].url, "https://upload.wikimedia.org/cat.jpg")
        self.assertIn("Cat.jpg", candidates[0].context)

    def test_control_commands(self):
        engine = build_engine(InMemoryWorld({}))
        control = ControlPlane(engine, "127.0.0.1", 0)

        keyword_response = control.dispatch({"type": "set_keywords", "keywords": ["cats"]})
        seed_response = control.dispatch({"type": "add_seeds", "seeds": ["https://example.com"]})
        state_response = control.dispatch({"type": "get_state"})

        self.assertTrue(keyword_response["ok"])
        self.assertEqual(keyword_response["keywords"], ["cats"])
        self.assertTrue(seed_response["ok"])
        self.assertEqual(seed_response["added"], ["https://example.com"])
        self.assertTrue(state_response["ok"])

    def test_engine_runs_against_inmemory_world_without_compliance(self):
        """The seam: identical engine, a non-internet world, zero compliance layers."""
        page_url = "http://example.test/page"
        image_url = "http://example.test/img.png"
        html = f'<html><title>t</title><body><img src="{image_url}" alt="x"></body></html>'
        world = InMemoryWorld(
            {
                page_url: InMemoryWorld.html(page_url, html),
                image_url: InMemoryWorld.image(image_url, make_png()),
            }
        )
        engine = build_engine(world)
        engine.add_seed(page_url)

        run(engine.crawl_once(worker_id=0))

        self.assertEqual(engine.stats["images_accepted"], 1)
        self.assertEqual(len(engine.sink.artifacts), 1)
        self.assertEqual(engine.sink.artifacts[0].kind, "image")
        self.assertIn(image_url, world.fetched)


class SeedingTests(unittest.TestCase):
    class _FakeSeedSource:
        name = "fake"

        def __init__(self, urls):
            self._urls = urls
            self.polls = 0

        async def poll(self, limit=1):
            from crawler.core.types import SeedNode

            self.polls += 1
            taken, self._urls = self._urls[:limit], self._urls[limit:]
            return [SeedNode(url=url) for url in taken]

    def test_empty_frontier_injects_seed_instead_of_idling(self):
        source = self._FakeSeedSource(["https://seed.example/one"])
        engine = CrawlEngine(
            Profile(random_seed=1),
            page_world=InMemoryWorld({}),
            media_world=InMemoryWorld({}),
            extractor=HtmlExtractor(),
            sink=_CollectingSink(),
            seed_sources=[source],
        )
        self.assertEqual(len(engine.frontier), 0)

        run(engine.crawl_once(worker_id=0))

        self.assertEqual(source.polls, 1)
        self.assertEqual(len(engine.frontier), 1)  # seeded, did not sleep

    def test_restart_probability_scales_with_exploration(self):
        engine = CrawlEngine(
            Profile(restart_probability=0.1),
            page_world=InMemoryWorld({}),
            media_world=InMemoryWorld({}),
            extractor=HtmlExtractor(),
            sink=_CollectingSink(),
        )
        engine.set_exploration(0.0)
        cold = engine.restart_probability()
        engine.set_exploration(1.0)
        self.assertGreater(engine.restart_probability(), cold)


class TemperatureControllerTests(unittest.TestCase):
    def test_static_mode_has_no_controller(self):
        from crawler.core.temperature import make_controller
        import random

        self.assertIsNone(
            make_controller("static", temperature=0.7, low=0.05, high=2.5, rng=random.Random(0))
        )

    def test_adaptive_reheats_on_low_novelty(self):
        from crawler.core.temperature import AdaptiveReheatTemperature

        controller = AdaptiveReheatTemperature(low=0.1, high=2.0, gain=0.5, start=0.5)
        # Feed a stream of low-novelty (stuck) signals -> temperature must climb.
        temps = [controller.update(0.0) for _ in range(10)]
        self.assertGreater(temps[-1], temps[0])
        # ...and a rich vein (high novelty) must cool it back down.
        cooled = [controller.update(1.0) for _ in range(10)]
        self.assertLess(cooled[-1], temps[-1])

    def test_ou_stays_within_bounds(self):
        from crawler.core.temperature import OUDriftTemperature
        import random

        controller = OUDriftTemperature(mean=0.7, low=0.05, high=2.5, rng=random.Random(1))
        for _ in range(500):
            value = controller.update(0.5)
            self.assertGreaterEqual(value, 0.05)
            self.assertLessEqual(value, 2.5)

    def test_engine_adaptive_mode_updates_temperature(self):
        engine = build_engine(InMemoryWorld({}), profile=Profile(temperature_mode="adaptive"))
        self.assertTrue(engine.autopilot.enabled)
        engine.last_novelty = 0.0
        run(engine.crawl_once(worker_id=0))  # empty frontier, but controller still ticks
        self.assertEqual(engine.state()["temperature_mode"], "autopilot")


class NoveltyTests(unittest.TestCase):
    def test_archive_scores_and_dedups(self):
        from crawler.core.signals.novelty import NoveltyArchive

        archive = NoveltyArchive(capacity=8)
        self.assertEqual(archive.novelty(0b1010), 1.0)  # empty -> maximally novel
        archive.add(0b1010)
        self.assertEqual(archive.novelty(0b1010), 0.0)  # identical -> zero distance
        self.assertGreater(archive.novelty(~0b1010 & ((1 << 64) - 1)), 0.9)  # opposite -> far

    def test_engine_skips_near_duplicate_images(self):
        page_url = "http://dup.test/page"
        img_a = "http://dup.test/a.png"
        img_b = "http://dup.test/b.png"
        png = make_png(128, 96, color=(10, 200, 40))
        html = f'<html><title>t</title><body><img src="{img_a}"><img src="{img_b}"></body></html>'
        world = InMemoryWorld(
            {
                page_url: InMemoryWorld.html(page_url, html),
                img_a: InMemoryWorld.image(img_a, png),
                img_b: InMemoryWorld.image(img_b, png),  # identical bytes -> duplicate
            }
        )
        engine = build_engine(world, profile=Profile(enable_novelty=True))
        engine.add_seed(page_url)
        run(engine.crawl_once(worker_id=0))

        self.assertEqual(engine.stats["images_accepted"], 1)
        self.assertEqual(engine.stats["images_duplicate"], 1)


class ImagePipelineTests(unittest.TestCase):
    def test_normalizes_to_png_canvas(self):
        processed = normalize_image(make_png(128, 96), size=64)
        with Image.open(io.BytesIO(processed.data)) as image:
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.format, "PNG")

    def test_produces_perceptual_hash(self):
        a = normalize_image(make_png(128, 96, color=(255, 0, 0)), size=64)
        b = normalize_image(make_png(128, 96, color=(255, 0, 0)), size=64)
        self.assertEqual(a.ahash, b.ahash)  # identical inputs -> identical fingerprint


if __name__ == "__main__":
    unittest.main()
