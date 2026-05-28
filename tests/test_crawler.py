import asyncio
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from crawler.config import CrawlerConfig
from crawler.discovery import discover, normalize_url, parse_srcset
from crawler.fetching import FetchError, SafeFetcher
from crawler.frontier import FrontierItem, URLFrontier
from crawler.image_pipeline import normalize_image
from crawler.service import CrawlerService
from crawler.topic_state import TopicState


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


class TopicStateTests(unittest.TestCase):
    def test_scores_keyword_matches(self):
        topic_state = TopicState()
        topic_state.set_keywords(["brutalism", "concrete"])

        self.assertGreater(
            topic_state.score_text("a concrete architecture photo"),
            topic_state.score_text("a forest landscape"),
        )


class FetchSafetyTests(unittest.TestCase):
    def test_blocks_local_addresses(self):
        fetcher = SafeFetcher(CrawlerConfig())
        with self.assertRaises(FetchError):
            fetcher.validate_public_url("http://127.0.0.1:8000/private")

    def test_blocks_redirect_to_local_address(self):
        class FakeHeaders:
            def get_content_type(self):
                return "text/html"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def geturl(self):
                return "http://127.0.0.1/private"

            def read(self, _size):
                return b""

        fetcher = SafeFetcher(CrawlerConfig())
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(FetchError):
                fetcher._fetch("https://example.com/page", 1024, "example.com")

    def test_robots_allow_and_deny(self):
        class FakeRobotParser:
            def set_url(self, url):
                self.url = url

            def read(self):
                return None

            def can_fetch(self, user_agent, url):
                return "allowed" in url

        fetcher = SafeFetcher(CrawlerConfig())
        with patch("urllib.robotparser.RobotFileParser", FakeRobotParser):
            self.assertTrue(fetcher.allowed_by_robots("https://example.com/allowed"))
            self.assertFalse(fetcher.allowed_by_robots("https://example.com/blocked"))


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


class CrawlerServiceTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)

    def test_unrelated_link_is_not_admitted_by_inherited_score(self):
        service = CrawlerService(CrawlerConfig())
        service.topic_state.set_keywords(["cats"])
        referrer = FrontierItem(url="https://example.com/cats", score=100.0, source="seed")

        admitted = service.add_link("https://example.com/dogs", "dog photos", referrer)

        self.assertFalse(admitted)
        self.assertEqual(len(service.frontier), 0)

    def test_keyword_update_rescores_frontier(self):
        service = CrawlerService(CrawlerConfig())
        service.topic_state.set_keywords(["cats"])
        service.frontier.add(FrontierItem(url="https://example.com/cats", context="cats", score=1))
        service.frontier.add(FrontierItem(url="https://example.com/dogs", context="dogs", score=0))

        service.topic_state.set_keywords(["dogs"])
        service.rescore_frontier()

        self.assertEqual(service.frontier.pop().url, "https://example.com/dogs")

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
                                        {
                                            "url": "https://upload.wikimedia.org/cat.jpg",
                                            "mime": "image/jpeg",
                                        }
                                    ],
                                }
                            }
                        }
                    }
                ).encode("utf-8")

        service = CrawlerService(CrawlerConfig())
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            candidates = service.fetch_commons_candidates("cats")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].url, "https://upload.wikimedia.org/cat.jpg")
        self.assertIn("Cat.jpg", candidates[0].context)

    def test_service_control_commands(self):
        class FakeWebSocket:
            def __init__(self, payload):
                self.payload = payload
                self.sent = None

            async def recv(self):
                return json.dumps(self.payload)

            async def send(self, value):
                self.sent = json.loads(value)

        async def run_command(payload):
            service = CrawlerService(CrawlerConfig())
            websocket = FakeWebSocket(payload)
            await service.handle_control(websocket)
            return websocket.sent

        keyword_response = self.loop.run_until_complete(run_command({"type": "set_keywords", "keywords": ["cats"]}))
        seed_response = self.loop.run_until_complete(run_command({"type": "add_seeds", "seeds": ["https://example.com"]}))
        state_response = self.loop.run_until_complete(run_command({"type": "get_state"}))

        self.assertTrue(keyword_response["ok"])
        self.assertEqual(keyword_response["keywords"], ["cats"])
        self.assertTrue(seed_response["ok"])
        self.assertEqual(seed_response["added"], ["https://example.com"])
        self.assertTrue(state_response["ok"])


class ImagePipelineTests(unittest.TestCase):
    def test_normalizes_to_png_canvas(self):
        source = Image.new("RGB", (128, 96), color=(255, 0, 0))
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG")

        processed = normalize_image(buffer.getvalue(), CrawlerConfig(image_size=64))
        with Image.open(io.BytesIO(processed.data)) as image:
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()

