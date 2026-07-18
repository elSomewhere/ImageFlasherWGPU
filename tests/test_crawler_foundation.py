import asyncio
import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from crawler.adapters.compliance.ratelimit import OriginScheduler
from crawler.adapters.compliance.robots import RobotsPolicy
from crawler.adapters.image_pipeline import ImageValidationError, normalize_image
from crawler.adapters.inmemory_world import InMemoryWorld
from crawler.adapters.sinks.websocket import ArtifactBroker
from crawler.core.artifact_protocol import (
    ArtifactProtocolError,
    decode_artifact_frame,
    encode_artifact,
)
from crawler.core.types import Artifact, RightsMetadata
from crawler.core.url_policy import canonicalize_url, rejection_reason
from crawler.ports.world import FetchError, Resource
from crawler.runtime.control import ControlPlane
from crawler.runtime.engine import CrawlEngine
from crawler.runtime.profile import Profile
from crawler.adapters.extractors.html import HtmlExtractor, select_srcset


def artifact(payload: bytes, *, rights: RightsMetadata | None = None) -> Artifact:
    return Artifact(
        kind="image",
        payload=payload,
        mime="image/png",
        source_url="https://media.example/image.png",
        rights=rights or RightsMetadata(),
    )


class ArtifactProtocolTests(unittest.TestCase):
    def test_round_trip_preserves_header_and_payload(self):
        original = artifact(
            b"png-bytes",
            rights=RightsMetadata(status="known", license="CC BY 4.0"),
        )
        original.session_id = "session"
        original.sequence = 42
        original.content_hash = "sha256:test"
        original.metadata = {"signal": 0.75}

        header, payload = decode_artifact_frame(encode_artifact(original))

        self.assertEqual(payload, b"png-bytes")
        self.assertEqual(header["protocol"], 1)
        self.assertEqual(header["sequence"], 42)
        self.assertEqual(header["rights"]["license"], "CC BY 4.0")
        self.assertEqual(header["metadata"]["signal"], 0.75)

    def test_rejects_truncated_payload(self):
        frame = encode_artifact(artifact(b"payload"))
        with self.assertRaises(ArtifactProtocolError):
            decode_artifact_frame(frame[:-1])

    def test_rejects_oversized_header_prefix(self):
        with self.assertRaises(ArtifactProtocolError):
            decode_artifact_frame((65_537).to_bytes(4, "little") + b"{}")


class ArtifactBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_without_viewer_and_evicts_oldest(self):
        broker = ArtifactBroker("127.0.0.1", 0, capacity=2)
        first = artifact(b"one")
        second = artifact(b"two")
        third = artifact(b"three")

        await broker.emit(first)
        await broker.emit(second)
        await broker.emit(third)

        self.assertEqual([item.payload for item in broker.snapshot()], [b"two", b"three"])
        self.assertEqual([item.sequence for item in broker.snapshot()], [2, 3])
        self.assertTrue(third.content_hash.startswith("sha256:"))
        self.assertTrue(third.acquired_at)
        self.assertEqual(broker.state()["artifacts_evicted"], 1)

    async def test_known_rights_policy_rejects_unknown_artifacts(self):
        broker = ArtifactBroker("127.0.0.1", 0, content_policy="open-license")
        self.assertFalse(await broker.emit(artifact(b"unknown")))
        self.assertTrue(
            await broker.emit(
                artifact(b"known", rights=RightsMetadata(status="known", license="CC0"))
            )
        )
        self.assertEqual(broker.qsize(), 1)
        self.assertEqual(broker.state()["artifacts_policy_rejected"], 1)

    async def test_gpu_ack_credit_window_paces_snapshot(self):
        class FakeSocket:
            def __init__(self):
                self.sent = []
                self.incoming = asyncio.Queue()

            async def send(self, frame):
                self.sent.append(frame)

            def __aiter__(self):
                return self

            async def __anext__(self):
                message = await self.incoming.get()
                if message is None:
                    raise StopAsyncIteration
                return message

        async def wait_for_count(socket, count):
            while len(socket.sent) < count:
                await asyncio.sleep(0)

        broker = ArtifactBroker(
            "127.0.0.1", 0, capacity=4, client_queue_size=2, client_inflight=1
        )
        for value in (b"one", b"two", b"three"):
            await broker.emit(artifact(value))
        client_id, client, initial = await broker._register()
        socket = FakeSocket()
        sender = asyncio.create_task(broker._sender(socket, client, initial))
        receiver = asyncio.create_task(broker._receiver(socket, client))
        try:
            await asyncio.wait_for(wait_for_count(socket, 1), timeout=1)
            await asyncio.sleep(0.01)
            self.assertEqual(len(socket.sent), 1)
            first_header, _ = decode_artifact_frame(socket.sent[0])
            await socket.incoming.put(
                json.dumps(
                    {
                        "type": "ack",
                        "stage": "gpu_uploaded",
                        "sequence": first_header["sequence"],
                    }
                )
            )
            await asyncio.wait_for(wait_for_count(socket, 2), timeout=1)
            self.assertEqual(len(socket.sent), 2)
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
            await broker._unregister(client_id)


class UrlPolicyTests(unittest.TestCase):
    def test_canonicalization_drops_tracking_and_sorts_query(self):
        self.assertEqual(
            canonicalize_url(
                "../A page/?b=2&utm_source=x&a=1#fragment",
                "HTTPS://Example.COM/root/index.html",
            ),
            "https://example.com/A%20page/?a=1&b=2",
        )

    def test_rejects_credentials_actions_and_crawl_traps(self):
        self.assertEqual(canonicalize_url("https://user:secret@example.com/"), "")
        self.assertEqual(
            rejection_reason("https://example.com/article?action=edit"), "action_url"
        )
        self.assertEqual(
            rejection_reason("https://example.com/2026/07/12/"), "calendar_trap"
        )

    def test_srcset_selects_smallest_sufficient_rendition(self):
        self.assertEqual(
            select_srcset("small.jpg 240w, target.jpg 480w, huge.jpg 1600w", 384),
            "target.jpg",
        )


class RobotsPolicyTests(unittest.IsolatedAsyncioTestCase):
    class FakeWorld:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        async def fetch(self, url):
            self.calls += 1
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    async def test_404_allows_while_server_failure_temporarily_denies(self):
        missing = RobotsPolicy(
            self.FakeWorld(FetchError("missing", status=404)), "InstallationBot"
        )
        unavailable = RobotsPolicy(
            self.FakeWorld(FetchError("down", status=503)), "InstallationBot"
        )
        self.assertTrue(await missing.allowed("https://example.com/page"))
        self.assertFalse(await unavailable.allowed("https://example.com/page"))

    async def test_rules_cache_and_publish_crawl_delay(self):
        resource = Resource(
            final_url="https://example.com/robots.txt",
            status=200,
            content_type="text/plain",
            body=b"User-agent: *\nDisallow: /private\nCrawl-delay: 3\n",
            headers={"cache-control": "max-age=120"},
        )
        world = self.FakeWorld(resource)
        scheduler = OriginScheduler(delay_seconds=0)
        policy = RobotsPolicy(world, "InstallationBot", scheduler=scheduler)

        self.assertFalse(await policy.allowed("https://example.com/private/file"))
        self.assertTrue(await policy.allowed("https://example.com/public"))
        self.assertEqual(world.calls, 1)
        self.assertEqual(scheduler.crawl_delays["example.com"], 3.0)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_origin_waiter_releases_global_permit(self):
        scheduler = OriginScheduler(
            global_concurrency=2, per_origin_concurrency=1, delay_seconds=0
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with scheduler.slot("https://same.example/one"):
                entered.set()
                await release.wait()

        async def waiter():
            async with scheduler.slot("https://same.example/two"):
                pass

        holding = asyncio.create_task(holder())
        await entered.wait()
        waiting = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        self.assertEqual(scheduler.global_semaphore._value, 1)
        release.set()
        await holding


class ValidationAndControlTests(unittest.TestCase):
    def test_pixel_limit_is_enforced(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, format="PNG")
        with self.assertRaises(ImageValidationError):
            normalize_image(buffer.getvalue(), max_pixels=9_999)

    def test_control_rejects_non_finite_exploration(self):
        world = InMemoryWorld({})
        engine = CrawlEngine(
            Profile(),
            page_world=world,
            media_world=world,
            extractor=HtmlExtractor(),
            sink=ArtifactBroker("127.0.0.1", 0),
        )
        control = ControlPlane(engine, "127.0.0.1", 0)
        response = control.dispatch({"type": "set_exploration", "exploration": "nan"})
        self.assertFalse(response["ok"])


if __name__ == "__main__":
    unittest.main()
