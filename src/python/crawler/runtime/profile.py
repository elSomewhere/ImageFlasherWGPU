"""Layer 4 — configuration split into transport vs. behavior.

``TransportConfig`` holds knobs the outer (real-web) adapters need. ``Profile`` holds
everything the core/engine needs plus which optional layers are switched on. A
generative run would reuse the same Profile with ``compliance=False`` and a different
world adapter — the core is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_USER_AGENT = (
    "ImageFlasherBot/2.0 "
    "(+https://github.com/elSomewhere/ImageFlasherWGPU; public art installation crawler)"
)


@dataclass(frozen=True)
class TransportConfig:
    user_agent: str = DEFAULT_USER_AGENT
    connect_timeout: float = 5.0
    request_timeout: float = 15.0
    page_delay_seconds: float = 1.0
    global_concurrency: int = 16
    per_origin_concurrency: int = 1
    max_page_bytes: int = 2_000_000
    max_image_bytes: int = 8_000_000
    max_robots_bytes: int = 512 * 1024
    max_redirects: int = 10
    max_robots_redirects: int = 5
    max_retries: int = 2
    circuit_breaker_failures: int = 5
    circuit_breaker_seconds: float = 300.0
    commons_api_limit: int = 30


@dataclass(frozen=True)
class Profile:
    # WebSocket servers
    image_host: str = "127.0.0.1"
    image_port: int = 5010
    control_host: str = "127.0.0.1"
    control_port: int = 5011

    # Engine behavior
    crawler_workers: int = 8  # compatibility alias for page_workers
    page_workers: int = 8
    media_workers: int = 8
    worker_delay_seconds: float = 0.2
    empty_frontier_delay_seconds: float = 1.0
    send_delay_seconds: float = 0.0  # deprecated; socket backpressure is used
    max_depth: int = 5
    max_frontier_size: int = 20_000
    max_frontier_per_domain: int = 200
    max_seen_urls: int = 100_000
    seen_ttl_seconds: float = 6 * 3600.0
    max_queue_size: int = 256  # artifact broker capacity
    client_queue_size: int = 32
    client_inflight: int = 16
    media_queue_size: int = 2_000
    media_candidates_per_page: int = 32

    # Media normalization
    image_size: int = 384
    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 40_000_000

    # Walk temperature: 0 = greedy argmax (focused), higher = more wandering.
    temperature: float = 0.7
    selection_window: int = 64
    random_seed: int | None = None

    # Self-driving temperature: "static" | "ou" | "adaptive" (optional). Bounds apply
    # to the drift modes; static stays manually controllable via the control plane.
    temperature_mode: str = "static"
    temperature_min: float = 0.05
    temperature_max: float = 2.5

    # Steering focus: 1 = chase relevance, 0 = pure exploration (host freshness + noise).
    # Steering biases the score; it never gates, so diversity is preserved.
    focus: float = 0.5

    # Public steering surface. Legacy focus/temperature controls map onto this.
    exploration: float = 0.55
    autopilot: bool = True
    chapter_seconds: float = 90.0
    chapter_pages: int = 60
    novelty_window: int = 64

    # Optional novelty signal (unlike-recent-content) — also de-dups the wall.
    enable_novelty: bool = True
    novelty_min: float = 0.05  # below this Hamming distance = near-duplicate, skipped
    novelty_capacity: int = 4096

    # Autonomy: generic seed sources + teleport probability (scaled by temperature).
    enable_wikipedia_seeds: bool = True
    restart_probability: float = 0.03
    frontier_seed_threshold: int = 100
    seed_interval_pages: int = 25

    # Broad web is transient/in-memory. ``open-license`` rejects unknown rights.
    content_policy: str = "broad"

    # Which outer layers to assemble. False -> bare world (e.g. a generative web).
    compliance: bool = True

    transport: TransportConfig = field(default_factory=TransportConfig)
