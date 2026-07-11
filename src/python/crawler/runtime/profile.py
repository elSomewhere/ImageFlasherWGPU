"""Layer 4 — configuration split into transport vs. behavior.

``TransportConfig`` holds knobs the outer (real-web) adapters need. ``Profile`` holds
everything the core/engine needs plus which optional layers are switched on. A
generative run would reuse the same Profile with ``compliance=False`` and a different
world adapter — the core is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_USER_AGENT = "ImageFlasherWGPU-Crawler/1.0 (+local art crawler; respects robots.txt)"


@dataclass(frozen=True)
class TransportConfig:
    user_agent: str = DEFAULT_USER_AGENT
    request_timeout: float = 8.0
    page_delay_seconds: float = 1.0
    max_page_bytes: int = 1_000_000
    max_image_bytes: int = 8_000_000
    commons_api_limit: int = 30


@dataclass(frozen=True)
class Profile:
    # WebSocket servers
    image_host: str = "127.0.0.1"
    image_port: int = 5010
    control_host: str = "127.0.0.1"
    control_port: int = 5011

    # Engine behavior
    crawler_workers: int = 2
    worker_delay_seconds: float = 0.2
    empty_frontier_delay_seconds: float = 1.0
    send_delay_seconds: float = 0.25
    max_depth: int = 2
    max_frontier_size: int = 5_000
    max_seen_urls: int = 25_000
    max_queue_size: int = 250

    # Media normalization
    image_size: int = 512
    min_image_width: int = 64
    min_image_height: int = 64

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

    # Optional novelty signal (unlike-recent-content) — also de-dups the wall.
    enable_novelty: bool = True
    novelty_min: float = 0.05  # below this Hamming distance = near-duplicate, skipped
    novelty_capacity: int = 4096

    # Autonomy: generic seed sources + teleport probability (scaled by temperature).
    enable_wikipedia_seeds: bool = True
    restart_probability: float = 0.03

    # Which outer layers to assemble. False -> bare world (e.g. a generative web).
    compliance: bool = True

    transport: TransportConfig = field(default_factory=TransportConfig)
