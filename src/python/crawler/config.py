from dataclasses import dataclass


@dataclass(frozen=True)
class CrawlerConfig:
    image_host: str = "127.0.0.1"
    image_port: int = 5010
    control_host: str = "127.0.0.1"
    control_port: int = 5011
    user_agent: str = "ImageFlasherWGPU-Crawler/1.0 (+local art crawler; respects robots.txt)"
    request_timeout: float = 8.0
    page_delay_seconds: float = 1.0
    worker_delay_seconds: float = 0.2
    empty_frontier_delay_seconds: float = 1.0
    send_delay_seconds: float = 0.25
    max_depth: int = 2
    max_page_bytes: int = 1_000_000
    max_image_bytes: int = 8_000_000
    max_frontier_size: int = 5_000
    max_seen_urls: int = 25_000
    max_queue_size: int = 250
    crawler_workers: int = 2
    image_size: int = 512
    min_image_width: int = 64
    min_image_height: int = 64
    min_link_topic_score: float = 0.1
    min_image_topic_score: float = 0.1
    commons_api_limit: int = 30

