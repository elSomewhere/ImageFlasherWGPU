#!/usr/bin/env python3
import argparse
import asyncio
import logging
import sys

from crawler.config import CrawlerConfig
from crawler.service import CrawlerService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic topic-steered image crawler")
    parser.add_argument("--image-host", default="127.0.0.1")
    parser.add_argument("--image-port", type=int, default=5010)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5011)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = CrawlerConfig(
        image_host=args.image_host,
        image_port=args.image_port,
        control_host=args.control_host,
        control_port=args.control_port,
    )
    service = CrawlerService(config)

    if args.keyword:
        service.topic_state.set_keywords(args.keyword)
        service.seed_from_keywords()
    for seed in args.seed:
        service.add_seed(seed)

    await service.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    asyncio.run(main())

