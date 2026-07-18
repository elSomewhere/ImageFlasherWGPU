#!/usr/bin/env python3
"""Entry point — builds a Profile from CLI flags and runs the assembled crawler.

Layer 5: pick a profile, let the Assembly wire the layers, run. The CLI surface is
unchanged so ``server.js`` launches it exactly as before.
"""
import argparse
import asyncio
import logging
import sys
from dataclasses import replace

from crawler.runtime.assembly import Assembly
from crawler.runtime.profile import Profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic topic-steered image crawler")
    parser.add_argument("--image-host", default="127.0.0.1")
    parser.add_argument("--image-port", type=int, default=5010)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5011)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--exploration", type=float, default=Profile.exploration)
    parser.add_argument("--no-autopilot", action="store_true")
    parser.add_argument(
        "--content-policy",
        choices=("broad", "open-license"),
        default=Profile.content_policy,
    )
    parser.add_argument("--page-delay", type=float, default=Profile().transport.page_delay_seconds)
    parser.add_argument(
        "--global-concurrency",
        type=int,
        default=Profile().transport.global_concurrency,
    )
    parser.add_argument(
        "--per-origin-concurrency",
        type=int,
        default=Profile().transport.per_origin_concurrency,
    )
    parser.add_argument("--page-workers", type=int, default=Profile.page_workers)
    parser.add_argument("--media-workers", type=int, default=Profile.media_workers)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--no-compliance",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.no_compliance:
        parser.error("compliance cannot be disabled for the real-web crawler")
    if not 0.0 <= args.exploration <= 1.0:
        parser.error("--exploration must be between 0 and 1")
    if args.page_delay < 0:
        parser.error("--page-delay must be non-negative")
    if min(
        args.global_concurrency,
        args.per_origin_concurrency,
        args.page_workers,
        args.media_workers,
    ) < 1:
        parser.error("worker and concurrency values must be positive")
    return args


async def main() -> None:
    args = parse_args()
    defaults = Profile()
    profile = replace(
        defaults,
        image_host=args.image_host,
        image_port=args.image_port,
        control_host=args.control_host,
        control_port=args.control_port,
        exploration=args.exploration,
        autopilot=not args.no_autopilot,
        content_policy=args.content_policy,
        page_workers=args.page_workers,
        media_workers=args.media_workers,
        random_seed=args.random_seed,
        compliance=True,
        transport=replace(
            defaults.transport,
            page_delay_seconds=args.page_delay,
            global_concurrency=args.global_concurrency,
            per_origin_concurrency=args.per_origin_concurrency,
        ),
    )
    assembly = Assembly(profile)
    engine = assembly.engine

    if args.keyword:
        engine.topic_state.set_keywords(args.keyword)
        engine.seed_from_keywords()
    for seed in args.seed:
        engine.add_seed(seed)

    await assembly.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    asyncio.run(main())
