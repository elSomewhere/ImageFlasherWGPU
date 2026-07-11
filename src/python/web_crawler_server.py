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
    parser.add_argument(
        "--no-compliance",
        action="store_true",
        help="Drop the robots/SSRF/rate-limit layers (for a non-internet/generative world).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    profile = replace(
        Profile(),
        image_host=args.image_host,
        image_port=args.image_port,
        control_host=args.control_host,
        control_port=args.control_port,
        compliance=not args.no_compliance,
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
