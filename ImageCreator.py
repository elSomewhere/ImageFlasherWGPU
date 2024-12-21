#!/usr/bin/env python3
import asyncio
import os
import io
import random
from typing import Optional, List, Tuple

import aiohttp
import websockets
import feedparser
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageOps

################################################################################
# 1) The "better" VHS effect from your original code, but adapted for PIL images
################################################################################

def apply_better_vhs_effect(
        frame: np.ndarray,
        luma_bandwidth=5,
        chroma_bandwidth=8,
        luma_noise_level=10.0,
        chroma_noise_level=20.0,
        ghost_shift_pixels=3,
        ghost_strength=0.3,
        line_glitch_probability=0.005,
        line_glitch_strength=0.8
) -> np.ndarray:
    """
    Applies a 'better' VHS effect by simulating bandwidth limitation, color bleed,
    ghosting, noise, and random glitch lines on an image (in BGR NumPy array form).
    Returns a BGR NumPy array with artifacts.
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(ycrcb)

    # -------------------------------------------------------------------
    # LUMA PROCESSING: simulate limited bandwidth by horizontal blur
    # -------------------------------------------------------------------
    if luma_bandwidth > 1:
        kernel_luma = np.ones(luma_bandwidth, dtype=np.float32) / luma_bandwidth
        for row_idx in range(Y.shape[0]):
            Y[row_idx, :] = np.convolve(Y[row_idx, :], kernel_luma, mode='same')

    # -------------------------------------------------------------------
    # CHROMA PROCESSING: color bleed, reduce bandwidth, offset
    # -------------------------------------------------------------------
    if chroma_bandwidth > 1:
        kernel_chroma = np.ones(chroma_bandwidth, dtype=np.float32) / chroma_bandwidth
        for row_idx in range(Cr.shape[0]):
            Cr[row_idx, :] = np.convolve(Cr[row_idx, :], kernel_chroma, mode='same')
            Cb[row_idx, :] = np.convolve(Cb[row_idx, :], kernel_chroma, mode='same')

    def shift_channel(channel, shift):
        """Shift channel horizontally by 'shift' pixels."""
        channel_shifted = np.zeros_like(channel)
        if shift >= 0:
            channel_shifted[:, shift:] = channel[:, :-shift]
        else:
            channel_shifted[:, :shift] = channel[:, -shift:]
        return channel_shifted

    # Mild color offset: shift Cr and Cb in opposite directions
    Cr = shift_channel(Cr, shift=2)
    Cb = shift_channel(Cb, shift=-2)

    # -------------------------------------------------------------------
    # SIGNAL REFLECTIONS / ghosting
    # -------------------------------------------------------------------
    grad_x = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    grad_x_abs = np.absolute(grad_x)
    ghost = shift_channel(grad_x_abs, ghost_shift_pixels)
    Y = cv2.addWeighted(Y, 1.0, ghost, ghost_strength, 0)

    # -------------------------------------------------------------------
    # NOISE & GLITCH
    # -------------------------------------------------------------------
    # Random noise
    noise_luma = np.random.normal(0, luma_noise_level, Y.shape).astype(np.float32)
    noise_cr = np.random.normal(0, chroma_noise_level, Cr.shape).astype(np.float32)
    noise_cb = np.random.normal(0, chroma_noise_level, Cb.shape).astype(np.float32)
    Y += noise_luma
    Cr += noise_cr
    Cb += noise_cb

    # Random horizontal glitch lines
    h, w = Y.shape
    for row in range(h):
        if random.random() < line_glitch_probability:
            glitch_offset = int(line_glitch_strength * (random.random() - 0.5) * w)
            Y[row, :] = np.roll(Y[row, :], glitch_offset)
            Cr[row, :] = np.roll(Cr[row, :], glitch_offset)
            Cb[row, :] = np.roll(Cb[row, :], glitch_offset)

    # Clip
    Y = np.clip(Y, 0, 255)
    Cr = np.clip(Cr, 0, 255)
    Cb = np.clip(Cb, 0, 255)

    # Recombine YCrCb -> BGR
    ycrcb_vhs = cv2.merge([Y, Cr, Cb]).astype(np.uint8)
    out_bgr = cv2.cvtColor(ycrcb_vhs, cv2.COLOR_YCrCb2BGR)
    return out_bgr

def apply_vhs_filter_pil(img: Image.Image) -> Image.Image:
    """
    Takes a PIL Image, converts it to a NumPy BGR array, applies the advanced VHS effect,
    and returns a new PIL Image in RGB.
    """
    # Convert PIL -> NumPy (RGB)
    img_rgb = np.array(img.convert("RGB"))
    # Convert RGB -> BGR for OpenCV
    frame_bgr = img_rgb[:, :, ::-1]

    # Apply the advanced VHS effect
    result_bgr = apply_better_vhs_effect(
        frame_bgr,
        luma_bandwidth=5,
        chroma_bandwidth=8,
        luma_noise_level=10.0,
        chroma_noise_level=20.0,
        ghost_shift_pixels=3,
        ghost_strength=0.3,
        line_glitch_probability=0.005,
        line_glitch_strength=0.8
    )

    # Convert back BGR -> RGB -> PIL
    result_rgb = result_bgr[:, :, ::-1]
    result_pil = Image.fromarray(result_rgb, mode="RGB")
    return result_pil


################################################################################
# 2) RSS-based image retrieval logic
################################################################################

import feedparser

import aiohttp
import websockets

# WebSocket server config
WS_HOST = "localhost"
WS_PORT = 5010

# Wait time between checks for each feed (in seconds)
PER_FEED_DELAY = 5

# The images will be resized to the same final size as your main app expects
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# A set of RSS feeds known for images
RSS_FEEDS = [
    # Example feeds
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reddit r/pics", "https://www.reddit.com/r/pics/.rss"),
    # ... more feed entries ...
]


def extract_image_url_from_entry(entry) -> Optional[str]:
    """
    Parse the feed entry to find the first plausible image URL.
    """
    # Check media_content
    media_content = entry.get('media_content', [])
    for mc in media_content:
        if 'url' in mc:
            return mc['url']

    # Check media_thumbnail
    media_thumbnail = entry.get('media_thumbnail', [])
    for mt in media_thumbnail:
        if 'url' in mt:
            return mt['url']

    # Check enclosure links
    for link in entry.get('links', []):
        if link.get('rel') == 'enclosure':
            href = link.get('href', '')
            typ = link.get('type', '')
            if ('image' in typ) or href.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                return href
    return None

async def fetch_feed(session: aiohttp.ClientSession, url: str) -> Optional[feedparser.FeedParserDict]:
    """
    Download the RSS feed data and parse it with feedparser.
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Error: Failed to fetch {url} HTTP {resp.status}")
                return None
            data = await resp.read()
            feed = feedparser.parse(data)
            if feed.bozo:
                print(f"Warning: Feed parse error for {url}: {feed.bozo_exception}")
                return None
            return feed
    except Exception as e:
        print(f"Error fetching feed {url}: {e}")
        return None

def is_newer(published: str, last_published: Optional[str]) -> bool:
    if last_published is None:
        return True
    return published > last_published

async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """
    Downloads the image from `url`, resizes it, applies advanced VHS effect,
    and returns PNG bytes.
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download image: HTTP {resp.status} from {url}")
                return None
            data = await resp.read()

            # Open via Pillow
            img = Image.open(io.BytesIO(data))

            # Convert to RGB
            img = img.convert("RGB")

            # Resize
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

            # Apply the advanced VHS effect
            img = apply_vhs_filter_pil(img)

            # Convert back to PNG bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
    except Exception as e:
        print(f"Error downloading/resizing image {url}: {e}")
        return None

async def fetch_new_images(session: aiohttp.ClientSession, feed_name: str, feed_url: str, last_published: Optional[str]) -> Tuple[List[bytes], Optional[str]]:
    feed = await fetch_feed(session, feed_url)
    if not feed or not hasattr(feed, 'entries'):
        return [], last_published

    entries = feed.entries
    # Sort them by publication time (the logic may differ if date formats vary)
    entries = sorted(entries, key=lambda e: e.get('published',''))

    new_images = []
    updated_last_published = last_published

    for entry in entries:
        published = entry.get('published', '')
        if not is_newer(published, last_published):
            continue

        img_url = extract_image_url_from_entry(entry)
        if img_url:
            print(f"[{feed_name}] Found new image: {img_url}")
            img_data = await fetch_image(session, img_url)
            if img_data:
                new_images.append(img_data)
                if updated_last_published is None or published > updated_last_published:
                    updated_last_published = published

    return new_images, updated_last_published

async def run_feed_task(session: aiohttp.ClientSession, websocket, feed_name: str, feed_url: str):
    """
    A task that runs continuously for a single feed:
    - Keeps track of the last published time.
    - Fetches new entries, sends images when found.
    - Sleeps a short delay between attempts.
    """
    last_published = None
    while True:
        try:
            images, updated_pub = await fetch_new_images(session, feed_name, feed_url, last_published)
            last_published = updated_pub
            if images:
                for img in images:
                    print(f"Sending image from {feed_name}")
                    await websocket.send(img)
                    print("Image sent")
                    await asyncio.sleep(0.05)  # small pause after sending
        except websockets.ConnectionClosed:
            print(f"Client disconnected. Stopping feed {feed_name}.")
            break
        except Exception as e:
            print(f"Unhandled error with {feed_name}: {e}")

        # Sleep before next check
        await asyncio.sleep(PER_FEED_DELAY)

async def image_sender(websocket):
    print("WebSocket client connected, starting parallel feed scanning.")
    async with aiohttp.ClientSession() as session:
        # Create a task per feed
        tasks = []
        for feed_name, feed_url in RSS_FEEDS:
            task = asyncio.create_task(run_feed_task(session, websocket, feed_name, feed_url))
            tasks.append(task)

        # Wait until the websocket closes or tasks fail
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # If websocket closes or error hits, cancel all tasks
        for t in pending:
            t.cancel()

async def main():
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
