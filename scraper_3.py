import asyncio
import os
import time
import threading
import random
import io
from collections import deque
from typing import Optional, List

import requests
import websockets
from websockets import WebSocketServerProtocol
from bs4 import BeautifulSoup
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageChops, ImageOps

# ------------------------------------------------------------------------------------
# 1) VHS effect code from your example (OpenCV + Pillow).
# ------------------------------------------------------------------------------------
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
):
    """
    Applies a 'better' VHS effect to a BGR NumPy image using OpenCV.
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(ycrcb)

    # --- LUMA bandwidth (horizontal blur) ---
    if luma_bandwidth > 1:
        kernel_luma = np.ones(luma_bandwidth, dtype=np.float32) / luma_bandwidth
        for row_idx in range(Y.shape[0]):
            Y[row_idx, :] = np.convolve(Y[row_idx, :], kernel_luma, mode='same')

    # --- CHROMA bandwidth (color bleed) ---
    if chroma_bandwidth > 1:
        kernel_chroma = np.ones(chroma_bandwidth, dtype=np.float32) / chroma_bandwidth
        for row_idx in range(Cr.shape[0]):
            Cr[row_idx, :] = np.convolve(Cr[row_idx, :], kernel_chroma, mode='same')
            Cb[row_idx, :] = np.convolve(Cb[row_idx, :], kernel_chroma, mode='same')

    def shift_channel(channel, shift):
        channel_shifted = np.zeros_like(channel)
        if shift >= 0:
            channel_shifted[:, shift:] = channel[:, :-shift]
        else:
            channel_shifted[:, :shift] = channel[:, -shift:]
        return channel_shifted

    # Mild color offset: shift Cr and Cb in opposite directions
    Cr = shift_channel(Cr, shift=2)
    Cb = shift_channel(Cb, shift=-2)

    # --- Ghosting / signal reflections ---
    grad_x = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    grad_x_abs = np.absolute(grad_x)
    ghost = shift_channel(grad_x_abs, ghost_shift_pixels)
    Y = cv2.addWeighted(Y, 1.0, ghost, ghost_strength, 0)

    # --- Noise ---
    noise_luma = np.random.normal(0, luma_noise_level, Y.shape).astype(np.float32)
    noise_cr = np.random.normal(0, chroma_noise_level, Cr.shape).astype(np.float32)
    noise_cb = np.random.normal(0, chroma_noise_level, Cb.shape).astype(np.float32)
    Y += noise_luma
    Cr += noise_cr
    Cb += noise_cb

    # --- Horizontal glitch lines ---
    h, w = Y.shape
    for row in range(h):
        if random.random() < line_glitch_probability:
            glitch_offset = int(line_glitch_strength * (random.random() - 0.5) * w)
            Y[row, :] = np.roll(Y[row, :], glitch_offset)
            Cr[row, :] = np.roll(Cr[row, :], glitch_offset)
            Cb[row, :] = np.roll(Cb[row, :], glitch_offset)

    # Clip & reconvert
    Y = np.clip(Y, 0, 255)
    Cr = np.clip(Cr, 0, 255)
    Cb = np.clip(Cb, 0, 255)

    ycrcb_vhs = cv2.merge([Y, Cr, Cb]).astype(np.uint8)
    out_bgr = cv2.cvtColor(ycrcb_vhs, cv2.COLOR_YCrCb2BGR)
    return out_bgr

def apply_vhs_filter_pil(img: Image.Image) -> Image.Image:
    """
    Converts PIL -> NumPy BGR, applies VHS, converts back -> PIL.
    """
    img_rgb = np.array(img.convert("RGB"))
    frame_bgr = img_rgb[:, :, ::-1]

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
    result_rgb = result_bgr[:, :, ::-1]
    return Image.fromarray(result_rgb, mode="RGB")


# ------------------------------------------------------------------------------------
# 2) Global FIFO queue to store the processed images (as PNG bytes).
#    Use maxlen to avoid unbounded growth.
# ------------------------------------------------------------------------------------
SCRAPED_IMAGES = deque(maxlen=1000)


# ------------------------------------------------------------------------------------
# 3) Scraping logic (Requests + BeautifulSoup).
#    Instead of saving images to disk, we fetch them, apply VHS, resize, store PNG bytes.
# ------------------------------------------------------------------------------------
def scrape_subreddit_images(
        subreddit: str,
        max_pages: int = 3,
        skip_keywords: Optional[List[str]] = None,
        page_delay: float = 2.0,
        image_delay: float = 1.0
):
    """
    Scrape images from old.reddit.com/r/<subreddit>, apply VHS and resizing,
    then store as PNG bytes in SCRAPED_IMAGES deque.
    """
    if skip_keywords is None:
        skip_keywords = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/111.0.0.0 Safari/537.36"
        )
    }
    base_url = "https://old.reddit.com"
    next_page_url = f"{base_url}/r/{subreddit}/"

    pages_scraped = 0

    # We'll keep a small set just to avoid re-downloading the same URL within this run:
    visited_urls = set()

    while pages_scraped < max_pages and next_page_url:
        print(f"[SCRAPER] Fetching page {pages_scraped + 1}: {next_page_url}")
        resp = requests.get(next_page_url, headers=headers)
        if resp.status_code != 200:
            print(f"[SCRAPER] ERROR: Got status {resp.status_code}, stopping.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        img_tags = soup.find_all("img")
        for img_tag in img_tags:
            src = img_tag.get("src", "")
            # Skip empty or data-URI images
            if not src or src.startswith("data:"):
                continue
            # Fix scheme if missing (//)
            if src.startswith("//"):
                src = "https:" + src
            # Skip if the URL contains certain keywords
            low_src = src.lower()
            if any(kw.lower() in low_src for kw in skip_keywords):
                print(f"[SCRAPER] Skipping '{src}' due to keyword filter.")
                continue
            # If not visited
            if src not in visited_urls:
                visited_urls.add(src)
                # Download + Transform
                try:
                    r_img = requests.get(src, timeout=5)
                    if r_img.status_code == 200 and r_img.content:
                        # Convert to Pillow
                        img_bytes = io.BytesIO(r_img.content)
                        pil_img = Image.open(img_bytes).convert("RGB")

                        # Resize to 512x512
                        pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)

                        # Apply VHS effect
                        pil_img = apply_vhs_filter_pil(pil_img)

                        # Convert back to PNG bytes
                        out_buf = io.BytesIO()
                        pil_img.save(out_buf, format="PNG")
                        out_buf.seek(0)

                        # Store in global queue
                        SCRAPED_IMAGES.append(out_buf.read())
                        print(f"[SCRAPER] Scraped & stored image from {src}")
                    else:
                        print(f"[SCRAPER] Could not download {src} (status={r_img.status_code}).")
                except Exception as e:
                    print(f"[SCRAPER] Error downloading {src}: {e}")

                # Sleep between each image
                time.sleep(image_delay)

        # Find next page link
        next_button_span = soup.find("span", class_="next-button")
        if next_button_span:
            next_link = next_button_span.find("a")
            if next_link and next_link.has_attr("href"):
                next_page_url = next_link["href"]
            else:
                break
        else:
            break

        pages_scraped += 1
        time.sleep(page_delay)

    print("[SCRAPER] Finished scraping.")


# ------------------------------------------------------------------------------------
# 4) WebSocket server logic: sends images from SCRAPED_IMAGES queue.
# ------------------------------------------------------------------------------------
WS_HOST = "localhost"
WS_PORT = 5010

SEND_DELAY = 0.5  # seconds between sends (you can adjust as needed)


async def image_sender(websocket: WebSocketServerProtocol):
    """
    Repeatedly send images from SCRAPED_IMAGES to the client.
    If the queue is empty, we wait until it has something.
    """
    print("[WS] Client connected.")
    try:
        while True:
            if SCRAPED_IMAGES:
                # Pop the oldest image from the left and send it
                img_data = SCRAPED_IMAGES.popleft()
                await websocket.send(img_data)
                print("[WS] Sent a scraped image.")
            else:
                print("[WS] No images in queue, waiting...")
                # Sleep a bit to avoid busy-loop
                await asyncio.sleep(1.0)

            await asyncio.sleep(SEND_DELAY)
    except websockets.ConnectionClosed:
        print("[WS] Client disconnected.")
    except Exception as e:
        print("[WS] Unhandled error in image_sender:", e)


async def start_server():
    print(f"[WS] Starting WebSocket server at ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        await asyncio.Future()  # Run forever


# ------------------------------------------------------------------------------------
# 5) Main entry point: we do both the scraping (in a separate thread) and the WS server.
# ------------------------------------------------------------------------------------
def main():
    # 1) Start scraping in a background thread
    def scraping_thread():
        # Example usage:
        # - Scrape the 'cats' subreddit, up to 3 pages, skipping any URL containing "icon"
        scrape_subreddit_images(
            subreddit="worldnews",
            max_pages=3,
            skip_keywords=["icon"],
            page_delay=2.0,
            image_delay=1.0
        )

    thread = threading.Thread(target=scraping_thread, daemon=True)
    thread.start()

    # 2) Start the WebSocket server (async)
    asyncio.run(start_server())


if __name__ == "__main__":
    main()
