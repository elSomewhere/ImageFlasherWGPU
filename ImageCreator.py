import asyncio
import os
import io
import time
from typing import Optional
import aiohttp
import websockets
from PIL import Image
import feedparser

WS_HOST = "localhost"
WS_PORT = 5000
CHECK_INTERVAL = 60  # seconds between checks to avoid hitting servers too frequently
RSS_FEED_URL = "http://feeds.bbci.co.uk/news/world/rss.xml"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

async def fetch_image(session: aiohttp.ClientSession, url: str):
    """
    Download the image from the given URL and resize to 512x512 PNG.
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download image: HTTP {resp.status}")
                return None
            data = await resp.read()
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGBA")
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
    except Exception as e:
        print("Error downloading/resizing image:", e)
        return None

def extract_image_url_from_entry(entry):
    """
    Extracts the first available image URL from a feed entry. Different feeds may have different
    ways of embedding images. For BBC, often <media:thumbnail> or <media:content> is used.
    We'll check both and return the first one we find.
    """
    # media_content can be a list of dicts with 'url' keys
    media_content = entry.get('media_content', [])
    if media_content:
        for mc in media_content:
            if 'url' in mc:
                return mc['url']

    # media_thumbnail can also hold images
    media_thumbnail = entry.get('media_thumbnail', [])
    if media_thumbnail:
        for mt in media_thumbnail:
            if 'url' in mt:
                return mt['url']

    # Some feeds might have 'links' with rel='enclosure' as images
    for link in entry.get('links', []):
        if link.get('rel') == 'enclosure' and 'image' in link.get('type', ''):
            return link.get('href')

    return None

async def fetch_new_images(last_published: Optional[str]) -> list:
    """
    Fetch the RSS feed and return a list of image URLs from new items.
    We'll track 'last_published' to only return newer items each time.
    """
    feed = feedparser.parse(RSS_FEED_URL)
    if feed.bozo:
        print("Error parsing feed:", feed.bozo_exception)
        return []

    entries = feed.entries
    # Sort entries by published time if available
    # Not all feeds provide a 'published_parsed' or 'updated_parsed'.
    # BBC feeds have 'published_parsed'. We'll use it.
    entries = sorted(entries, key=lambda e: e.get('published_parsed', time.gmtime(0)))

    new_images = []
    latest_published = last_published

    for entry in entries:
        # published is a string like 'Thu, 05 Oct 2023 10:00:00 GMT'
        published = entry.get('published', '')
        if last_published and published <= last_published:
            continue

        img_url = extract_image_url_from_entry(entry)
        if img_url:
            new_images.append((published, img_url))
            # Track the latest published time
            if not latest_published or published > latest_published:
                latest_published = published

    return new_images, latest_published

async def image_sender(websocket):
    """
    Continuously fetch images from the news RSS feed and send them to the client.
    This simulates real world news flow by using a live feed.
    """
    print("WebSocket client connected, starting news image fetching.")
    last_published = None
    no_new_count = 0

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                new_images, last_published = await fetch_new_images(last_published)
                if new_images:
                    no_new_count = 0
                    for published, img_url in new_images:
                        img_data = await fetch_image(session, img_url)
                        if img_data:
                            print(f"Sending image from: {img_url}")
                            await websocket.send(img_data)
                            print("Image sent")
                            await asyncio.sleep(0.01)
                else:
                    no_new_count += 1
                    if no_new_count > 5:
                        print("No new images found for a while, sleeping longer.")
                        # Sleep longer before next check
                        await asyncio.sleep(CHECK_INTERVAL * 5)
                    else:
                        print("No new images found, will check again after interval.")

            except websockets.ConnectionClosed:
                print("Client disconnected.")
                break
            except aiohttp.ClientError as ce:
                print("Network error fetching images:", ce)
                await asyncio.sleep(CHECK_INTERVAL * 5)
            except Exception as e:
                print("Unhandled error:", e)
                await asyncio.sleep(CHECK_INTERVAL * 5)

async def main():
    """
    Start the WebSocket server that sends images to the client.
    """
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        print(f"WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
