import asyncio
import os
import io
import time
from typing import Optional, List, Tuple
import aiohttp
import websockets
from PIL import Image
import feedparser

# WebSocket server config
WS_HOST = "localhost"
WS_PORT = 5000

# Wait time between processing each feed (in seconds)
# Lowering this means you check feeds more frequently.
PER_FEED_DELAY = 5

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# A large, comprehensive list of RSS feeds from various sources
# Some may not always have images, but we try to maximize chances.
RSS_FEEDS = [
    # Major World News
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC In Pictures", "http://feeds.bbci.co.uk/news/in_pictures/rss.xml"),
    ("Reuters World", "http://feeds.reuters.com/Reuters/worldNews"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    ("DW World", "https://rss.dw.com/rdf/rss-en-world"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ("CNN Top Stories", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    ("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ("ABC News International", "https://abcnews.go.com/abcnews/internationalheadlines"),
    ("France24 World", "http://www.france24.com/en/world/rss"),
    ("Euronews", "http://feeds.feedburner.com/euronews/en/news"),
    ("RFI English", "http://en.rfi.fr/general/rss"),
    ("AP World", "https://feeds.apnews.com/apf-intlnews"),
    ("Japan Times World", "https://www.japantimes.co.jp/news_category/world/feed/"),
    ("CBC World", "https://www.cbc.ca/cmlink/rss-world"),

    # Reddit (image-heavy subreddits)
    ("Reddit r/pics", "https://www.reddit.com/r/pics/.rss"),
    ("Reddit r/EarthPorn", "https://www.reddit.com/r/EarthPorn/.rss"),
    ("Reddit r/Photojournalism", "https://www.reddit.com/r/Photojournalism/.rss"),

    # NASA and Astronomy
    ("NASA Image of the Day", "https://www.nasa.gov/rss/dyn/lg_image_of_the_day.rss"),

    # Photography News
    ("PetaPixel", "https://feeds.feedburner.com/Petapixel"),

    # Flickr Public Photos
    ("Flickr Interesting Photos", "https://www.flickr.com/services/feeds/photos_public.gne?tags=interesting&format=rss2"),

    # National Geographic Daily Photo (via RSSHub)
    ("National Geographic Daily Photo", "https://rsshub.app/natgeo/dailyphoto"),

    # Twitter via Nitter (if accessible)
    ("NASA Twitter via Nitter", "https://nitter.net/NASA/rss"),

    # Additional feeds for variety
    ("BBC News in Video", "http://feeds.bbci.co.uk/news/video_and_audio/news_front_page/rss.xml"),
    ("Euronews Videos", "http://feeds.feedburner.com/euronews/en/videos")
]

def extract_image_url_from_entry(entry):
    """
    Extract the first available image URL from a feed entry.
    We'll check media_content, media_thumbnail, and enclosure links.
    """
    media_content = entry.get('media_content', [])
    for mc in media_content:
        if 'url' in mc:
            return mc['url']

    media_thumbnail = entry.get('media_thumbnail', [])
    for mt in media_thumbnail:
        if 'url' in mt:
            return mt['url']

    for link in entry.get('links', []):
        if link.get('rel') == 'enclosure':
            href = link.get('href', '')
            typ = link.get('type', '')
            if 'image' in typ or href.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                return href
    return None

async def fetch_feed(session: aiohttp.ClientSession, url: str) -> Optional[feedparser.FeedParserDict]:
    """
    Fetch the RSS feed asynchronously and parse with feedparser from memory.
    Returns the parsed feed or None if failed.
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Error: Failed to fetch {url} HTTP {resp.status}")
                return None
            data = await resp.read()
            feed = feedparser.parse(data)
            if feed.bozo:
                print(f"Warning: Feed {url} parse error: {feed.bozo_exception}")
                return None
            return feed
    except Exception as e:
        print(f"Error fetching feed {url}: {e}")
        return None

def is_newer(published: str, last_published: Optional[str]) -> bool:
    if last_published is None:
        return True
    # Simple lexicographical comparison might not always be perfect,
    # but often RSS feeds have consistently formatted dates.
    return published > last_published

async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """
    Download and resize image to PNG 512x512.
    """
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download image: HTTP {resp.status} from {url}")
                return None
            data = await resp.read()
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGBA")
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
    except Exception as e:
        print(f"Error downloading/resizing image {url}: {e}")
        return None

async def fetch_new_images(session: aiohttp.ClientSession, feed_name: str, feed_url: str, last_published: Optional[str]) -> Tuple[List[bytes], Optional[str]]:
    """
    Fetch and return new images from the given feed since last_published time.
    Updates the last_published if newer content is found.
    """
    feed = await fetch_feed(session, feed_url)
    if not feed or not hasattr(feed, 'entries'):
        return [], last_published

    entries = feed.entries
    # Sort by published for consistency
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

async def image_sender(websocket):
    print("WebSocket client connected, starting continuous feed scanning.")
    # Track last published time per feed
    feed_states = {(name, url): None for (name, url) in RSS_FEEDS}

    async with aiohttp.ClientSession() as session:
        # We'll loop through feeds continuously in a round-robin fashion.
        # After checking one feed, we wait a small delay and then move to the next.
        # This creates a smooth, constant scanning pattern.
        feed_list = list(feed_states.keys())
        idx = 0
        while True:
            (feed_name, feed_url) = feed_list[idx]
            last_pub = feed_states[(feed_name, feed_url)]

            try:
                images, updated_pub = await fetch_new_images(session, feed_name, feed_url, last_pub)
                feed_states[(feed_name, feed_url)] = updated_pub
                if images:
                    for img in images:
                        print(f"Sending image from {feed_name}")
                        await websocket.send(img)
                        print("Image sent")
                        await asyncio.sleep(0.01)  # tiny pause after sending
            except websockets.ConnectionClosed:
                print("Client disconnected.")
                break
            except Exception as e:
                print(f"Unhandled error with {feed_name}: {e}")

            # Move to next feed
            idx = (idx + 1) % len(feed_list)

            # Short delay between feeds to avoid all-at-once bursts
            await asyncio.sleep(PER_FEED_DELAY)

async def main():
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
