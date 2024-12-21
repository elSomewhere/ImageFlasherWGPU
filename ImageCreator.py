import asyncio
import os
import io
from typing import Optional, List, Tuple
import aiohttp
import websockets
from PIL import Image
import feedparser
import random
import numpy as np
from PIL import Image, ImageChops, ImageDraw

def apply_vhs_filter(img: Image.Image) -> Image.Image:
    """
    Applies a rudimentary 'VHS/bad TV' style effect:
    - Slight channel offset (color bleed)
    - Horizontal scan lines
    - Random noise
    """
    # Ensure RGBA
    img = img.convert("RGBA")

    # --- 1) Color channel offset ---
    # Let's shift the red channel a few pixels to the right.
    # We'll do this by splitting channels and recombining.
    r, g, b, a = img.split()

    # Shift the red channel horizontally by e.g. 3 pixels
    # We'll use ImageChops.offset
    r_shifted = ImageChops.offset(r, xoffset=3, yoffset=0)
    # You can also shift the blue or green in another direction for more effect
    b_shifted = ImageChops.offset(b, xoffset=-1, yoffset=0)

    # Recombine the channels (r_shifted, g, b_shifted)
    img = Image.merge("RGBA", (r_shifted, g, b_shifted, a))

    # --- 2) Horizontal scan lines ---
    # We'll darken every other row, for example.
    scanline_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(scanline_overlay)

    # For every other row, draw a slightly translucent dark line
    width, height = img.size
    for y in range(0, height, 2):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, 50))

    # Combine overlay with the image
    img = Image.alpha_composite(img, scanline_overlay)

    # --- 3) Random noise (static) ---
    # We'll add random speckles across the image.
    # Create a random grayscale noise image, then alpha-blend it slightly.
    noise = Image.effect_noise(img.size, 64)  # 64 is noise severity
    noise = noise.convert("RGBA")

    # We want to lighten/darken randomly. Let's colorize it somewhat
    # so it's not purely grayscale noise. Or keep it as is for static.
    # For demonstration, keep it grayscale, but we give it partial alpha:
    alpha_mask = Image.new("L", img.size, 64)  # 64 out of 255 => ~25% visible
    noise.putalpha(alpha_mask)

    # Overlay the noise on top
    img = Image.alpha_composite(img, noise)

    return img


# WebSocket server config
WS_HOST = "localhost"
WS_PORT = 5000

# Wait time between checks for each feed (in seconds)
PER_FEED_DELAY = 5

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# A large set of RSS feeds
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

    # NASA, Photography, etc.
    ("NASA Image of the Day", "https://www.nasa.gov/rss/dyn/lg_image_of_the_day.rss"),
    ("PetaPixel", "https://feeds.feedburner.com/Petapixel"),
    ("Flickr Interesting Photos", "https://www.flickr.com/services/feeds/photos_public.gne?tags=interesting&format=rss2"),
    ("National Geographic Daily Photo", "https://rsshub.app/natgeo/dailyphoto"),
    ("NASA Twitter via Nitter", "https://nitter.net/NASA/rss"),
    ("BBC News in Video", "http://feeds.bbci.co.uk/news/video_and_audio/news_front_page/rss.xml"),
    ("Euronews Videos", "http://feeds.feedburner.com/euronews/en/videos"),

    # Image-heavy from Reddit (existing)
    ("Reddit r/pics", "https://www.reddit.com/r/pics/.rss"),
    ("Reddit r/EarthPorn", "https://www.reddit.com/r/EarthPorn/.rss"),
    ("Reddit r/Photojournalism", "https://www.reddit.com/r/Photojournalism/.rss"),

    # Add a lot more Reddit subs known for images:
    ("Reddit r/aww", "https://www.reddit.com/r/aww/.rss"),
    ("Reddit r/Art", "https://www.reddit.com/r/Art/.rss"),
    ("Reddit r/photoshopbattles", "https://www.reddit.com/r/photoshopbattles/.rss"),
    ("Reddit r/HistoryPorn", "https://www.reddit.com/r/HistoryPorn/.rss"),
    ("Reddit r/SpacePorn", "https://www.reddit.com/r/SpacePorn/.rss"),
    ("Reddit r/CarPorn", "https://www.reddit.com/r/CarPorn/.rss"),
    ("Reddit r/FoodPorn", "https://www.reddit.com/r/FoodPorn/.rss"),
    ("Reddit r/AlbumArtPorn", "https://www.reddit.com/r/AlbumArtPorn/.rss"),
    ("Reddit r/ArchitecturePorn", "https://www.reddit.com/r/ArchitecturePorn/.rss"),
    ("Reddit r/AdorablePictures", "https://www.reddit.com/r/AdorablePictures/.rss"),
    ("Reddit r/AnimalPorn", "https://www.reddit.com/r/AnimalPorn/.rss"),
    ("Reddit r/WaterPorn", "https://www.reddit.com/r/WaterPorn/.rss"),
    ("Reddit r/WinterPorn", "https://www.reddit.com/r/WinterPorn/.rss"),
    ("Reddit r/CityPorn", "https://www.reddit.com/r/CityPorn/.rss"),
    ("Reddit r/DesertPorn", "https://www.reddit.com/r/DesertPorn/.rss"),
    ("Reddit r/LakePorn", "https://www.reddit.com/r/LakePorn/.rss"),
    ("Reddit r/ImaginaryLandscapes", "https://www.reddit.com/r/ImaginaryLandscapes/.rss"),
    ("Reddit r/ImaginaryMaps", "https://www.reddit.com/r/ImaginaryMaps/.rss"),
    ("Reddit r/SkyPorn", "https://www.reddit.com/r/SkyPorn/.rss"),
    ("Reddit r/WeatherPorn", "https://www.reddit.com/r/WeatherPorn/.rss"),
    ("Reddit r/EyeCandy", "https://www.reddit.com/r/EyeCandy/.rss"),
    ("Reddit r/ExposurePorn", "https://www.reddit.com/r/ExposurePorn/.rss"),
    ("Reddit r/PerfectTiming", "https://www.reddit.com/r/PerfectTiming/.rss"),
    ("Reddit r/RuralPorn", "https://www.reddit.com/r/RuralPorn/.rss"),
    ("Reddit r/VillagePorn", "https://www.reddit.com/r/VillagePorn/.rss"),
    ("Reddit r/MuseumPorn", "https://www.reddit.com/r/MuseumPorn/.rss"),
    ("Reddit r/CastlePorn", "https://www.reddit.com/r/CastlePorn/.rss"),
    ("Reddit r/BoatPorn", "https://www.reddit.com/r/BoatPorn/.rss"),
    ("Reddit r/MountainPorn", "https://www.reddit.com/r/MountainPorn/.rss"),
    ("Reddit r/JunglePorn", "https://www.reddit.com/r/JunglePorn/.rss"),
    ("Reddit r/AutumnPorn", "https://www.reddit.com/r/AutumnPorn/.rss"),
    ("Reddit r/SpringPorn", "https://www.reddit.com/r/SpringPorn/.rss"),
    ("Reddit r/FallPorn", "https://www.reddit.com/r/FallPorn/.rss"),
    ("Reddit r/MacroPorn", "https://www.reddit.com/r/MacroPorn/.rss"),
    ("Reddit r/MiniaturePorn", "https://www.reddit.com/r/MiniaturePorn/.rss"),
    ("Reddit r/UnderwaterPhotography", "https://www.reddit.com/r/UnderwaterPhotography/.rss"),
    ("Reddit r/RoadPorn", "https://www.reddit.com/r/RoadPorn/.rss"),
    ("Reddit r/SculpturePorn", "https://www.reddit.com/r/SculpturePorn/.rss"),
    ("Reddit r/MachinePorn", "https://www.reddit.com/r/MachinePorn/.rss"),
    ("Reddit r/InfrastructurePorn", "https://www.reddit.com/r/InfrastructurePorn/.rss")
]


def extract_image_url_from_entry(entry):
    # Extract first available image URL
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
    return published > last_published

async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to download image: HTTP {resp.status} from {url}")
                return None
            data = await resp.read()

            # Open via Pillow
            img = Image.open(io.BytesIO(data))

            # Convert to RGBA (if not already)
            img = img.convert("RGBA")

            # Resize to your target size
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

            # ----- APPLY VHS FILTER HERE -----
            img = apply_vhs_filter(img)
            # ----------------------------------

            # Finally, convert back to bytes (PNG)
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
                    await asyncio.sleep(0.01)  # small pause after sending
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

        # Wait until the websocket closes
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        # If websocket closes, cancel all tasks
        for t in pending:
            t.cancel()

async def main():
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
