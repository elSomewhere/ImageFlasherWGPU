import os
import time
import requests
import urllib.request
from bs4 import BeautifulSoup
from collections import deque

def scrape_subreddit_images(
        subreddit: str,
        download_folder: str = "images",
        max_pages: int = 3,
        page_delay: float = 2.0,
        image_delay: float = 1.0,
        history_size: int = 1000,
        skip_keywords=None
):
    """
    Scrape images from old.reddit.com/r/<subreddit> by following
    the "next" link until we reach max_pages or no more pages.
    Also keeps track of recently downloaded images in a fixed-size FIFO queue.

    :param subreddit: Subreddit name, e.g., 'cats'
    :param download_folder: Folder where images will be saved
    :param max_pages: Maximum number of pages to scrape
    :param page_delay: Seconds to wait between each page fetch
    :param image_delay: Seconds to wait between each image download
    :param history_size: Maximum number of previously downloaded image URLs to store
    :param skip_keywords: A list of keywords to skip if found in the image URL (case-insensitive).
    """

    if skip_keywords is None:
        skip_keywords = []

    # Ensure the download folder exists
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Basic headers to mimic a “normal” browser request
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/111.0.0.0 Safari/537.36"
        )
    }

    # Start with the main subreddit URL (old Reddit)
    base_url = "https://old.reddit.com"
    next_page_url = f"{base_url}/r/{subreddit}/"

    pages_scraped = 0

    # Fixed-size FIFO queue for downloaded URLs
    downloaded_urls = deque(maxlen=history_size)

    while pages_scraped < max_pages and next_page_url:
        print(f"[INFO] Fetching page {pages_scraped + 1}: {next_page_url}")
        response = requests.get(next_page_url, headers=headers)
        if response.status_code != 200:
            print(f"[ERROR] Failed to retrieve {next_page_url} (status={response.status_code}). Stopping.")
            break

        # Use BeautifulSoup to parse the page
        soup = BeautifulSoup(response.text, "html.parser")

        # 1) Find all img tags on the page
        img_tags = soup.find_all("img")
        for img_tag in img_tags:
            src = img_tag.get("src", "")
            # Skip empty or data-URI images
            if not src or src.startswith("data:"):
                continue

            # Fix scheme if it's missing (i.e., //external-preview...)
            if src.startswith("//"):
                src = "https:" + src

            # --- NEW KEYWORD FILTER ---
            # If any of the skip keywords appear in the image URL, skip downloading.
            lower_src = src.lower()
            if any(keyword.lower() in lower_src for keyword in skip_keywords):
                print(f"[SKIP] {src} contains a skip keyword.")
                continue

            # If we've never downloaded this URL, attempt to download
            if src not in downloaded_urls:
                downloaded_urls.append(src)
                download_image(src, download_folder)
                time.sleep(image_delay)

        # 2) Find the "next" button link
        next_button_span = soup.find("span", class_="next-button")
        if next_button_span:
            next_link = next_button_span.find("a")
            if next_link and next_link.has_attr("href"):
                next_page_url = next_link["href"]
            else:
                print("[INFO] No valid 'next' link found. Stopping.")
                break
        else:
            print("[INFO] No 'next-button' found. Stopping.")
            break

        pages_scraped += 1
        # Respectful delay to avoid spamming the server
        time.sleep(page_delay)

    print("[INFO] Finished scraping.")


def download_image(url: str, folder: str):
    """
    Given an image URL, download it to the specified folder.
    """
    # Derive a filename from the URL
    filename = url.split("/")[-1].split("?")[0]  # basic filename from the URL
    filepath = os.path.join(folder, filename)

    try:
        if not os.path.exists(filepath):
            print(f"[DOWNLOAD] {url} -> {filepath}")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"[SKIP] Already downloaded: {url}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url} - {e}")


if __name__ == "__main__":
    # Example usage:
    subreddit = "cats"  # e.g. "cats", "pics", "wallpaper"
    # We'll skip images where the URL has "icon" in it
    scrape_subreddit_images(
        subreddit=subreddit,
        max_pages=3,
        history_size=1000,
        skip_keywords=["icon"]
    )
