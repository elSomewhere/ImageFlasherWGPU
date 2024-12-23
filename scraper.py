import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import os
import time
import re
import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime

# Configure logging
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_file = "reddit_crawler.log"
log_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Default variables
CHROME_OPTIONS = uc.ChromeOptions()
CHROME_OPTIONS.headless = False
CHROME_OPTIONS.add_argument("--disable-search-engine-choice-screen")
MAIN_PAGE_URL = "https://www.reddit.com/r/laundry/"
CUTOFF_DATE = "19.11.2024"  # D-M-Y


class RedditCrawler:
    def __init__(self):
        self.driver = None
        self.timeout = 20
        self.wait = None
        self.folder_path = "./images"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.session = requests.Session()
        self.mount_retry_adapter()
        self.processed_images = set()
        self.cutoff_date = datetime.strptime(
            CUTOFF_DATE, r"%d.%m.%Y"
        )  # Define the cutoff date

    def mount_retry_adapter(self):
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def initialize_driver(self):
        try:
            self.driver = uc.Chrome(use_subprocess=True, options=CHROME_OPTIONS)
            self.wait = WebDriverWait(self.driver, self.timeout)
            logger.info("WebDriver initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def start(self):
        while True:
            try:
                self.initialize_driver()
                self.driver.get(MAIN_PAGE_URL)
                logger.info("Starting the crawling process.")
                while True:
                    try:
                        self.process_visible_images()
                        self.process_gallery_carousels()
                        if not self.scroll_down():
                            logger.info(
                                "Reached the end of the page. Restarting from the top."
                            )
                            break
                    except Exception as e:
                        logger.error(f"Error while processing images or scrolling: {e}")
                        logger.error(traceback.format_exc())
                        break
            except WebDriverException as e:
                logger.error(f"WebDriver exception occurred: {e}")
                logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Unexpected error occurred: {e}")
                logger.error(traceback.format_exc())
            finally:
                self.terminate()

            logger.info("Restarting the crawler after a short delay...")
            time.sleep(60)  # Wait for 1 minute before restarting

    def process_visible_images(self):
        try:

            zoomable_img_elements = self.wait.until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "zoomable-img"))
            )
            logger.info(
                f"{len(zoomable_img_elements)} images found in the current page, starting the download process."
            )
            for zoomable_img_element in zoomable_img_elements:
                try:
                    post_timestamp = zoomable_img_element.find_element(
                        By.XPATH, "./ancestor::shreddit-post"
                    ).get_attribute("created-timestamp")
                    # Convert the timestamp to datetime format
                    post_date = datetime.fromisoformat(post_timestamp.split("+")[0])
                    # Check if the post date is older than the cutoff date
                    if post_date < self.cutoff_date:
                        logger.info(
                            f"Post date {post_date} is older than the cutoff date {self.cutoff_date}. Stopping crawler."
                        )
                        self.terminate()
                        return
                except Exception as e:
                    logger.error(f"Error getting the image post date: {e}")
                try:
                    img_element = zoomable_img_element.find_element(By.TAG_NAME, "img")
                    image_src = img_element.get_attribute("src")
                    if image_src in self.processed_images:
                        continue
                    self.processed_images.add(image_src)
                    if self.is_valid_image(image_src):
                        self.download_image(image_src)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
        except TimeoutException:
            logger.warning("No images found, continuing to scroll...")

    def process_gallery_carousels(self):
        try:
            carousel_elements = self.wait.until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "gallery-carousel"))
            )
            logger.info(
                f"{len(carousel_elements)} gallery carousels found in the current page, processing images."
            )
            for carousel in carousel_elements:
                try:
                    img_elements = carousel.find_elements(By.TAG_NAME, "img")
                    for img in img_elements:
                        image_src = img.get_attribute("src")
                        if image_src and image_src not in self.processed_images:
                            self.processed_images.add(image_src)
                            image_id = self.extract_image_id(image_src)
                            if image_id:
                                full_res_url = f"https://i.redd.it/{image_id}"
                                self.download_image(full_res_url)
                except Exception as e:
                    logger.error(f"Error processing carousel image: {e}")
        except TimeoutException:
            logger.warning("No gallery carousels found, continuing to scroll...")

    def extract_image_id(self, url):
        match = re.search(r"v0-([^?]+)", url)
        return match.group(1) if match else None

    def is_valid_image(self, src):
        try:
            return not src.startswith("https://external-preview.redd.it")
        except Exception as e:
            logger.error(f"Error checking image validity: {e}")
            return False

    def download_image(self, image_url):
        filename = os.path.basename(image_url)
        image_path = os.path.join(self.folder_path, filename)

        if not os.path.exists(image_path):
            try:
                response = self.session.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(image_path, "wb") as image_file:
                        image_file.write(response.content)
                    logger.info(f"Downloaded: {filename}")
                else:
                    logger.warning(f"Failed to download: {filename}")
            except requests.RequestException as e:
                logger.error(f"Error downloading {filename}: {e}")
        else:
            logger.info(f"Image already exists: {filename}, skipping.")
        time.sleep(2)  # to limit the requests per minute

    def scroll_down(self, retries=5):
        attempt = 0
        while attempt < retries:
            try:
                last_height = self.driver.execute_script(
                    "return document.body.scrollHeight"
                )
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(5)
                new_height = self.driver.execute_script(
                    "return document.body.scrollHeight"
                )

                if new_height > last_height:
                    return True  # Scrolling succeeded, exit the loop
                else:
                    logger.info(f"Scroll attempt {attempt + 1} failed, retrying...")
                    attempt += 1

            except Exception as e:
                logger.error(f"Error scrolling: {e}")
                attempt += 1

        logger.warning(f"Failed to scroll after {retries} attempts.")
        return False  # Return False after all retries are exhausted

    def terminate(self):
        logger.info("Terminating the current session.")
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error while quitting the driver: {e}")
        self.driver = None
        self.wait = None


def main():
    logger.info("-------Starting the Reddit Crawler--------")
    crawler = RedditCrawler()
    crawler.start()


if __name__ == "__main__":
    main()