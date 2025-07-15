import time
import pyperclip

# Instead of "from selenium import webdriver", import undetected_chromedriver
import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def scroll_down(driver, pause_seconds=2, scroll_attempts=5):
    """
    Scrolls down the page a specified number of times (scroll_attempts).
    Waits pause_seconds between each scroll, allowing new content to load.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")

    for _ in range(scroll_attempts):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_seconds)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # No more new content to load
            break
        last_height = new_height


def click_hamburger_copy_prompt(driver):
    """
    Clicks:
      - The 'hamburger' menu,
      - The 'Copy' button,
      - Then the 'Prompt' button,
    Then reads the text from the system clipboard via pyperclip.
    Returns the text found in the clipboard.
    """
    try:
        # 1) Hamburger button
        hamburger_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@title='Open Options']"))
        )
        hamburger_button.click()

        # 2) "Copy" button
        copy_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((
                By.XPATH,
                ".//button[.//span[contains(text(), 'Copy')]]"
            ))
        )
        copy_button.click()
        time.sleep(0.5)  # small pause

        # 3) "Prompt" button
        prompt_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((
                By.XPATH,
                ".//button[.//span[contains(text(), 'Prompt')]]"
            ))
        )
        prompt_button.click()
        time.sleep(0.5)  # allow some time for the clipboard to update

        # Now read from the system clipboard
        prompt_text = pyperclip.paste()
        return prompt_text

    except (TimeoutException, NoSuchElementException):
        print("Could not interact with hamburger menu or Copy/Prompt buttons.")
        return None


def close_lightbox(driver):
    """
    Clicks the Close (X) button to hide the lightbox.
    """
    try:
        close_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@title='Close Lightbox']"))
        )
        close_btn.click()
        time.sleep(0.5)
    except (TimeoutException, NoSuchElementException):
        print("Could not close the lightbox.")


def main():
    # 1) Setup undetected-chromedriver
    options = uc.ChromeOptions()

    # If you want a visible browser:
    # options.add_argument("--start-maximized")

    # Optionally, you can try headless (though many sites may detect it):
    # options.add_argument("--headless=new")

    options.add_argument("user-data-dir=/Users/estebanlanter/Library/Application Support/Google/Chrome/Default")

    # Create the undetected-chromedriver instance
    driver = uc.Chrome(options=options)

    try:
        # 2) Navigate to Midjourney's archive
        driver.get("https://www.midjourney.com/archive")
        time.sleep(60)  # Give yourself time to log in manually if needed

        # 3) Scroll down to load more images
        scroll_down(driver, pause_seconds=2, scroll_attempts=5)

        # 4) Get all thumbnail <a> elements
        # The CSS classes might change over time; this is just an example
        thumbnail_links = driver.find_elements(
            By.CSS_SELECTOR,
            "a.absolute.border-transparent.dark\\:border-transparent.ease-out-quad.cursor-pointer"
        )

        print(f"Found {len(thumbnail_links)} thumbnail(s).")

        # 5) Loop through each thumbnail
        for i, link in enumerate(thumbnail_links):
            try:
                # Scroll element into view, then click it
                driver.execute_script("arguments[0].scrollIntoView(true);", link)
                time.sleep(0.5)
                link.click()

                # Wait for the lightbox to appear
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        'div[class*="bg-dark-950"]'  # Lightbox overlay
                    ))
                )

                # Click hamburger -> Copy -> Prompt
                prompt_text = click_hamburger_copy_prompt(driver)

                # Print the prompt text we retrieved from the clipboard
                if prompt_text:
                    print(f"Thumbnail #{i} prompt:\n{prompt_text}\n{'-'*50}")

                # Close the lightbox
                close_lightbox(driver)

            except Exception as e:
                print(f"Error handling thumbnail #{i}: {e}")
                # Attempt to close if stuck
                close_lightbox(driver)

        # End of loop
        print("All done.")

    finally:
        time.sleep(3)  # Just a pause
        driver.quit()


if __name__ == "__main__":
    main()
