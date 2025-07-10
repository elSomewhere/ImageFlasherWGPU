import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def open_midjourney_job_page():
    # Create a ChromeOptions object (optional if you want specific settings)
    options = uc.ChromeOptions()

    # Example: Run in headless mode
    # options.add_argument('--headless')

    # Start undetected Chrome driver
    driver = uc.Chrome(options=options)

    # Navigate to the Midjourney job page
    url = "https://www.midjourney.com/jobs/eb9606a5-2604-4049-933f-41b1de1baa2c?index=0"
    driver.get(url)

    # Optional: wait for a specific element or for the page to load.
    # Example: Wait up to 10 seconds until an element on the page is present
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        print("Page loaded successfully!")
    except:
        print("Timed out waiting for page to load.")

    # Do something on the page, scrape data, or just keep it open for demonstration
    time.sleep(5)  # Let us see the page for 5 seconds

    # Print current page title
    print("Page Title:", driver.title)

    # Close the driver
    driver.quit()

if __name__ == "__main__":
    open_midjourney_job_page()
