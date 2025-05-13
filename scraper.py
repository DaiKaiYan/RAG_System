from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time


class WechatScraper:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()

    def search(self, keyword):
        self.driver.get("https://weixin.sogou.com/")
        search_box = self.driver.find_element(By.ID, "query")
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)
        return self._extract_links()

    def _extract_links(self):
        articles = self.driver.find_elements(By.CSS_SELECTOR, ".news-box a")
        top_5_articles = articles[:5]
        return [article.get_attribute("href") for article in top_5_articles]

    def get_content(self, link):
        self.driver.get(link)
        time.sleep(2)
        try:
            title = self.driver.find_element(By.TAG_NAME, "h1").text
        except:
            title = "No title found"
        try:
            content = self.driver.find_element(By.ID, "js_content").text
        except:
            content = "No content found"
        return {"title": title, "content": content}

    def close(self):
        self.driver.quit()