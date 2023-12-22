import numpy as np
import itertools
from PIL import Image
import imagehash
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import favicon
import requests
import dnstwist
import re
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class WebsiteAnalysis:
    def __init__(self, original_url, test_url):
        self.original_url = original_url
        self.test_url = test_url
        self.screenshot_path1 = "screenshot1.png"
        self.screenshot_path2 = "screenshot2.png"
        self.model = self._initialize_model()
        self.similarity_score = None

    def _initialize_model(self):
        base_model = VGG16(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def _capture_screenshot_firefox(self, url, screenshot_path):
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Firefox(options=options)

        try:
            driver.get(url)
            current_scroll_position = driver.execute_script("return window.pageYOffset;")
            duration = 1
            total_page_height = driver.execute_script("return document.body.scrollHeight;")
            increment = (total_page_height - current_scroll_position) / (duration * 10)

            for _ in range(int(duration * 10)):
                current_scroll_position += increment
                driver.execute_script(f"window.scrollTo(0, {current_scroll_position});")
                time.sleep(0.1)
            driver.save_full_page_screenshot(screenshot_path)
        finally:
            driver.quit()

    def _preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _extract_features(self, img_path):
        img_array = self._preprocess_image(img_path)
        features = self.model.predict(img_array)
        return features.flatten()

    def calculate_similarity(self):
        self._capture_screenshot_firefox(self.original_url, self.screenshot_path1)
        self._capture_screenshot_firefox(self.test_url, self.screenshot_path2)

        image_path1 = self.screenshot_path1
        image_path2 = self.screenshot_path2

        features1 = self._extract_features(image_path1)
        features2 = self._extract_features(image_path2)

        self.similarity_score = 1 - cosine(features1, features2)
        print(f"Similarity Score: {self.similarity_score}")

class URLComparator:
    def __init__(self, url1, url2):
        self.url1 = url1
        self.url2 = url2

    def extract_domain_parts(self, url):
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if match:
            domain = match.group(1)
            parts = domain.split('.')
            return parts
        return None

    def calculate_string_similarity(self, str1, str2):
        m = len(str1)
        n = len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(len(str1), len(str2))
        similarity = 1 - (dp[m][n] / max_len)
        return similarity

    def get_tld_similarity_score(self):
        parts1 = self.extract_domain_parts(self.url1)
        parts2 = self.extract_domain_parts(self.url2)
        if parts1 is not None and parts2 is not None:
            tld_similarity = self.calculate_string_similarity(parts1[-1], parts2[-1])
            return tld_similarity
        else:
            return None

    def get_domain_name_similarity_score(self):
        parts1 = self.extract_domain_parts(self.url1)
        parts2 = self.extract_domain_parts(self.url2)
        parts1.pop(-1)
        parts2.pop(-1)
        parts1 = ''.join(parts1)
        parts2 = ''.join(parts2)

        if parts1 is not None and parts2 is not None:
            domain_name_similarity = self.calculate_string_similarity(parts1, parts2)
            return domain_name_similarity
        else:
            return None

class URLAnalysis:
    @staticmethod
    def api_analysis(original_url):
        print("You selected option 3.")
        url = original_url
        data = dnstwist.run(domain=url, registered=True, format='null')
        print(data)

    @staticmethod
    def similarity_analysis(original_url):
        url = original_url
        data = dnstwist.run(domain=url, format='list')
        print(data)

# class FaviconDownloader:
#     def __init__(self, website_url, output_image_path):
#         self.website_url = website_url
#         self.output_image_path = output_image_path

#     def download_favicon(self, favicon_url):
#         try:
#             response = requests.get(favicon_url)
#             response.raise_for_status()
#             with open(self.output_image_path, 'wb') as f:
#                 f.write(response.content)
#             print(f"Downloaded favicon: {favicon_url} -> {self.output_image_path}")
#             return True
#         except Exception as e:
#             print(f"Error downloading favicon {favicon_url}: {e}")
#             return False

#     def get_favicons_and_download(self):
#         try:
#             response = requests.get(self.website_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.text, 'html.parser')
#             favicon_links = []

#             for link in soup.find_all('link', rel=['favicon', 'icon', 'shortcut icon', 'SHORTCUT ICON', 'Shortcut Icon']):
#                 favicon_url = link.get('href')
#                 if favicon_url:
#                     favicon_links.append(urljoin(self.website_url, favicon_url))

#             for favicon_url in favicon_links:
#                 success = self.download_favicon(favicon_url)
#                 if not success:

#                     self.output_image_path = None
#         except Exception as e:
#             print(f"Error fetching favicons: {e}")

#             self.output_image_path = None


VOTING_WEIGHTS = {
    'website_similarity': 1,
    'tld_similarity': 1,
    'domain_name_similarity': 1,
    # 'favicon_difference': 1,
}

THRESHOLD = 1

suspicious_links = []
with open('original.txt', 'r') as file1:
    original_urls = [line.strip() for line in file1]
    with open('database.txt', 'r') as file2:
        test_urls = [line.strip() for line in file2]
for original_url, test_url in itertools.product(original_urls, test_urls):
    try:
        print(original_url)
        print(test_url)
        website_analysis = WebsiteAnalysis(original_url, test_url)
        website_analysis.calculate_similarity()
        website_similarity_vote = 1 if website_analysis.similarity_score >= 0.75 else 0

        url_comparator = URLComparator(original_url, test_url)
        tld_similarity = url_comparator.get_tld_similarity_score()
        domain_name_similarity = url_comparator.get_domain_name_similarity_score()
        tld_similarity_vote = 1 if tld_similarity >= 0.5 else 0
        domain_name_similarity_vote = 1 if domain_name_similarity >= 0.5 else 0

        # favicon_downloader1 = FaviconDownloader(original_url, 'favicon1.png')
        # favicon_downloader1.get_favicons_and_download()
        # favicon_difference_vote = 1 if favicon_downloader1.output_image_path is not None else 0

        # favicon_downloader2 = FaviconDownloader(test_url, 'favicon2.png')
        # favicon_downloader2.get_favicons_and_download()
        # favicon_difference_vote *= 1 if favicon_downloader2.output_image_path is not None else 0

        total_votes = (
            VOTING_WEIGHTS['website_similarity'] * website_similarity_vote +
            VOTING_WEIGHTS['domain_name_similarity'] * domain_name_similarity_vote 
            # + VOTING_WEIGHTS['favicon_difference'] * favicon_difference_vote
        )

        if total_votes >= THRESHOLD:
            suspicious_links.append(test_url)
    except Exception as e:
        print(f"Error processing URLs: {e}")

print(suspicious_links)
