import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash
from fuzzywuzzy import fuzz
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from Screenshot import Screenshot
import time
import favicon
import requests
import dnstwist

class WebsiteAnalysis:
    def __init__(self, original_url, test_url):
        self.original_url = original_url
        self.test_url = test_url
        self.screenshot_path1 = "screenshot1.png"
        self.screenshot_path2 = "screenshot2.png"
        self.model = self._initialize_model()

    def _initialize_model(self):
        base_model = VGG16(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def _capture_screenshot_firefox(self, url, screenshot_path):
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        driver = webdriver.Firefox()

        try:
            driver.get(url)
            current_scroll_position = driver.execute_script("return window.pageYOffset;")
            duration = 3
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

        similarity_score = 1 - cosine(features1, features2)
        print(f"Similarity Score: {similarity_score}")


class FaviconAnalysis:
    def __init__(self, original_url):
        self.original_url = original_url
        self.model = self._initialize_model()

    def _initialize_model(self):
        base_model = VGG16(weights='imagenet')
        return Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

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

    def calculate_similarity(self, test_favicon_path):
        image_path1 = 'Data-Test/favicons/favicon1.png'
        image_path2 = test_favicon_path

        features1 = self._extract_features(image_path1)
        features2 = self._extract_features(image_path2)

        similarity_score = 1 - cosine(features1, features2)
        print(f"Similarity Score: {similarity_score}")


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


def main():
    original_url = "https://www.icicibank.com/"
    list_urls = ["http://icicicardservices.in/"]

    i = 0
    while True:
        print("Main Menu:")
        print("1. Website Screenshot Analysis")
        print("2. Favicon Analysis")
        print("3. Find Similar looking URLs that are Already registered")
        print("4. Generate Permutation of URL")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if 0 <= i < len(list_urls):
            test_url = list_urls[i]
            i = i + 1

        if choice == "1":
            website_analysis = WebsiteAnalysis(original_url, test_url)
            website_analysis.calculate_similarity()

        elif choice == "2":
            test_favicon_path = input("Enter the path to the favicon image: ")
            favicon_analysis = FaviconAnalysis(original_url)
            favicon_analysis.calculate_similarity(test_favicon_path)

        elif choice == "3":
            URLAnalysis.api_analysis(original_url)

        elif choice == "4":
            URLAnalysis.similarity_analysis(original_url)

        elif choice == "5":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
