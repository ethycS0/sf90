import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash
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


originalURL = "https://www.hdfcbank.com/"
listURL = ["http://hdfcrewards.in/","https://chat.openai.com/c/10e9ddb7-62a0-46e2-ac6b-706a6d20f0f7","https://stackoverflow.com/questions/49343024/getting-typeerror-failed-to-fetch-when-the-request-hasnt-actually-failed","https://www.searchenginejournal.com/linkedin-alternatives/297409/","https://icicirewards.online/","https://www.myopportunity.com/","https://www.w3schools.com/html/","https://github.com/ethycS0/SF90","https://cloudconvert.com/png-converter","https://www.indiamart.com/"]

def website_image():
        print("You selected option 1.")

        screenshot_path1 = "screenshot1.png"
        screenshot_path2 = "screenshot2.png"
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)



        def capture_screenshot_firefox(url, screenshot_path):
            options = webdriver.FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox()

            try: 
                driver.get(url)
                current_scroll_position = driver.execute_script("return window.pageYOffset;")
                duration=3
                total_page_height = driver.execute_script("return document.body.scrollHeight;")
                increment = (total_page_height - current_scroll_position) / (duration * 10) 

                for _ in range(int(duration * 10)):
                    current_scroll_position += increment
                    driver.execute_script(f"window.scrollTo(0, {current_scroll_position});")
                    time.sleep(0.1)
                driver.save_full_page_screenshot(screenshot_path)
            finally:
                driver.quit()

        def preprocess_image(img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array

        def extract_features(img_path, model):
            img_array = preprocess_image(img_path)
            features = model.predict(img_array)
            return features.flatten()

        def calculate_similarity(feature1, feature2):
            return 1 - cosine(feature1, feature2)

        url1 = originalURL
        url2 = testURL


        capture_screenshot_firefox(url1, screenshot_path1)
        capture_screenshot_firefox(url2, screenshot_path2)

        image_path1 = 'screenshot1.png'
        image_path2 = 'screenshot2.png'

        features1 = extract_features(image_path1, model)
        features2 = extract_features(image_path2, model)

        similarity_score = calculate_similarity(features1, features2)
        print(f"Similarity Score: {similarity_score}")

def favicon():
        print("You selected option 2.")
        hash1 = imagehash.average_hash(Image.open('Test-Data/favicons/HDFC/fake.png'))
        hash2 = imagehash.average_hash(Image.open('Test-Data/favicons/HDFC/real.ico'))
        diff = hash1 - hash2
        print(diff)
        def preprocess_image(img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array

        def extract_features(img_path, model):
            img_array = preprocess_image(img_path)
            features = model.predict(img_array)
            return features.flatten()

        def calculate_similarity(feature1, feature2):
            return 1 - cosine(feature1, feature2)
        icon = favicon.get(originalURL)

        image_path1 = 'Data-Test/favicons/favicon1.png'
        image_path2 = 'Test-Data/favicons/favicon2.png'

        features1 = extract_features(image_path1, model)
        features2 = extract_features(image_path2, model)

        similarity_score = calculate_similarity(features1, features2)
        print(f"Similarity Score: {similarity_score}")

def url_api():
        print("You selected option 3.")
        url = originalURL
        data = dnstwist.run(domain=url, registered=True, format='null')
        print(data)

def url_sim():
        url = originalURL
        data = dnstwist.run(domain=url, format='list')
        print(data)

def main():
        i=0
        global testURL
        while True:
            print("Main Menu:")
            print("1. Website Screenshot Analysis")
            print("2. Favicon Analysis")
            print("3. Find Similiar looking URLs that are Already registered")
            print("4. Generate Permutation of URL")
            print("5. Exit")

            choice = input("Enter your choice: ")
            if 0 <= i < len(listURL):
                testURL = listURL[i]
                i=i+1
            if choice == "1":
                website_image()
            elif choice == "2":
                favicon()
            elif choice == "3":
                url_api()
            elif choice == "4":
                url_sim()
            elif choice == "5":
                print("Exiting program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
        main()