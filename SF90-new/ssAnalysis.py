import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from Screenshot import Screenshot
import time
from selenium.webdriver.firefox.options import Options

screenshot_path1 = "screenshot1.png"
screenshot_path2 = "screenshot2.png"
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)



def capture_screenshot_firefox(url, screenshot_path):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

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

url1 = "https://hdfcrewards.in/"
url2 = "https://www.hdfcbank.com/"


capture_screenshot_firefox(url1, screenshot_path1)
capture_screenshot_firefox(url2, screenshot_path2)

image_path1 = 'screenshot1.png'
image_path2 = 'screenshot2.png'

features1 = extract_features(image_path1, model)
features2 = extract_features(image_path2, model)

similarity_score = calculate_similarity(features1, features2)
print(f"Similarity Score: {similarity_score}")