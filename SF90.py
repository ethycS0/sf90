# Install all modules using pip install (search package name on google and install it eg "pip install bs4")

# Script modules/libraries
import ipaddress
import re
from googlesearch import search
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse, parse_qs
import json
import pickle
import time
# from http.server import BaseHTTPRequestHandler, HTTPServer
# import asyncio
import re

# ML modules/libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')


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


# hostName = "0.0.0.0"
# serverPort = 8000

# class MyServer(BaseHTTPRequestHandler):
#         def do_GET(self):
#             # url = self.path[1:]
#             parsed_url = urlparse(self.path)
#             query_params = parse_qs(parsed_url.query)

#             # Get the value of the 'url' parameter
#             url = query_params.get('url', [''])[0]

#             # Do something with the 'url' parameter
#             print('Received URL parameter:', url)
#             url_param = query_params.get('url', [''])[0]

#             # print(self.path[1:])
#             loaded_model = pickle.load(open("models/full.pkl", "rb"))
#             obj = FeaturesFinder(url)
#             Features = obj.getFeaturesList()
#             print(Features)
#             start_time = time.time()
#             url = Features.pop(0)
#             y_predicted = loaded_model.predict([Features])
#             print("Prediction processing finished --- %s seconds ---" % (time.time() - start_time))
#             print("URL is: ", y_predicted)
#             if y_predicted == 1:
#                 response_data = {'safety': '1', 'url_param': url_param}
#             elif y_predicted == 0:
#                 response_data = {'safety': '0', 'url_param': url_param}
#             elif y_predicted == -1:
#                 response_data = {'safety': '-1', 'url_param': url_param}
        

#             # print(response_data)
#             response_json = json.dumps(response_data)
#             self.send_response(200)
#             self.send_header('Content-type', 'application/json')
#             self.end_headers()
#             self.wfile.write(bytes(response_json, 'utf-8'))



class FeaturesFinder:
    features = []
    def __init__(self,url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""
        self.start_time = time.time()

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        try:
            self.whois_response = whois.query(self.domain)
        except:
            pass

        self.features.append(self.url)
        self.features.append(self.UsingIp())
        self.features.append(self.longUrl())
        self.features.append(self.shortUrl())
        self.features.append(self.symbol())
        self.features.append(self.redirecting())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.https())
        self.features.append(self.DomainRegLen())
        self.features.append(self.Favicon())
        self.features.append(self.NonStdPort())
        # self.features.append(self.HTTPSDomainURL())
        self.features.append(self.RequestURL())
        self.features.append(self.AnchorURL())
        self.features.append(self.LinksInScriptTags())
        self.features.append(self.ServerFormHandler())
        self.features.append(self.InfoEmail())
        self.features.append(self.AbnormalURL())
        self.features.append(self.WebsiteForwarding())
        self.features.append(self.StatusBarCust())
        self.features.append(self.DisableRightClick())
        self.features.append(self.UsingPopupWindow())
        self.features.append(self.IframeRedirection())
        self.features.append(self.AgeofDomain())
        self.features.append(self.DNSRecording())
        # self.features.append(self.WebsiteTraffic())
        self.features.append(self.PageRank())
        self.features.append(self.GoogleIndex())
        self.features.append(self.LinksPointingToPage())
        # self.features.append(self.StatsReport())


    # 1.Using IP
    def UsingIp(self):
        try:
            ipaddress.ip_address(self.domain)
            return -1
        except:
            return 1

    # 2.long URL
    def longUrl(self):
        if len(self.url) < 54:
            return 1
        if len(self.url) >= 54 and len(self.url) <= 75:
            return 0
        return -1

    # 3.short URL
    def shortUrl(self):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net', self.domain)
        if match:
            return -1
        return 1

    # 4.@ Symbol
    def symbol(self):
        if re.findall("@",self.url):
            return -1
        return 1
    
    # 5.// Redirecting
    def redirecting(self):
        if self.url.count('//') > 1 or (self.url.find('//') != 5 and self.url.find('//') != 6):
            return -1
        else:
            return 1

    # 6.PrefixSuffix
    def prefixSuffix(self):
        try:
            match = re.findall('\-', self.domain)
            if match:
                return -1
            return 1
        except:
            return -1
    
    # 7.SubDomains
    def SubDomains(self):
        dot_count = len(re.findall("\.", self.domain))
        if dot_count == 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8.HTTPS
    def https(self):
        try:
            response = requests.head(self.url, allow_redirects=True, timeout=5)
            final_url = response.url

            if final_url.startswith('https://'):
                return 1
            else:
                return -1
        except:
            return -1



    # 9.DomainRegLen
    def DomainRegLen(self):
        try:
            expiration_date = self.whois_response.expiration_date
            creation_date = self.whois_response.creation_date
            try:
                if(len(expiration_date)):
                    expiration_date = expiration_date[0]
            except:
                pass
            try:
                if(len(creation_date)):
                    creation_date = creation_date[0]
            except:
                pass

            age = (expiration_date.year-creation_date.year)*12+ (expiration_date.month-creation_date.month)
            if age >=12:
                return 1
            return -1
        except:
            return -1

    # 10. Favicon
    def Favicon(self):
        try:
            for head in self.soup.find_all('head'):
                for head.link in self.soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                    if self.url in head.link['href'] or len(dots) == 1 or self.domain in head.link['href']:
                        return 1
            return -1
        except:
            return -1

    # 11. NonStdPort
    def NonStdPort(self):
        try:
            port = self.domain.split(":")
            if len(port)>1:
                return -1
            return 1
        except:
            return -1

    # 12. HTTPSDomainURL
    # def HTTPSDomainURL(self):
    #     try:
    #         if 'https' in self.domain:
    #             return -1
    #         return 1
    #     except:
    #         return -1
    
    # 13. RequestURL
    def RequestURL(self):
        try:
            success = 0
            i=0
            for img in self.soup.find_all('img', src=True):
                dots = [x.start(0) for x in re.finditer('\.', img['src'])]
                if self.url in img['src'] or self.domain in img['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for audio in self.soup.find_all('audio', src=True):
                dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
                if self.url in audio['src'] or self.domain in audio['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for embed in self.soup.find_all('embed', src=True):
                dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
                if self.url in embed['src'] or self.domain in embed['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for iframe in self.soup.find_all('iframe', src=True):
                dots = [x.start(0) for x in re.finditer('\.', iframe['src'])]
                if self.url in iframe['src'] or self.domain in iframe['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            try:
                percentage = 1-success/float(i) * 100
                if percentage < 22.0:
                    return 1
                elif((percentage >= 22.0) and (percentage < 61.0)):
                    return 0
                else:
                    return -1
            except:
                return 0
        except:
            return -1
    
    # 14. AnchorURL
    def AnchorURL(self):
        try:
            i,unsafe = 0,0
            for a in self.soup.find_all('a', href=True):
                if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (self.url in a['href'] or not self.domain in a['href']):
                    unsafe = unsafe + 1
                i = i + 1

            try:
                percentage = unsafe / float(i) * 100
                if percentage < 31.0:
                    return 1
                elif ((percentage >= 31.0) and (percentage < 67.0)):
                    return 0
                else:
                    return -1
            except:
                return -1

        except:
            return -1

    # 15. LinksInScriptTags
    def LinksInScriptTags(self):
        try:
            i,success = 0,0
        
            for link in self.soup.find_all('link', href=True):
                dots = [x.start(0) for x in re.finditer('\.', link['href'])]
                if self.url in link['href'] or self.domain in link['href'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for script in self.soup.find_all('script', src=True):
                dots = [x.start(0) for x in re.finditer('\.', script['src'])]
                if self.url in script['src'] or self.domain in script['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            try:
                percentage = success / float(i) * 100
                if percentage < 17.0:
                    return 1
                elif((percentage >= 17.0) and (percentage < 81.0)):
                    return 0
                else:
                    return -1
            except:
                return 0
        except:
            return -1

    # 16. ServerFormHandler
    def ServerFormHandler(self):
        try:
            if len(self.soup.find_all('form', action=True))==0:
                return 1
            else :
                for form in self.soup.find_all('form', action=True):
                    if form['action'] == "" or form['action'] == "about:blank":
                        return -1
                    elif self.url in form['action'] or self.domain in form['action'] or "https" not in form['action'] or 'http' not in form['action']:
                        return 1
                    else:
                        return 0
        except:
            return -1

    # 17. InfoEmail
    def InfoEmail(self):
        try:
            if "mailto:" in self.response.text:
                return -1
            else:
                return 1
        except:
            return -1

    # 18. AbnormalURL
    def AbnormalURL(self):
        try:
            if self.whois_response.name in self.domain:
                return 1
            else:
                return -1
        except:
            return -1

    # 19. WebsiteForwarding
    def WebsiteForwarding(self):
        try:
            if len(self.response.history) <= 1:
                return 1
            elif len(self.response.history) <= 4:
                return 0
            else:
                return -1
        except:
             return -1

    # 20. StatusBarCust
    def StatusBarCust(self):
        try:
            if re.findall("<script>.+onmouseover.+</script>", self.response.text):
                return -1
            else:
                return 1
        except:
             return -1

    # 21. DisableRightClick
    def DisableRightClick(self):
        try:
            if re.findall(r"event.button ?== ?2", self.response.text):
                return -1
            else:
                return 1
        except:
             return -1

    # 22. UsingPopupWindow
    def UsingPopupWindow(self):
        try:
            if re.findall(r"alert\(", self.response.text):
                return -1
            else:
                return 1
        except:
             return -1

    # 23. IframeRedirection
    def IframeRedirection(self):
        try:
            if re.findall(r"[<iframe>|<frameBorder>]", self.response.text):
                return -1
            else:
                return 1
        except:
             return -1

    # 24. AgeofDomain
    def AgeofDomain(self):
        try:
            creation_date = self.whois_response.creation_date
            try:
                if(len(creation_date)):
                    creation_date = creation_date[0]
            except:
                pass

            today  = date.today()
            age = (today.year-creation_date.year)*12+(today.month-creation_date.month)
            if age >=6:
                return 1
            return -1
        except:
            return -1

    # 25. DNSRecording    
    def DNSRecording(self):
        try:
            creation_date = self.whois_response.creation_date
            try:
                if(len(creation_date)):
                    creation_date = creation_date[0]
            except:
                pass

            today  = date.today()
            age = (today.year-creation_date.year)*12+(today.month-creation_date.month)
            if age >=6:
                return 1
            return -1
        except:
            return -1

    # 26. WebsiteTraffic   
    # def WebsiteTraffic(self):
    #     try:
    #         response = requests.post("https://websiteseochecker.com/website-traffic-checker/", {"ckwebsite": self.domain})
    #         time.sleep(3)
    #         html = response.text
    #         soup = BeautifulSoup(html, "html.parser")

    #         table = soup.find('table')

    #         header = []
    #         rows = []
    #         for i, row in enumerate(table.find_all('tr')):
    #             if i == 0:
    #                 header = [el.text.strip() for el in row.find_all('th')]
    #             else:
    #                 rows.append([el.text.strip() for el in row.find_all('td')])

    #         a = 0
    #         for row in rows:
    #             a += 1
    #             if a == 2:
    #                 traffic = row[2]

    #         if int(traffic) > 150:
    #             return 1
    #         return 0
    #     except:
    #         return 0

    # 27. PageRank
    def PageRank(self):
        try:
            for i in search(self.domain, sleep_interval=5, num_results=5):
                parsed = urlparse(i)
                parseddom = parsed.netloc
                if parseddom == self.domain:
                    return 1
            else:
                return -1
        except:
            return -1

    # 28. GoogleIndex
    def GoogleIndex(self):
        try:
            for i in search(self.domain, sleep_interval=3, num_results=3):
                parsed = urlparse(i)
                parseddom = parsed.netloc
                if parseddom == self.domain:
                    return 1
            else:
                return -1
        except:
            return -1

    # 29. LinksPointingToPage
    def LinksPointingToPage(self):
        try:
            number_of_links = len(re.findall(r"<a href=", self.response.text))
            if number_of_links == 0:
                return 1
            elif number_of_links <= 2:
                return 0
            else:
                return -1
        except:
            return -1

    # 30. StatsReport
    # def StatsReport(self):
    #     try:
    #         url_match = re.search(
    #       'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly', self.url)
    #         ip_address = socket.gethostbyname(self.domain)
    #         ip_match = re.search('146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
    #                             '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
    #                             '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
    #                             '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
    #                             '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
    #                             '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42', ip_address)
    #         if url_match:
    #             return -1
    #         elif ip_match:
    #             return -1
    #         return 1
    #     except:
    #         return 1
    
    def getFeaturesList(self):

        print("Prediction processing finished --- %s seconds ---" % (time.time() - self.start_time))
        return self.features
        # try:
        #     responsel = requests.get(self.url, timeout=20)
        #     return self.features

        # except RequestException as e:
        #     print(f"Error extracting features from {self.url}: {e}")

# webServer = HTTPServer((hostName, serverPort), MyServer)
# print("Server started http://%s:%s" % (hostName, serverPort))
# webServer.serve_forever()

for item in suspicious_links:
    url = item
    loaded_model = pickle.load(open("/home/arjun/Sync/Notes/SF90-old/models/full.pkl", "rb"))
    obj = FeaturesFinder(url)
    Features = obj.getFeaturesList()
    print(Features)

    start_time = time.time()
    url = Features.pop(0)
    y_predicted = loaded_model.predict([Features])
    print("Prediction processing finished --- %s seconds ---" % (time.time() - start_time))
    print("URL is: ", y_predicted)
# if y_predicted == 1:
#     response_data = {'safety': '1', 'url_param': url_param}
# elif y_predicted == 0:
#     response_data = {'safety': '0', 'url_param': url_param}
# elif y_predicted == -1:
#     response_data = {'safety': '-1', 'url_param': url_param}


# Choose coloumns, 1 for usual searching and 2 for spreadsheet paste

# 1
# column_names = ["UsingIP","LongURL","ShortURL","Symbol@","Redirecting//","PrefixSuffix","SubDomains","HTTPS","DomainRegLen","Favicon","NonStdPort","HTTPSDomainURL","RequestURL","AnchorURL","LinksInScriptTags","ServerFormHandler","InfoEmail","AbnormalURL","WebsiteForwarding","StatusBarCust","DisableRightClick","UsingPopupWindow","IframeRedirection","AgeofDomain","DNSRecording","WebsiteTraffic","PageRank","GoogleIndex","LinksPointingToPage","StatsReport"]

# 2
# column_names = ["","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]

# for n, v in zip(column_names, Features):
#    print("{}{}".format(n, v))


# ... (Your existing imports)

# ... (Your existing code)

# async def handle_request(reader, writer):
#     data = await reader.read(1000)

#     message = data.decode()
#     addr = writer.get_extra_info('peername')
    
#     # print("Received request from ", message)

#     parsed_url = urlparse(data)
    
#     query_params = parse_qs(parsed_url.query)
    
#     # print(query_params)
#         # Get the value of the 'url' parameter
    
#     regEx = "b'(https?://[^\s]*)"
    
#     res = re.findall(regEx, query_params)

#     url = res[0]
#     print(url)

#         # Do something with the 'url' parameter
#     print('Received URL parameter:', url)

#         # Do something with the 'url' parameter
#     print('Received URL parameter:', url)
#     url_param = query_params.get('url', [''])[0]

#     # print(self.path[1:])
#     loaded_model = pickle.load(open("models/main.pkl", "rb"))
#     obj = FeaturesFinder(url)
#     Features = obj.getFeaturesList()
#     print(Features)
#     start_time = time.time()
#     url = Features.pop(0)
#     y_predicted = loaded_model.predict([Features])
#     print("Prediction processing finished --- %s seconds ---" % (time.time() - start_time))
#     print("URL is: ", y_predicted)
#     if y_predicted == 1:
#             response_data = {'safety': '1', 'url_param': url_param}
#     elif y_predicted == 0:
#             response_data = {'safety': '0', 'url_param': url_param}
#     elif y_predicted == -1:
#             response_data = {'safety': '-1', 'url_param': url_param}
       

#         # print(response_data)
#     response_json = json.dumps(response_data)
#     addr.send_response(200)
#     addr.send_header('Content-type', 'application/json')
#     addr.end_headers()
#     addr.wfile.write(bytes(response_json, 'utf-8'))


#     # print("Closing connection from {}: {!r}".format(addr, message))
#     writer.close()

# async def main():
#     server = await asyncio.start_server(
#         handle_request, host=hostName, port=serverPort
#     )

#     addr = server.sockets[0].getsockname()
#     print(f'Serving on {addr}')

#     async with server:
#         await server.serve_forever()

# if __name__ == "__main__":
#     asyncio.run(main())
