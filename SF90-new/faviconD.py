from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import imagehash

class FaviconDownloader:
    def __init__(self, website_url, output_image_path):
        self.website_url = website_url
        self.output_image_path = output_image_path

    def download_favicon(self, favicon_url):
        try:
            response = requests.get(favicon_url)
            response.raise_for_status()
            with open(self.output_image_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded favicon: {favicon_url} -> {self.output_image_path}")
        except Exception as e:
            print(f"Error downloading favicon {favicon_url}: {e}")

    def get_favicons_and_download(self):
        try:
            response = requests.get(self.website_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            favicon_links = []

            for link in soup.find_all('link', rel=['favicon', 'icon', 'shortcut icon', 'SHORTCUT ICON', 'Shortcut Icon']):
                favicon_url = link.get('href')
                if favicon_url:
                    favicon_links.append(urljoin(self.website_url, favicon_url))

            for favicon_url in favicon_links:
                self.download_favicon(favicon_url)
        except Exception as e:
            print(f"Error fetching favicons: {e}")


imagt1 = 'favicon1.png'
imagt2 = 'favicon2.png'

website_url1 = "https://www.youtube.com/"
website_url2 = "https://www.youtube.com/"

favicon_downloader1 = FaviconDownloader(website_url1, imagt1)
favicon_downloader1.get_favicons_and_download()

favicon_downloader2 = FaviconDownloader(website_url2, imagt2)
favicon_downloader2.get_favicons_and_download()

hash1 = imagehash.average_hash(Image.open(imagt1))
hash2 = imagehash.average_hash(Image.open(imagt2))
diff = hash1 - hash2
print(diff)


