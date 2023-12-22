import requests
from bs4 import BeautifulSoup

def get_backlinks(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad requests

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the <a> tags with 'href' attribute
        links = soup.find_all('a', href=True)

        # Extract and print the 'href' attribute of each link
        for link in links:
            href = link['href']
            print(href)

    except Exception as e:
        print(f"Error: {e}")

# Replace 'https://example.com' with the target website URL
target_url = 'https://www.youtube.com'

get_backlinks(target_url)