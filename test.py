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
from urllib.parse import urlparse
import json
import pickle
import time
import sys

success = 0
i=0
url =""
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
urlparse = urlparse(url)
domain = urlparse.netloc

whois_response = whois.query(domain)
print(response.history)
