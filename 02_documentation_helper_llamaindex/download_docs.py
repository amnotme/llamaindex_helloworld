import urllib.parse
import requests
from bs4 import BeautifulSoup
import os

url = 'https://gpt-index.readthedocs.io/en/stable/'

output_dir = './llamaindex_docs/'

os.makedirs(output_dir, exist_ok=True)

response = requests.get(url=url)
soup = BeautifulSoup(response.text, 'html.parser')

links = soup.find_all('a', href=True)

for link in links:
    href = link['href']

    if href.endswith('.html'):
        if not href.startswith('http'):
            href = urllib.parse.urljoin(base=url, url=href)

        print(f"downloading {href}")
        file_response = requests.get(href)

        file_name = os.path.join(output_dir, os.path.basename(href))
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(file_response.text)