import requests
from bs4 import BeautifulSoup

url = "https://es.wikipedia.org/wiki/Bernardo_O%27Higgins"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

print(soup.prettify())
