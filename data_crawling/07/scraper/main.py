from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from pageparse import ApartmentParser
import config
import psycopg2
from time import sleep


def connect_db():
  connection = psycopg2.connect(
      host=config.DB_HOST,
      database=config.DB_NAME,
      user=config.DB_USER,
      password=config.DB_PASSWORD,
  )
  return (connection, connection.cursor())


def bd_insert_appartment(cursor, connection, ap, id):
  cursor.execute("INSERT INTO apartment VALUES (default,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                 (
                     ap.uah_price,
                     ap.usd_price,
                     ap.description,
                     ap.street,
                     ap.state,
                     ap.total_area,
                     ap.rooms,
                     ap.construction_year,
                     ap.heating,
                     ap.seller,
                     ap.wall_material,
                     ap.verified_price,
                     ap.verified_app,
                     ap.latitude,
                     ap.longitude,
                     ap.photos,
                     ap.city,
                     ap.title,
                 )
                 )
  connection.commit()


def get_page_entries(url):
  try:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

  except requests.exceptions.RequestException as e:
    print(f"Cannot access page {url}")
  else:
    return soup.find_all("a", attrs={"class": "realtyPhoto"})


def run():
  connection, cursor = connect_db()
  verbose_rounds = 100
  apps_crawled = 0
  current_page = 2
  while apps_crawled < config.ENTRIES_TO_PARSE:
    page_to_crawl = urljoin(config.BASE_URL, config.PAGES_URL + str(current_page))
    page_entries = get_page_entries(page_to_crawl)
    if len(page_entries) == 0:
      # no more pages to parse
      break
    for entry in page_entries:
      link = entry["href"]
      house = requests.get(urljoin(config.BASE_URL, link))
      house_parse = BeautifulSoup(house.text, "html.parser")
      apartment = ApartmentParser(house_parse)

      apartment.extract_all()
      bd_insert_appartment(cursor, connection, apartment, apps_crawled)

      apps_crawled += 1
      sleep(0.01)
      if apps_crawled % verbose_rounds == 0:
        print('Done ', apps_crawled)
      if apps_crawled == config.ENTRIES_TO_PARSE:
        break
    else:
      current_page += 1
      continue
    break

  cursor.close()
  connection.close()


if __name__ == "__main__":
  run()
