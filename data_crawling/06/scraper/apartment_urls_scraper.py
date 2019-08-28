import csv
import requests
from urllib.parse import urljoin
from time import time, sleep
from bs4 import BeautifulSoup, SoupStrainer


class ApartmentUrlsScraper:
    def __init__(self):
        self.base_url = 'https://dom.ria.com/'
        self.scrap_url = 'https://dom.ria.com/prodazha-kvartir/kiev/?page={page}'
        self._start_time = time()

    # dom.ria can't detect socks(TOR) proxy
    @staticmethod
    def get_tor_session():
        session = requests.session()
        # Tor uses the 9050 or 9150 port as the default socks port
        session.proxies = {'http': 'socks5://127.0.0.1:9150', 'https': 'socks5://127.0.0.1:9150'}
        return session

    @staticmethod
    def _save_to_csv(save_list, filename):
        with open(filename, 'w+', newline='') as file:
            wr = csv.writer(file, delimiter='\n')
            wr.writerow(save_list)

    def _save_logs(self, page, save_list):
        if page % 100 == 0 or page == 5:
            path_to_save = f'logs/dr_kiev_urls_{page}.csv'
            self._save_to_csv(save_list, path_to_save)
            print(f'Successfully saved to {path_to_save} in {time() - self._start_time}')

    def _get_dom_ria_page(self, page):
        re_list = []
        tor_session = self.get_tor_session()
        response = tor_session.get(self.scrap_url.format(page=page))
        if response.status_code == 200:
            strainer = SoupStrainer('a', class_='blue')
            soup = BeautifulSoup(response.content, 'lxml', parse_only=strainer)
            for element in soup.find_all('a'):
                element_ref = element.get('href')
                if element_ref:
                    re_element = urljoin(self.base_url, element_ref)
                    re_list.append(re_element)
        return re_list

    def scrap_dom_ria_pages(self):
        page = 1
        re_all_list = []
        while page <= 1506:
            self._save_logs(page, re_all_list)
            re_all_list += self._get_dom_ria_page(page)
            page += 1
            sleep(2)
        self._save_to_csv(re_all_list, 'data/dr_kiev_urls.csv')
        return re_all_list
