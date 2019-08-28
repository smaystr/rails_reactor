import json
import re
from collections import defaultdict


class ApartmentParser():
    def __init__(self, page):
        self.page = page
        self.stats = self.get_stats()

    def extract_all(self):
        self.title = self.get_title()
        self.usd_price, self.uah_price = self.get_prices()
        self.description = self.get_description()
        self.verified_price, self.verified_app = self.get_verification()
        self.photos = self.get_photos()
        self.longitude, self.latitude = self.get_location()
        self.properties = self.get_house_properties()
        self.street, self.city, self.state = self.get_address()

        self.house_info = self.get_house_properties()
        self.floor = self.house_info['Этаж']
        self.rooms = self.house_info['Комнат']
        self.total_area = int(self.house_info['Площадь'][0])
        self.wall_material = self.house_info['Тип стен']
        self.seller = self.house_info['Тип предложения']
        self.construction_year = self.house_info['Год постройки']
        self.heating = self.house_info['Отопление']

    def get_house_properties(self):
        house_info = defaultdict(lambda: None)
        for descriptor in self.page.find_all("li", attrs={"class": "mt-15 boxed v-top"}):
            app_info = descriptor.text.strip().split("\n")
            house_info[app_info[0]] = app_info[2].strip()
        if "Площадь" in house_info.keys():
            house_info['Площадь'] = re.findall("\d+", house_info["Площадь"])
        return house_info

    def get_verification(self, verifiers=["Перевірена ціна", "Перевірена квартира"]):
        # verify order corresponds to the order of verifiers
        verify = [False] * len(verifiers)

        for attribute in self.page.find_all("span", attrs={"class": "blue"}):
            str_attr = attribute.text
            if str_attr in verifiers:
                find_verifier = verifiers.index(str_attr)
                verify[find_verifier] = True

        return verify

    def get_address(self):
        house_location = self.page.find_all("span", attrs={"itemprop": "title"})
        return (house_location[-1].text,
                house_location[-2].text,
                house_location[-3].text)

    def get_description(self):
        return self.page.find("div", attrs={"id": "descriptionBlock"}).text

    def get_prices(self):
        # [usd,uah]
        first_price = int(re.findall("\d+", self.page.find("span", attrs={"class": "grey size13"}).text.replace(" ", ""))[0])
        second_price = int(re.findall("\d+", self.page.find("span", attrs={"class": "price"}).text.replace(" ", ""))[0])
        # usd price is smaller than uah price
        return (first_price, second_price) if first_price < second_price else (second_price, first_price)

    def get_title(self):
        return self.page.find("div", attrs={"class": "finalPage"}).text[2:]

    def get_photos(self, size_format="xg"):
        photos = []
        for photo in self.stats["dataForFinalPage"]["realty"]["photos"]:

            photo_url = photo["beautifulUrl"]

            format_start_index = photo_url.rfind(".")

            photo_with_size = photo_url[:format_start_index] + size_format + photo_url[format_start_index:]

            photos.append(photo_with_size)

        return photos

    def get_location(self):
        if 'longitude' not in self.stats["dataForFinalPage"]["realty"].keys() or self.stats["dataForFinalPage"]["realty"]["longitude"] == '':
            return (None, None)
        return (float(self.stats["dataForFinalPage"]["realty"]["longitude"]),
                float(self.stats["dataForFinalPage"]["realty"]["latitude"]))

    def get_stats(self):
        return json.loads(re.search(r"window.__INITIAL_STATE__\s*=\s*({.*});", self.page.text).group(1))
