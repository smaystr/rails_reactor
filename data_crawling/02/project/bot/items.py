import scrapy


class DomRiaItem(scrapy.Item):
    # Item ID
    id = scrapy.Field()
    # Offer
    title = scrapy.Field()
    seller = scrapy.Field()
    price = scrapy.Field()
    price_verification = scrapy.Field()
    apartment_verification = scrapy.Field()

    # Subinfo
    publish_date = scrapy.Field()
    image_urls = scrapy.Field()

    # Apartment
    year = scrapy.Field()
    total_area = scrapy.Field()
    living_area = scrapy.Field()
    kitchen_area = scrapy.Field()
    floor = scrapy.Field()
    rooms = scrapy.Field()

    # Additions
    description = scrapy.Field()
    heating = scrapy.Field()
    walls = scrapy.Field()

    # Location
    longitude = scrapy.Field()
    latitude = scrapy.Field()
    region = scrapy.Field()
    city = scrapy.Field()
    street = scrapy.Field()
