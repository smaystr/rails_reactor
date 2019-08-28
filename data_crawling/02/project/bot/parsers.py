import re


def parse_images(
        img_urls,
        img_format='m.jpg'
):
    """
    Maximize the image size with format
    :type img_urls: list
    :type img_format: str
    """
    if type(img_urls) is not list:
        return None
    if not len(img_urls):
        return None
    filtered_img_urls = []
    for img_url in img_urls:
        if img_url.find('m.jpg') != -1 or img_url.find('xg.jpg') != -1 or img_url.find('fl.jpg') != -1 or img_url.find('f.jpg') != -1:
            filtered_img_urls.append(img_url)
        elif img_url.find('i.jpg') != -1:
            filtered_img_urls.append(img_url.replace('i.jpg', img_format))
    if not len(filtered_img_urls):
        return None
    return filtered_img_urls


def parse_year(year):
    """
    Get digits from item
    :type year: object
    """
    if type(year) is not str:
        return year

    numbers = [int(substring) for substring in re.split(r'-| ', year) if substring.isdigit()]

    if len(numbers) == 1:
        return numbers[0]

    if len(numbers) == 2:
        return round(numbers[0] + numbers[1]) / 2


def parse_verification(verification) -> tuple:
    """
    Parse the obtained list of verifications
    :type verification: list
    """
    price_verification, apartment_verification = False, False
    if verification is None:
        return price_verification, apartment_verification
    if len(verification) == 1:
        price_verification = verification[0] == 'Перевірена ціна'
        apartment_verification = verification[0] == 'Перевірена квартира'
    if len(verification) == 2:
        price_verification = verification[0] == 'Перевірена ціна'
        apartment_verification = verification[1] == 'Перевірена квартира'
    return price_verification, apartment_verification


def parse_price(price):
    """
    Extract the price from the obtained text
    :type price: object
    """
    if type(price) is not str:
        return price
    return float(price \
                 .replace(' ', '') \
                 .replace('$', ''))
