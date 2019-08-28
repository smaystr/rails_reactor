IMAGE_URL_PATTERN = "//div[@class='app-content']//img/@src"
APARTMENT_PAGE_PATTERN = "//a[contains(@class,'realtyPhoto')]//@href"
NEXT_PAGE_PATTERN = "//span[@class='page-item next text-r']/a/@href"
DESCRIPTION_PATTERN = "//div[@id='descriptionBlock']//text()"

HEATING_PATTERN = "//li[contains(.,'Отопление') and boolean(@data-v-4d6155be)]/div[@class='indent']/text()"
OFFER_TYPE_PATTERN = "//li[contains(.,'Тип предложения') and boolean(@data-v-4d6155be)]/div[@class='indent']/text()"
BUILT_TIME_PATTERN = "//li[contains(.,'Год постройки') and boolean(@data-v-4d6155be)]/div[@class='indent']/text()"
TITLE_PATTERN = "//h1/text()"

IS_VERIFIED_PATTERN = "//li[contains(@class,'labelHot checked')]"
TAGS_PATTERN = "//li[@class='labelHot']/text()"

INITIAL_STATE_PATTERN = "window.__INITIAL_STATE__=(.+}}}});"

ATTRIBUTES_FROM_INIT_WINDOW = ('rooms_count',
                               'publishing_date',
                               'floor',
                               'levels',
                               'realty_id',
                               'total_square_meters',
                               'floors_count',
                               'city_id',
                               'city_name',
                               'wall_type',
                               'latitude',
                               'longitude')