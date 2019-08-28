GET_VIEWS = "//div[@class='app-content']//ul[@class='greyLight size13 unstyle']//li//b/text()"

GET_PRICE_VERIFICATION = "//div[@class='app-content']//ul[@class='unstyle']//span[@class='blue']/text()"

GET_APARTMENT_VERIFICATION = "//div[@class='app-content']//ul[@class='unstyle']//span[@class='blue']/text()"

GET_IMAGE_URLS = "//div[@class='app-content']//img/@src"

GET_DESCRIPTION = "//div[@class='row finalPage']//main[@class='span8']//div[@id='descriptionBlock']/text()"

GET_PRICE = "//div[@class='row finalPage']//ul[@class='unstyle']//div[@class='ml-30']//span[@class='price']/text()"

GET_TITLE = "//div[@class='app-content']//h1/text()"

GET_WINDOW_INITIAL_STATE = '//script[contains(., "window.__INITIAL_STATE")]/text()'

GET_PAGE_CONTENT = "//div[@id='catalogResults']//section//a/@href"

GET_NEXT_PAGE = "//div[@id='pagination']//span[@class='page-item next text-r']//a[@class='page-link']/@href"
