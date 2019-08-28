# Homework 7

1. Scrape apartments for sale on https://dom.ria.com/. Gather at least 15 000 entries. For each one you should have:

    - title of page
    - price in UAH and USD
    - url to images
    - description
    - street
    - region
    - square (total and per rooms)
    - number of rooms
    - floor
    - year of construction
    - heating
    - who is selling
    - walls material
    - verified price
    - verified apartment
    - geolocation
    - (optional) number of viewers
    - (optional) additional description (water, building condition, number of elevators, distance to airport/railway stations and etc)

2. You should store all the items in the PostgreSQL DB. Consider the database structure and relations between entities. **Attention!** don't check it into the repo.

3. Build HTTP API using Flask with two JSON endpoints:
GET /api/v1/statistics -> should return the number of apartments stored in the DB.
GET /api/v1/records -> should return a list of apartments sorted by the publication date, also should support limit and offset parameters (For example /api/v1/records?limit=10&offset=5 returns 10 records starting from the 5th position)

## Bonus points:
- build EDA on text stats, most common words, language used in review
- add more useful dataset information into /api/v1/statistics endpoint

## Tips:
- try using [devtools](https://developers.google.com/web/tools/chrome-devtools/) to figure out the structure of the page before running your scraper
- you could try seeing how page changes by [disabling javascript](chrome://settings/content/javascript)
- don't forget to do timeouts
- think about code structure. The data you crawled will be used for training models in the future, so it's your foundation for your next homework


# Deadline

**Due on 10.08.2019 23:59**
