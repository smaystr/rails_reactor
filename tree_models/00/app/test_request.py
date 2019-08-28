import requests
import json
import yaml


def preprocess_output(string):
    return string.replace("'", '"').replace("None", "null")


def main():
    apartment_params = {
        "floor_located": 29,
        "number_of_floors_in_the_house": 30,
        "wall_type": "кирпич",
        "number_rooms": 3,
        "position": 10,
        "construction_period": "2018",
        "image_urls": [[]],
        "latitude": 0,
        "longitude": 0,
        "description": "Квартира на Осокорках",
        "apartment_area": 100,
        "tags": [[]],
        "city_id": "10",
        "heating": "централизованное",
        "offer_type": "от собственника",
    }

    response1 = requests.get(
        "http://127.0.0.1:5000/app/v1/predict",
        params={"model": "lgbm", "features": json.dumps(apartment_params)},
    )
    output1 = response1.text

    response2 = requests.get("http://127.0.0.1:5000/app/v1/statistics")
    output2 = response2.text

    response3 = requests.get(
        "http://127.0.0.1:5000/app/v1/record", params={"limit": 2, "offset": 20}
    )
    # Yaml can easily read unicode :)
    output3 = yaml.safe_load(response3.text)

    for output in [output1, output2, output3]:
        with open("api_outputs.json", "a", encoding="utf-8") as file:
            data = json.loads(preprocess_output(str(output)))
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.write(",\n")


if __name__ == "__main__":
    main()
