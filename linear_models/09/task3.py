import requests
from pathlib import Path
import multiprocessing
from tqdm import tqdm

data_path = Path(Path(__file__).parent) / 'data'
data_path.mkdir(exist_ok=True)


def get_file(file_name):
    base_url = 'http://ps2.railsreactor.net/datasets/medicine/'
    file_url =   base_url + file_name
    file_content = requests.get(file_url).text
    with open(data_path / file_name, 'w+', encoding='utf-8') as output_file:
        output_file.write(file_content)


def get_data():
    file_names = [
        'heart_test.csv',
        'heart_train.csv',
        'insurance_test.csv',
        'insurance_train.csv',
    ]
    with multiprocessing.Pool(8) as pool:
        list(tqdm(pool.imap(get_file, file_names), total=len(file_names)))


if __name__ == '__main__':
    get_data()
