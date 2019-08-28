import requests
import argparse
import multiprocessing
from pathlib import Path
from collections import Counter


def get_articles_urls(url: str) -> list:
    """
    :param url:
    :return: list with articles urls
    """
    file_with_articles_filenames = 'files.txt'
    try:
        response = requests.get(f'{url}/{file_with_articles_filenames}')
        if response:
            return [f'{url}/{article_name}' for article_name in response.text.strip().split('\n')]
        else:
            raise Exception(f'Response: {response}')
    except Exception as err:
        raise Exception(f'While reading \'{file_with_articles_filenames}\' from {url} occurred exception:\n{err}')


def get_articles_data_and_download(url: str):
    """
    Function that downloads article
    :param url:
    :return: article data
    """
    article_name = Path(url).name
    try:
        response = requests.get(url)
        if response:
            data = response.content.decode('utf-8')
            (Path('./dataset') / article_name).write_text(data, encoding='utf-8')
            return data
        else:
            raise Exception(f'Response: {response}')
    except Exception as err:
        raise Exception(f'While reading {url} occurred exception:\n{err}')


def write_results_to_file(char_counts, filepath='.', filename='result.txt'):
    """
    Function that writes results (char_counts) to file
    :param char_counts:
    :param filepath:
    :param filename:
    """
    with open(Path(filepath) / filename, 'w', encoding='utf-8') as result_file:
        for key, value in sorted(char_counts.items()):
            if key == '\n':
                key = '\\n'
            elif key == ' ':
                key = '<space>'
            elif key == '\t':
                key = '\\t'
            result_file.write(f'{key} {value}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Letter counter')

    # default url => http://ps2.railsreactor.net/datasets/wikipedia_articles
    parser.add_argument('--url', help='url to download dataset', required=True, type=str)
    parser.add_argument('--num_processes', help='number of processes for multiprocessing',
                        required=False, default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()
    dataset_url = args.url
    num_processes = args.num_processes
    Path('./dataset').mkdir(mode=0o777, exist_ok=True)
    articles_urls = get_articles_urls(dataset_url)

    char_counts = Counter()

    with multiprocessing.Pool(num_processes) as pool:
        article_data = pool.map(get_articles_data_and_download, articles_urls)
    for data in article_data:
        char_counts.update(list(data))

    write_results_to_file(char_counts)
