import argparse
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

import requests


def download_article(url):
    name = Path(url).name
    try:
        r = requests.get(url)
        with open(name, 'w', encoding='utf-8') as f:
            f.write(r.text)
        return Counter(r.text)
    except requests.exceptions.RequestException as e:
        print(f'Request Error: Failed to get {name}')
        return Counter()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Symbol counter')
    parser.add_argument('--url', help='Dataset url')
    parser.add_argument('--num_processes', type=int, help='Number of processes for parallelization', default=4)
    args = parser.parse_args()
    url = args.url
    try:
        files = requests.get(requests.compat.urljoin(url, 'files.txt')).text.split()
        urls = list(map(lambda x: requests.compat.urljoin(url, x), files))
        symbol_counts = Counter()
        with Pool(args.num_processes) as pool:
            counters = pool.map(download_article, urls)
        symbol_counts = sum(counters, Counter())
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write('\\n {}\n'.format(symbol_counts["\n"]))
            f.write(f' (space) {symbol_counts[" "]}\n')
            del symbol_counts['\n']
            del symbol_counts[' ']
            for i in symbol_counts:
                f.write(f'{i} {symbol_counts[i]}\n')
    except requests.exceptions.RequestException as e:
        print('Request Error: Failed to get the article list')
        
    

