import requests
from  argparse import ArgumentParser
from pathlib import Path
from collections import Counter
import multiprocessing as mp
from functools import partial

DATA_PATH = '../data/hw_1'
LISTING = 'files.txt'
COUNTER_PATH = 'result.txt'

CHARS_MAPPING = {
    '\n': '<next_line>',
    ' ' : '<space>',
    '\t': '<tab>'
}


def download_and_count(filename: str, url: str, path: Path) -> Counter:
    req = requests.get(f'{url}/{filename}')
    if req.status_code == 200:
        req = req.content.decode('utf-8')
    else:
        return Counter()
    (path / filename).write_text(req, 'utf-8')
    return Counter(req)


def main(url: str, n_proc: int):
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    files = requests.get(f'{url}/{LISTING}')
    if files.status_code == 200:
        files = files.content.decode('utf-8').strip().split('\n')
    elif files.status_code == 404:
        raise RuntimeError(f'"{LISTING}" not found by URL {url}/{LISTING}')
    else:
        raise RuntimeError(
            (f'Something went wrong, while downloading "{LISTING}". '
            f'Response status code is {files.status_code}. URL: {url}/{LISTING}'))

    print('Loading data...')
    with mp.Pool(processes=n_proc) as pool:
        result_counter = sum(
            pool.map(partial(download_and_count, url=url, path=data_dir), files),
            Counter()
        )

    print('Writing counts...')
    with (data_dir.parent / COUNTER_PATH).open('w', encoding='utf-8') as f:
        for k, v in sorted(result_counter.items(), key=lambda x: x[1], reverse=True):
            f.write(f'{CHARS_MAPPING.get(k, k)} {v}\n')


if __name__ == "__main__":
    parser = ArgumentParser(
        description=('Count symbols in files by given URL.'
            f'Folder on URL must contain file named "{LISTING}" with list of all files'
            f'in directory except "{LISTING}".'),
            epilog='Vladyslav Rudenko')
    parser.add_argument('--url', type=str, required=True, help='url where files are located')
    parser.add_argument('--num_processes', type=int, metavar='N', default=mp.cpu_count(),
        help=f'number of processes to run (default: max available CPU cores <{mp.cpu_count()}>)')

    args = parser.parse_args()
    main(args.url, args.num_processes)
