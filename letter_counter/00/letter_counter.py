import time
import logging
import multiprocessing as mp
import re

from functools import wraps
from requests import Session
from pathlib import Path
from urlpath import URL
from clint.textui import progress
from argparse import ArgumentParser
from collections import Counter

logger = logging.getLogger(__name__)

data_path = Path('data')
flist_name = 'files.txt'
res_name = 'result.txt'


def download_file(url: str):
    with Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()
        with open(data_path.joinpath(url.name), 'wb') as downloaded_file:
            total_length = int(response.headers.get('content-length'))
            for chunk in progress.bar(response.iter_content(chunk_size=1024),
                                      expected_size=(total_length / 1024) + 1,
                                      label='downloading {}'.format(url.name)):
                if chunk:
                    downloaded_file.write(chunk)
                    downloaded_file.flush()


# it's easier to manage the data set from the list
def path_handler(_path):
    with open(data_path.joinpath(flist_name), 'r') as reader:
        return [_path.joinpath(str(line).strip()) for line in reader]


def get_char_counts(_path: str):
    char_counts = Counter()
    with open(_path, 'r') as file:
        [char_counts.update(re.sub("[\W\-_\d]", "", line.lower())) for line in file]
        return char_counts


def get_arguments():
    parser = ArgumentParser(description='Character counter for the provided dataset')
    parser.add_argument("--num_processes", type=int, metavar="NP",
                        default=1, help="number of processes")
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument("--url", type=str, metavar="URL",
                                default=None, help="url to dataset", required=True)
    return parser.parse_args()


def measure_time(func):
    logger.info(f"{func.__name__} starts ...")

    @wraps(func)
    def _wrapper_time(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return result

    return _wrapper_time


@measure_time
def download_processing(num_proc, path):
    with mp.Pool(num_proc) as pool:
        pool.map(download_file, path_handler(path))


@measure_time
def data_processing(num_proc, path):
    char_counts = Counter()
    with mp.Pool(num_proc) as pool:
        [char_counts.update(counter) for counter in pool.map(get_char_counts, path_handler(path))]

    with open(data_path.joinpath(res_name), 'w', encoding="utf-8") as writer:
        for key, value in dict(sorted(char_counts.items())).items():
            writer.writelines(f"{key} {value} \n")


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = get_arguments()
    download_processing(args.num_processes, URL(args.url))
    data_processing(args.num_processes, data_path)


if __name__ == '__main__':
    main()
