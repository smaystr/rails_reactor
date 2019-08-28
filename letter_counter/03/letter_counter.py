import requests
import pathlib
import multiprocessing
import argparse
from urllib.parse import urljoin

from collections import Counter
from functools import reduce, partial
from operator import add, itemgetter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://ps2.railsreactor.net/datasets/wikipedia_articles/",
        help="""Base url for parsing the file list; 
                        files.txt will be automatically appended to the base path""",
        type=str,
    )

    parser.add_argument(
        "--num_processes",
        default=multiprocessing.cpu_count(),
        help="""Number of processes to use simultaneously during the parsing of pages content""",
        type=int,
    )

    parser.add_argument(
        "--output_file",
        default="results.txt",
        help=""" Path of the file where results will be written.""",
        type=str,
    )

    parser.add_argument(
        "--verbose",
        default=1,
        help=""" Set verbosity flag. If 1, url of the pages which are 
                        being parsed printed to the console. 0 otherwise""",
        type=int,
    )

    args = parser.parse_args()

    return args


def parse_file(url, verbose=True):
    if verbose:
        print(f"parsing {url}")
    try:
        data = requests.get(url).text
    except Exception as e:
        print(f"Caught exception {e} when parsing {url}")
        return Counter()
    return Counter(data)


def write_to_file(counter_merged, file_name="result.txt"):
    file_name = pathlib.Path(file_name)
    if file_name.exists():
        file_name.unlink()
    for char, count in counter_merged:
        with file_name.open(mode="a") as f:
            f.write(repr(char) + " " + str(count) + "\n")


def letter_counter(base_url, num_processes, file_name, verbose):

    base_full_path = urljoin(base_url, "files.txt")
    try:
        file_names = requests.get(base_full_path).text.split()
    except Exception as e:
        print(f"Exception when trying to parse file names :\n{e}")
        return

    file_urls = [urljoin(base_url, file_name) for file_name in file_names]

    with multiprocessing.Pool(processes=num_processes) as p:
        counters = p.map(partial(parse_file, verbose=verbose), file_urls[:10])

    final_counter = reduce(add, counters)
    final_counter_sorted = sorted(
        final_counter.items(), key=itemgetter(1), reverse=True
    )

    write_to_file(final_counter_sorted, file_name)

def main():
    args = parse_args()
    letter_counter(args.url, args.num_processes, args.output_file, bool(args.verbose))

if __name__ == "__main__":
    main()
