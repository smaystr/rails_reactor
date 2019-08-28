import argparse
import requests
import multiprocessing
from pathlib import Path
import os
import re
from collections import Counter
from time import time
from urllib.parse import urljoin


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="url with articles.", type=str)
    parser.add_argument(
        "--num_processes",
        help="number of processes to use.",
        type=int,
        default=multiprocessing.cpu_count(),
    )

    return parser.parse_args()


def download_text(url):
    if Path("downloads/" + url[0]).exists():
        return
    try:
        text = requests.get(url[1])
        with open("downloads/" + url[0], "w+") as f:
            f.write(text.text)
    except requests.exceptions.RequestException as e:
        print(f"Cannot download {url[0]}")


def get_articles(path):
    if Path(path).exists():
        return [x.split()[0] for x in open(path, "r").readlines()]
    Exception("files.txt is not present.")


def process_file(file):
    wanted = re.compile(
        "([^-_a-zA-Z0-9!@#%&=,/'\";:~`\$\^\*\(\)\+\[\]\.\{\}\|\?\<\>\\]+|[^\s]+)"
    )
    with open("downloads/" + file) as file:
        content = Counter(list(wanted.sub(" ", file.read())))
    return content


def write_result(result_file, letters):
    with open(result_file, "w+") as out:
        for key, val in letters.items():
            out.write("%s %s\n" % (key, val))


def merge_counts(counts):
    super_counter = Counter()
    for c in counts:
        super_counter.update(c)

    return super_counter


def download_process(url_files):
    download_text(url_files)
    return process_file(url_files[0])


def run():
    start = time()
    args = arguments()
    num_processes = args.num_processes
    url = args.url

    if not Path("downloads").exists():
        os.makedirs("downloads")

    file_url = [(file, urljoin(url, file)) for file in get_articles("files.txt")]

    with multiprocessing.Pool(num_processes) as pool:
        res = pool.map(download_process, file_url)

    write_result("result.txt", merge_counts(res))
    print(f"Done in {int(time()-start)} seconds.")


if __name__ == "__main__":
    run()
