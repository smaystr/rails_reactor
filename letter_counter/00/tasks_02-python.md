# Homework 2

You need to implement the character counter for the provided dataset. Dataset is available [here](http://ps2.railsreactor.net/datasets/wikipedia_articles/) and contains a lot of txt files. The list of all `txt` files could be found in `files.txt` by [link](http://ps2.railsreactor.net/datasets/wikipedia_articles/files.txt). The solution should be console utility that accepts the `url` to dataset and number of processes as parameters.
The dataset should be downloaded and saved into a separate folder and don't forget to ignore `files.txt`. We encourage you to think about parallelization approach. The output is the `result.txt` file. Each row of the file is symbol/letter and counter divided by space. Commit **only your `py` files** (please don't commit dataset folder) which are necessary to run the code.

Please use the following libs:
- `requests` to download the dataset (set of files)
- `argparse` for params parsing
- `pathlib` for handling the dataset folder structure
- `collections` for letter counting
- `multiprocessing` for multi cores utilization


So, your script execution should look like
```
python letter_counter.py --url https://dataset.com/rr_ml_ss --num_processes 4
```

Output (`result.txt`):
```
a 7782212
b 89123
c 123123
...
```

## Deadline - next Thursday 04.07.2019

## Please ask all questions regarding homework in Discord
