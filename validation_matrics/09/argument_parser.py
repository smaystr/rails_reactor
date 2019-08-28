from argparse import ArgumentParser
from pathlib import Path
import requests
import numpy as np
import preprocessing as prep

PENALTY = [None, 'l1', 'l2']
LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
EPOCH_GRID = [10, 100, 500, 1000, 10000, 100000]
C_GRID = [0.001, 0.01, 0.1, 1, 10]

LR_RAND = ['uniform', 1e-3, 1, 50]
EPOCH_RAND = ['uniform', 10, 1000, 50]
C_RAND = ['uniform', 0.01, 100, 50]


def get_parser():
    parser = ArgumentParser(description='End-to-end model training lifecycle util')
    parser.add_argument('--dataset', type=str, required=True, help='Path to local .csv or valid url')
    parser.add_argument('--target', type=str, required=True, help='dataset target variable name')
    parser.add_argument('--task', type=str, required=True,
                        choices=['classification', 'regression'], help='order type of linear model')
    parser.add_argument('--output', type=str, required=True, help='path to folder with output.info and output.model')
    parser.add_argument('--validation', type=str, default='train-test',
                        choices=['train-test', 'k-fold', 'leave-one-out'],
                        help='cross-validation name, "train-test" by default')
    parser.add_argument('--test_size', type=float, default=0.2, help='split size parameter, 0.2 by default')
    parser.add_argument('time_series_column', type=str, help='time series column for timeseries validation')
    parser.add_argument('--param_search', default='grid', choices=['grid', 'random'],
                        help='hyperparameter searching algorithm, grid by default')
    parser.add_argument('n_iter', type=int, default=10, help='Number of iterations, it is 10 by default')
    parser.add_argument('--lr', nargs='*', help=f'parameters to search from for GridSearch, {LR_GRID} by default'
                        f'or {LR_RAND} for RandomSearch)')
    parser.add_argument('--epoch', nargs='*', help='Number of iterations, 10 by default')
    parser.add_argument('--penalty', nargs='*', default=PENALTY,
                        help=f'parameter to search penalty, by default is {PENALTY}')
    parser.add_argument('--C', nargs='*', help=f'parameters to search from for GridSearch, {C_GRID} by default'
                        f'or {C_RAND} for RandomSearch)')
    return parser


def process_parser_args(args):
    if not args.C:
        if args.param_search == 'grid':
            args.C = C_GRID if args.param_serach == 'grid' else C_RAND
    if not args.lr:
        args.lr = LR_GRID if args.lr == 'grid' else LR_RAND
    if not args.epoch:
        args.epoch = EPOCH_GRID if args.param_search == 'grid' else EPOCH_RAND
    return args


def get_arguments():
    return process_parser_args(get_parser().parse_args())
