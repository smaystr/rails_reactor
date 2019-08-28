import multiprocessing
from tqdm import tqdm
from utils import get_fold
from process import process
from functools import reduce
import torch


def k_fold(task, df, target_name, use_gpu, k=10):
    params = [(task, target_name, fold, use_gpu) for fold in get_fold(df, k)]
    with torch.multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        clfs = list(tqdm(pool.imap(process, params), total=k))

    mean_coefs = torch.mean(torch.stack([coef for coef, scores in clfs]))
    scores = [scores for coef, scores in clfs]

    sum_scores = reduce(lambda x, y: {key: x[key] + y[key] for key in x.keys()}, scores)
    mean_scores = {key: value / k for key, value in sum_scores.items()}

    return mean_coefs, mean_scores
