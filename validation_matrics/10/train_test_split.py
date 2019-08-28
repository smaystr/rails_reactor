from utils import split_to_X_and_y, get_dataset, get_fold, randomize, get_scores
from process import process


def train_test_split(task, df, target_name, use_gpu, split_size=0.1):
    fold = list(get_fold(df, round(1 / split_size)))[0]
    train, test = fold
    coef, scores = process((task, target_name, (train, test), use_gpu))

    return coef, scores
