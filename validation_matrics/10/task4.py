import argparse
from utils import split_to_X_and_y, get_dataset, get_fold, randomize, get_scores
from loocv import loocv
from train_test_split import train_test_split
from k_fold import k_fold
import time
from pathlib import Path


output_path = Path(Path(__file__).parent) / 'output'
output_path.mkdir(exist_ok=True)


def write_output_info(metrics, used_time, feature_importance):
    with open(output_path / 'output.info', 'w+', encoding='utf-8') as info_file:
        content = ''
        content += '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        content += f'\ntraining time is: {used_time}'
        if (feature_importance):
            content += 'Feature importance'
            content += '\n'.join([f'{key}: {value}' for key, value in feature_importance.items()])

        info_file.write(content)


def write_output_model(task, coefs, best_hyperparameters):
    with open(output_path / 'output.model', 'w+', encoding='utf-8') as model_file:
        content = f'Task: {task}\n'
        content += f'Coefs:\n{coefs}]'
        if (best_hyperparameters):
            content += 'Best hyperparameters'
            content += '\n'.join([f'{key}: {value}' for key, value in best_hyperparameters.items()])

        model_file.write(content)


def main(args):
    df = randomize(get_dataset(args.dataset_path))

    mapSplitTypeToValidator = {
        'train-test-split': train_test_split,
        'k-fold': k_fold,
        'loocv': loocv,
    }
    validator = mapSplitTypeToValidator[args.split_type]


    start_time = time.time()
    coefs, metrics = validator(args.task, df, args.target, args.use_gpu)
    used_time = time.time() - start_time
    write_output_info(metrics, used_time, None)
    write_output_model(args.task, coefs, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training utility')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--task', required=True, choices=['classification', 'regression'])
    parser.add_argument('--split_type', default='train-test-split', choices=['train-test-split', 'k-fold', 'loocv'])
    parser.add_argument('--split_size', default=0.1)
    parser.add_argument('--time')
    parser.add_argument('--fitting_algo', choices=['grid', 'random'])
    parser.add_argument('--use_gpu', choices=[True, False], default=False, type=bool)
    args = parser.parse_args()
    main(args)
