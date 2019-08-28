import argparse
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent


def parse_args():
    """
    Program argument parsing
    :return: all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Fourth homework. Models with metrics, etc.')
    parser.add_argument('--dataset',
                        help='Path or url to the dataset.',
                        type=str,
                        required=True)
    parser.add_argument('--device',
                        help='Device for training model: cpu or gpu, cpu by default',
                        type=str,
                        default='cpu')
    parser.add_argument('--batch-size',
                        help='Batch size for training model, default: 32',
                        type=int,
                        default=32)
    parser.add_argument('--optimizer',
                        help='Model training optimizer, default SGD',
                        type=str,
                        default='SGD')
    parser.add_argument('--lr',
                        help='Model training learning rate, default: 0.1',
                        type=float,
                        default=0.1)
    parser.add_argument('--epochs',
                        help='Model training epochs, default 1000',
                        type=int,
                        default=1000)
    parser.add_argument('--target', '--T',
                        help='Define target column in csv file, default "target"',
                        type=str,
                        required=True)
    parser.add_argument('--task',
                        help='Define model task: linear/regression',
                        type=str,
                        required=True)
    parser.add_argument('--batch_size',
                        help='parameter for validation split size',
                        type=int,
                        default=32)
    parser.add_argument('--categorical',
                        help='model categorical values passed as one string comma-separated without spaces',
                        type=str,
                        default=None)
    parser.add_argument('--na',
                        help='model na values that will be fit with median passed as one string comma-separated without'
                             ' spaces',
                        type=str,
                        default=None)
    parser.add_argument('--verbose', '--V',
                        help='Printing whole program log to the console.',
                        action='store_true')
    parser.add_argument('--log',
                        help='Path to log file.',
                        type=str,
                        default='model.log')

    return parser.parse_args()
