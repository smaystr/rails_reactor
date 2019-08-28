import argparse
from os import environ

from settings import load_environment


def argument_namespace():
    parser = argparse.ArgumentParser(
        description='Web Application'
    )
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        help='The state of the program, which will be defined by the filename of the environment file.'
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='Path to the environment file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_namespace()
    load_environment(
        path=args.path,
        filename=args.filename
    )
    from project.app import run
    run()