import logging
from pathlib import Path

from dotenv import load_dotenv


def load_env():
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)


def set_up_logging(log_file: str, verbose: bool):
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        filename=log_file,
                        filemode='a')
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    logging.info('ARGS PARSED, LOGGING CONFIGURED.')
