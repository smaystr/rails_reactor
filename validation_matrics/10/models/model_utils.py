import yaml
from pathlib import Path

config_file_path = Path(Path(__file__).parent) / 'config.yml'

with open(config_file_path, 'r', encoding='utf-8') as stream:
    config = yaml.safe_load(stream)

