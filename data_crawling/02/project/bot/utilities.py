from pathlib import Path
from dotenv import load_dotenv


def find(feature, dictionary):
    """
    Finding the feature value in dict
    :type feature: str
    :type dictionary: dict
    """
    if not isinstance(dictionary, dict):
        return
    for key, value in dictionary.items():
        if key == feature:
            yield value
        elif isinstance(value, dict):
            for result in find(feature, value):
                yield result
        elif isinstance(value, list):
            for inner_dict in value:
                for result in find(feature, inner_dict):
                    yield result


def find_feature_in(
        generator_obj,
        feature,
        feature_key='name',
        feature_value='value'
):
    """
    Finding the feature in the list of dicts
    :type generator_obj: generator
    :type feature: str
    :type feature_key: str
    :type feature_value: str
    """
    for item in next(generator_obj, "STOP"):
        if not isinstance(item, dict):
            return None
        if item[feature_key] == feature:
            return item[feature_value]
    return None


def load_environment(
        path,
        filename
):
    """
    Load the environment due to the mentioned state.
    :type path: str
    :type filename: str
    """
    print("LOADING ENVIRONMENT")
    env_path = Path('.') / f'{path}' / f'{filename}'
    load_dotenv(
        dotenv_path=env_path,
        verbose=True
    )