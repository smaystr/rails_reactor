import itertools
import numpy as np

def product_dict(param_dict):
    keys = param_dict.keys()
    vals = param_dict.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output
