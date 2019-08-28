from scipy.stats import expon, uniform
from metrics import *


CLASSIFICATION_METRICS = [precision, recall, f1_score]
REGRESSION_METRICS = [mape, rmse]

CARDINALITY_THRESHOLD = 0.05

#1,.1,.01 and so on
learning_rate_distribution_discrete = list(np.float_power(10,-np.arange(5)))
learning_rate_distribution_continuous = expon(scale=0.1)

c_dist_discrete = np.float_power(10, np.arange(6)-3)

num_iterations_discrete = list(1000*np.arange(5)[1:])
num_iterations_continuous = uniform(loc=1000,
                                    scale=10000)


CLASS_GRID = {"C":c_dist_discrete,
              "num_iterations": num_iterations_discrete,
              "learning_rate": learning_rate_distribution_discrete}

CLASS_RAND = {"C":c_dist_discrete,
              "num_iterations": num_iterations_continuous,
              "learning_rate": learning_rate_distribution_continuous}

REG_GRID = {k.replace('C', 'lam'): v for k, v in CLASS_GRID.items()}
REG_RAND = {k.replace('C', 'lam'): v for k, v in CLASS_RAND.items()}







