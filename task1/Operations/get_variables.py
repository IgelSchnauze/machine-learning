import numpy as np


def get_stochastic_var(mu, sigma, N):
    variables = np.random.normal(mu, sigma, N)
    return variables
