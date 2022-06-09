import numpy as np


def normal_var(mu, sigma, N):
    return np.random.normal(mu, sigma, N)


def equidist_var(a, b, N):
    return np.linspace(a, b, N)


def some_formula(vec):
    return 100*np.sin(vec) + 0.5*np.e**(vec) + 300