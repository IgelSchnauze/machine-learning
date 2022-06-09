import numpy as np
from numpy.linalg import inv


def design_mtrx_polinom(vec_x, degree):
    return np.array([[x ** i for i in range(degree)] for x in vec_x])


def design_mtrx_func(vec_x, func):
    return np.array([f(vec_x) for f in func]).T


def count_W(design_m, vec_t):
    return inv(np.dot(design_m.T, design_m)).dot(design_m.T).dot(vec_t)


def count_regress(W, design_m):
    return np.dot(W.T, design_m.T)
