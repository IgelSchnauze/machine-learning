import numpy as np
from numpy.linalg import inv


def design_mtrx_polinom(vec_x, degree):
    return np.array([[x ** i for i in range(degree)] for x in vec_x])


def count_W(design_m, vec_t):
    return inv(np.dot(design_m.T, design_m)).dot(design_m.T).dot(vec_t)


def polinom_regres(W, DM):
    return np.dot(W.T, DM.T)


# def polinom_regres_one_x(W, basis_f_vec):
#     return np.dot(W.T, basis_f_vec)

