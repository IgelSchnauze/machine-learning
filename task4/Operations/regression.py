import numpy as np
from numpy.linalg import inv


def design_mtrx_polinom(vec_x, degree):
    return np.array([[x ** i for i in range(degree)] for x in vec_x])


def count_W(design_m, vec_t, lamb):
    # return inv(np.dot(design_m.T, design_m)).dot(design_m.T).dot(vec_t)
    return inv(np.dot(design_m.T, design_m) + np.eye(design_m.shape[1]) * lamb).dot(design_m.T).dot(vec_t)


def count_regress(W, design_m):
    return np.dot(W.T, design_m.T)


def MSE(vec_t, vec_y):
    return (1/len(vec_t)) * np.sum(np.power((vec_t - vec_y), 2))


def loss(vec_t, vec_y):
    return 0.5 * np.sum(np.power((vec_t - vec_y), 2))


def gradient(vec_t, design_m, W, lamb):
    return -1 * np.dot(vec_t - count_regress(W, design_m), design_m) + lamb*W


def gradient_descent(w_prev, step, gradient):
    return w_prev - step * gradient
