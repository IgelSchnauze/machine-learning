import numpy as np
from random import getrandbits


def binary_class_rand(vec_x):
    return np.array([getrandbits(1) for _ in vec_x])


def binary_class_threshold(vec_x, tao):
    res = np.zeros(len(vec_x))
    res[vec_x > tao] = 1
    return res
