import numpy as np


def normal_var(mu, sigma, N):
    return np.random.normal(mu, sigma, N)


def equidist_var(a, b, N):
    return np.linspace(a, b, N)


def some_formula(vec):
    return 100*np.sin(vec) + 0.5*np.e**(vec) + 300


def distribute_data(x, t, share_tr, share_val):
    # maybe use np.random.choice
    rand_i = np.random.permutation(len(x))
    n1 = int(len(x) * share_tr)
    n2 = int(len(x) * share_val)
    return x[rand_i[0: n1]], t[rand_i[0: n1]],\
           x[rand_i[n1: n1 + n2]], t[rand_i[n1: n1 + n2]], \
           x[rand_i[n1 + n2:]], t[rand_i[n1 + n2:]]
