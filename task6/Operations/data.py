import numpy as np


def distribute_data(x, t, share_tr, share_val, K):
    train_x = []
    train_t = []
    val_x = []
    val_t = []
    test_x = []
    test_t = []
    for k in range(K):
        class_mask = t == k
        n1 = int(sum(class_mask) * share_tr)
        n2 = int(sum(class_mask) * share_val)

        train_x.extend(x[class_mask][0: n1])
        train_t.extend(t[class_mask][0: n1])
        val_x.extend(x[class_mask][n1: n1 + n2])
        val_t.extend(t[class_mask][n1: n1 + n2])
        test_x.extend(x[class_mask][n1 + n2:])
        test_t.extend(t[class_mask][n1 + n2:])

    return np.array(train_x), np.array(train_t), np.array(val_x), np.array(val_t), np.array(test_x), np.array(test_t)


def normalization(vec_x):
    min_x = np.min(vec_x)
    max_x = np.max(vec_x)
    return np.array([(2 * (x_i - min_x) / (max_x - min_x)) for x_i in vec_x])


def to_one_hot(vec_t, K):
    I = np.eye(K)
    return np.array([I[t_i] for t_i in vec_t])


def not_one_hot(vec_t):
    return np.array([np.where(t_i == 1)[0] for t_i in vec_t]).squeeze()  # squeeze() у [] из одного эл-та уберет []
