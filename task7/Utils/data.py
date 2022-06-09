import pickle
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


def save_to_testing(x, t, k_classes, tree_file_name, to_file):
    with open(to_file, 'wb') as f:
        pickle.dump((x, t, k_classes, tree_file_name), f)


def load_to_testing(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data