import numpy as np
import sklearn
from sklearn import datasets
import collections
import matplotlib.pyplot as plt

from task7.Model_class import DecisionTree
from task7.Utils.data import distribute_data, save_to_testing
from task7.Vizualization.plt_graphs import train_val_accuracy_graph

if __name__ == '__main__':
    data, target = sklearn.datasets.load_digits(return_X_y=True)
    N = len(data)  # amount of images
    M = len(data[0])  # amount of characteristics
    K = len(collections.Counter(target))  # amount of classes

    train_x, train_t, val_x, val_t, test_x, test_t = distribute_data(data, target, 0.8, 0.1, K)

    # for graph
    train_accur_all = []
    val_accur_all = []
    for len_tree in range(1, 11):  # от 1 до 10
        tree_model = DecisionTree(
            K, max_len_tree=len_tree, min_len_leaf_node=1,
            min_entropy_node=0, num_split_feature=10
        )
        tree_model.training_simple(train_x, train_t)
        train_y = [tree_model.predict(train_x_i) for train_x_i in train_x]
        train_accur_all.append(tree_model.calc_accuracy(train_y, train_t))

        val_y = [tree_model.predict(val_x_i) for val_x_i in val_x]
        val_accur_all.append(tree_model.calc_accuracy(val_y, val_t))
    train_val_accuracy_graph(train_accur_all, val_accur_all)

    # validation
    best_tree_model = None
    best_tree_accur = 0
    for len_tree in range(1, 11):  # от 1 до 10
        for len_leaf in range(2, 9, 2):  # от 2 до 8 с шагом 2
            for entropy in np.arange(0, 0.4, 0.1):  # от 0 до 0.3 с шогом 0.1
                tree_model = DecisionTree(
                    K, max_len_tree=len_tree, min_len_leaf_node=len_leaf,
                    min_entropy_node=entropy, num_split_feature=10
                )
                tree_model.training_simple(train_x, train_t)
                # print("train", len_tree, len_leaf, entropy)

                val_y = [tree_model.predict(val_x_i) for val_x_i in val_x]
                val_accur = tree_model.calc_accuracy(val_y, val_t)
                if val_accur > best_tree_accur:
                    best_tree_accur = val_accur
                    best_tree_model = tree_model

    print('best_tree_accur=', best_tree_accur)
    tree_file_name = best_tree_model.save_tree()
    save_to_testing(test_x, test_t, K, tree_file_name, 'Test_data.pickle')

    # чтобы посмотреть на какую картинку что предсказывает
    # test_idx = 5
    # pred_y = tree_model.predict(test_x[test_idx])
    #
    # plt.imshow(test_x[test_idx].reshape(8, 8, 1), cmap='binary')
    # plt.title(f"pred - {pred_y} target - {test_t[test_idx]}")
    # plt.show()
