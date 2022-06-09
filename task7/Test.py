import numpy as np
from tqdm.auto import tqdm

from task7.Model_class import DecisionTree
from task7.Utils.data import load_to_testing
from task7.Vizualization.plt_graphs import conf_matrix_graph


def make_conf_matrix(vec_t, vec_predict, K):
    C = np.zeros((K, K))
    for t, y in zip(vec_t, vec_predict):
        C[t, y] += 1
    return C


if __name__ == '__main__':
    test_x, test_t, K_classes, tree_file_name = load_to_testing('Test_data.pickle')
    tree_model = DecisionTree(K_classes)
    tree_model.load_tree(tree_file_name)

    test_y = [tree_model.predict(test_x_i) for test_x_i in tqdm(test_x)]
    test_accur = tree_model.calc_accuracy(test_y, test_t)
    # print("Accuracy on test data = ", test_accur)

    CM = make_conf_matrix(test_t, test_y, K_classes)
    conf_matrix_graph(CM, test_accur)