import numpy as np
import math
import pickle

from task7.Node_class import TreeNode


class DecisionTree:
    def __init__(self, k_classes, max_len_tree=None, min_len_leaf_node=None, min_entropy_node=None, num_split_feature=None):
        self.child_count = 2
        self.root = None
        self.k_classes = k_classes

        self.max_len_tree = max_len_tree
        self.min_len_leaf_node = min_len_leaf_node
        self.min_entropy_node = min_entropy_node
        self.num_split_feature = num_split_feature

    def entropy(self, t):
        N = len(t)
        N_k_classes = [np.sum(t == k) for k in range(self.k_classes)]
        return - np.sum([((N_k / N) * math.log(N_k / N, self.k_classes) if N_k else 0) for N_k in N_k_classes])

    def training_simple(self, train_x, train_t):
        self.root = self.create_new_node(train_x, train_t, 0)

    def create_new_node(self, x, t, len_tree):
        vec_probabil_K_classes = [np.sum(t == k) / len(t) for k in range(self.k_classes)]

        if len_tree >= self.max_len_tree:  # проверка на max глубину
            return TreeNode(None, None, vec_probabil_K_classes, True, None, None)
        if len(t) < self.min_len_leaf_node:  # проверка на мин кол-во эл-тов в узле
            return TreeNode(None, None, vec_probabil_K_classes, True, None, None)
        if self.entropy(t) < self.min_entropy_node:  # проверка на мин энтропию в узле
            return TreeNode(None, None, vec_probabil_K_classes, True, None, None)

        feature_index, feature_border = self.find_split(x, t)
        left_child_x, left_child_t, right_child_x, right_child_t = self.split_data(x, t, feature_index, feature_border)

        left_child = self.create_new_node(left_child_x, left_child_t, len_tree + 1)
        right_child = self.create_new_node(right_child_x, right_child_t, len_tree + 1)

        return TreeNode(feature_index, feature_border, vec_probabil_K_classes, False, left_child, right_child)

    def split_data(self, x, t, feature_index, feature_border):
        X_slice_one_feature = x[:, feature_index]
        sort_indexes = np.argsort(X_slice_one_feature)
        x = x[sort_indexes]
        t = t[sort_indexes]
        index_split = np.sum(X_slice_one_feature <= feature_border)
        return x[:index_split], t[:index_split], x[index_split:], t[index_split:]

    def find_split(self, x, t):
        max_I = 0
        max_I_feature_index = 0
        max_I_feature_border = 0

        N_i = x.shape[0]
        H_Si = self.entropy(t)

        for feature_index in range(x.shape[1]):  # перебор харак-к
            X_slice_one_feature = x[:, feature_index]
            all_feature_borders = np.linspace(min(X_slice_one_feature), max(X_slice_one_feature),
                                              self.num_split_feature)

            for border in all_feature_borders:  # перебор тао для данной харак-ки
                mask_border_down = X_slice_one_feature <= border
                N_left = np.sum(mask_border_down)
                mask_border_up = X_slice_one_feature > border
                N_right = np.sum(mask_border_up)

                H_Sij = ((N_left / N_i) * self.entropy(t[mask_border_down])) + \
                        ((N_right / N_i) * self.entropy(t[mask_border_up]))
                Inf_gain = H_Si - H_Sij

                if Inf_gain > max_I:
                    max_I = Inf_gain
                    max_I_feature_index = feature_index
                    max_I_feature_border = border

        return max_I_feature_index, max_I_feature_border

    def training_vector(self):  # ждет лучших времен, как и я
        pass

    def predict(self, x_i):
        current_node = self.root

        while not current_node.is_terminal:
            feature = x_i[current_node.split_feature_index]
            if feature <= current_node.split_feature_border:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        return np.argmax(current_node.vec_probabil_K_class)

    def calc_accuracy(self, predict, t):
        return np.sum(predict == t)/len(predict)

    def save_tree(self):
        file_name = f'bestTree_maxLenTree_{self.max_len_tree}_minLenLeaf_{self.min_len_leaf_node}' \
                    f'_minEntr_{self.min_entropy_node}.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(self.root, f)
        return file_name

    def load_tree(self, file_name):
        with open(file_name, 'rb') as f:
            self.root = pickle.load(f)
