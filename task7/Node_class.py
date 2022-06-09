

class TreeNode:
    def __init__(self, split_feature_index, split_feature_border, vec_probabil_K_class, is_terminal, left_child = None, right_child = None):
        self.is_terminal = is_terminal
        self.left_child = left_child
        self.right_child = right_child

        self.split_feature_index = split_feature_index
        self.split_feature_border = split_feature_border
        self.vec_probabil_K_class = vec_probabil_K_class

    # def is_node_terminal(self):
    #     return self.left_child is None and self.right_child is None
