import matplotlib.pyplot as plt
import numpy as np


def train_val_accuracy_graph(all_train_accur, all_val_accur):
    plt.figure()
    x = np.array(range(1,11))
    plt.plot(x, all_train_accur, label='Train')
    plt.plot(x, all_val_accur, label='Val')
    plt.title("Depending accuracy from max len tree", fontsize=14, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Max len tree")
    plt.legend()
    plt.grid()

    plt.savefig('Train_val_accuracy.png')
    plt.show()


def conf_matrix_graph(matrix, accur):
    plt.figure(2)
    plt.imshow(matrix, cmap="pink")
    plt.title(f"Confusion Matrix for test data, accuracy = {accur:.4}", fontsize=14, pad=15)
    plt.ylabel("Real class")
    plt.xlabel("Predict class")
    plt.xticks(range(matrix.shape[0]))
    plt.yticks(range(matrix.shape[0]))

    plt.savefig('Test_conf_matrix.png')
    plt.show()
