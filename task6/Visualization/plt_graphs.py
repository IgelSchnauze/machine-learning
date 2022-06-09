import matplotlib.pyplot as plt
import numpy as np


def loss_graph(all_loss):
    plt.figure()
    plt.plot(all_loss)
    plt.title(f"Loss depend on iteration, min = {np.min(np.array(all_loss)):.5}", fontsize = 14, pad=20)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    plt.savefig('Loss.png')
    plt.show()


def val_accuracy_graph(all_accur, step_for_val):
    plt.figure()
    x = np.array(range(0, len(all_accur) * step_for_val, step_for_val))
    plt.plot(x, all_accur)
    plt.title("Accuracy on validation data", fontsize=14, pad=20)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.grid()

    plt.savefig('Val_accuracy.png')
    plt.show()


def prec_rec_bar_graph(prec_per_class, rec_per_class, K_classes):
    plt.figure(1, figsize=(14, 7))
    column_pos_x = np.arange(K_classes)
    width = 0.3
    graph_tr = plt.bar(column_pos_x - width / 2, prec_per_class, width, label="Precision", color='pink')
    graph_val = plt.bar(column_pos_x + width / 2, rec_per_class, width, label="Recall", color='lavender')
    plt.xlabel('Class')
    plt.title('Check metrics per class', fontsize='x-large', pad=20)
    plt.xticks(column_pos_x, range(K_classes))
    plt.legend(loc=3)

    for rect in graph_tr + graph_val:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize='small')

    plt.savefig('Test_prec_rec.png')
    # plt.show()


def conf_matrix_graph(matrix):
    plt.figure(2)
    plt.imshow(matrix, cmap="pink")  # YlGn
    plt.title("Confusion Matrix for test data", fontsize=14, pad=15)
    plt.ylabel("Real class")
    plt.xlabel("Predict class")
    plt.xticks(range(matrix.shape[0]))
    plt.yticks(range(matrix.shape[0]))

    plt.savefig('Test_conf_matrix.png')
    plt.show()
