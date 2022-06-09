import matplotlib.pyplot as plt
import numpy as np


def bar_error_train_val(best_model_indexes, all_W, all_func_set, all_train_RMSY, all_val_RMSY, functions_name):
    labels = []
    for i in best_model_indexes:
        weights = np.array(all_W)[i]
        func_names = functions_name[np.array(all_func_set)[i]]
        labels.append(f"{weights[0]:.2f}" + "".join([f" + {w:.2f}*{func}" for w, func in zip(weights[1:], func_names)]))
        # labels.append("".join([f"{w:.2f}*{func} + " for w, func in zip(weights, func_names)]) + f"{weights[-1]:.2f}")

    plt.figure(figsize=(13, 7))
    column_pos_x = np.arange(len(best_model_indexes))
    width = 0.2
    graph_tr = plt.bar(column_pos_x - width / 2, np.array(all_train_RMSY)[best_model_indexes], width,
                       label="Min train_data error", color = 'olive')
    graph_val = plt.bar(column_pos_x + width / 2, np.array(all_val_RMSY)[best_model_indexes], width,
                        label="Min val_data error", color = 'brown')
    plt.ylabel('MSE')
    plt.title('Compare model error', fontsize='x-large')
    plt.xticks(column_pos_x, labels)
    plt.legend(loc=3)
    plt.grid()

    for rect in graph_tr + graph_val:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.show()
