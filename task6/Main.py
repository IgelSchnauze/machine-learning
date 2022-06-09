import sklearn
from sklearn import datasets
import collections

from task6.Model import ModelTrainer
from task6.Operations.data import distribute_data, normalization, to_one_hot, not_one_hot
from task6.Utils.with_pickle import save_to, load_from
from task6.Visualization.plt_graphs import loss_graph, val_accuracy_graph


if __name__ == '__main__':
    data, target = sklearn.datasets.load_digits(return_X_y=True)
    N = len(data)  # amount of images
    M = len(data[0])  # amount of characteristics
    K = len(collections.Counter(target))  # amount of classes

    norm_data = normalization(data)
    train_x, train_t, val_x, val_t, test_x, test_t = distribute_data(norm_data, target, 0.8, 0.1, K)
    train_t = to_one_hot(train_t, K)

    model = ModelTrainer(M, K)
    step_count = N * 30
    step_for_save_and_val = int(step_count / 100)
    all_loss, val_accuracy = model.training(train_x, train_t, val_x, val_t, step_count, step_for_save_and_val)
    model.save_weights_to('Best_weight.pickle')

    save_to((test_x, test_t, K), 'Test_data.pickle')

    loss_graph(all_loss)
    val_accuracy_graph(val_accuracy, step_for_save_and_val)
