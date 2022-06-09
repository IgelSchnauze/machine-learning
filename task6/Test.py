from task6.Model import ModelTrainer
from task6.Operations.metrics import calc_precision, calc_recall, conf_matrix
from task6.Utils.with_pickle import load_from
from task6.Visualization.plt_graphs import conf_matrix_graph, prec_rec_bar_graph

if __name__ == '__main__':
    test_x, test_t, K_classes = load_from('Test_data.pickle')

    model = ModelTrainer(len(test_x[0]), K_classes)
    model.load_weights_from('Best_weight.pickle')

    Z = model.regress_all_objects(test_x)
    prec_per_class = []
    rec_per_class = []
    # choose predictions to objects from 1 class
    for K in range(K_classes):
        prec_per_class.append(calc_precision(test_t, Z, K))
        rec_per_class.append(calc_recall(test_t, Z, K))

    prec_rec_bar_graph(prec_per_class, rec_per_class, K_classes)
    Conf_matrix = conf_matrix(test_t, Z, K_classes)
    conf_matrix_graph(Conf_matrix)
