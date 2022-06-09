import numpy as np

from task5.Operations.classification import binary_class_rand, binary_class_threshold
from task5.Operations.metrics import calc_TP, calc_TN, calc_FP, calc_FN, calc_accuracy, calc_precision, calc_recall, \
    calc_average_precision
from task5.Visualization.plt_graphics import prec_rec_graph

if __name__ == '__main__':
    N = 500
    x0 = np.random.randn(N) * 20 + 160
    x1 = np.random.randn(N) * 10 + 190
    t0 = np.zeros(N)
    t1 = np.ones(N)
    x_ = np.concatenate((x0, x1))
    t_ = np.concatenate((t0, t1))

    treshhold_height = 190
    y_predict_rand = binary_class_rand(x_)
    y_predict_height = binary_class_threshold(x_, treshhold_height)

    for y_, name_y in zip([y_predict_rand, y_predict_height], ["random", "treshhold"]):
        tp = calc_TP(t_, y_)
        tn = calc_TN(t_, y_)
        fp = calc_FP(t_, y_)
        fn = calc_FN(t_, y_)
        acc = calc_accuracy(t_, y_)
        prec = calc_precision(t_, y_)
        rec = calc_recall(t_, y_)
        print("Prediction with", name_y)
        print(f"TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}, Accuracy = {acc:.3}, Precision = {prec:.3}, Recall = {rec:.3}")

    vec_height = range(90, 231, 10)
    all_prec = [calc_precision(t_, binary_class_threshold(x_, height)) for height in vec_height]
    all_rec = [calc_recall(t_, binary_class_threshold(x_, height)) for height in vec_height]

    aver_prec = calc_average_precision(all_prec, all_rec)
    prec_rec_graph(all_prec, all_rec, vec_height, aver_prec)
