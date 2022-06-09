import numpy as np


def calc_TP(vec_t, vec_predict):
    mask_true_answer = vec_t == vec_predict
    mask_ones = vec_predict == 1
    mask_tp = np.logical_and(mask_true_answer, mask_ones)
    return sum(mask_tp)


def calc_TN(vec_t, vec_predict):
    mask_true_answer = vec_t == vec_predict
    mask_zero = vec_predict == 0
    mask_tn = np.logical_and(mask_true_answer, mask_zero)
    return sum(mask_tn)


def calc_FP(vec_t, vec_predict):
    mask_false_answer = vec_t != vec_predict
    mask_ones = vec_predict == 1
    mask_fp = np.logical_and(mask_false_answer, mask_ones)
    return sum(mask_fp)


def calc_FN(vec_t, vec_predict):
    mask_false_answer = np.logical_not(vec_t == vec_predict)
    mask_zero = vec_predict == 0
    mask_fn = np.logical_and(mask_false_answer, mask_zero)
    return sum(mask_fn)


def calc_accuracy(vec_t, vec_predict):
    tp = calc_TP(vec_t, vec_predict)
    tn = calc_TN(vec_t, vec_predict)
    return (tp + tn) / max(1, len(vec_t))


def calc_precision(vec_t, vec_predict):
    tp = calc_TP(vec_t, vec_predict)
    fp = calc_FP(vec_t, vec_predict)
    return max(1, tp) / max(1, (tp + fp))


def calc_recall(vec_t, vec_predict):
    tp = calc_TP(vec_t, vec_predict)
    fn = calc_FN(vec_t, vec_predict)
    return max(1, tp) / max(1, (tp + fn))


def calc_average_precision(vec_prec, vec_rec):
    # S = (a + b) * h/2
    S = np.sum([(vec_prec[i] + vec_prec[i+1]) * abs(vec_rec[i+1] - vec_rec[i]) / 2
                for i in range(len(vec_rec)-1)])
    return S
