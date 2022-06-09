import numpy as np


def from_matrix_to_vec(matrix_predict):
    # take one predicted class with max probability
    return np.array([np.argmax(y_i) for y_i in matrix_predict])


def calc_TP(vec_t, vec_predict, class_i):
    mask_true_answer = vec_t == vec_predict
    mask_say_class_i = vec_predict == class_i
    mask_tp = np.logical_and(mask_true_answer, mask_say_class_i)
    return sum(mask_tp)


def calc_TN(vec_t, vec_predict, class_i):
    mask_true_answer = vec_t == vec_predict
    mask_say_not_class_i = vec_predict != class_i
    mask_tn = np.logical_and(mask_true_answer, mask_say_not_class_i)
    return sum(mask_tn)


def calc_FP(vec_t, vec_predict, class_i):
    mask_false_answer = vec_t != vec_predict
    mask_say_class_i = vec_predict == class_i
    mask_fp = np.logical_and(mask_false_answer, mask_say_class_i)
    return sum(mask_fp)


def calc_FN(vec_t, vec_predict, class_i):
    mask_false_answer = vec_t != vec_predict
    mask_was_class_i = vec_t == class_i
    mask_fn = np.logical_and(mask_false_answer, mask_was_class_i)
    return sum(mask_fn)


def calc_accuracy(vec_t, matrix_predict):
    vec_predict = from_matrix_to_vec(matrix_predict)
    mask_true_answers = vec_t == vec_predict
    return sum(mask_true_answers) / max(1, len(vec_t))


def calc_precision(vec_t, matrix_predict, class_i):
    vec_predict = from_matrix_to_vec(matrix_predict)
    tp = calc_TP(vec_t, vec_predict, class_i)
    fp = calc_FP(vec_t, vec_predict, class_i)
    return tp / max(1, (tp + fp))


def calc_recall(vec_t, matrix_predict, class_i):
    vec_predict = from_matrix_to_vec(matrix_predict)
    tp = calc_TP(vec_t, vec_predict, class_i)
    fn = calc_FN(vec_t, vec_predict, class_i)
    return tp / max(1, (tp + fn))


def conf_matrix(vec_t, matrix_predict, K):
    vec_predict = from_matrix_to_vec(matrix_predict)
    C = np.zeros((K, K))
    for t, y in zip(vec_t, vec_predict):
        C[t, y] += 1
    return C
