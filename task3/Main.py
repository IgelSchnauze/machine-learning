import numpy as np
from itertools import combinations

from task3.Operations.get_variables import equidist_var, distribute_data, some_formula, normal_var
from task3.Operations.regression import design_mtrx_func, count_W, count_regress
from task3.Utils.make_graph import bar_error_train_val

if __name__ == '__main__':
    N = 1000

    x_ = equidist_var(0, 2 * np.pi, N)
    ground_truth_ = some_formula(x_)
    error_ = normal_var(0, np.sqrt(100), N)
    t_ = ground_truth_ + error_

    # len(test) <= len(val) (max difference = 1)
    train_x, train_t, test_x, test_t, val_x, val_t = distribute_data(x_, t_, 0.8, 0.1)

    functions = np.array([np.sin, np.cos, lambda x: np.log(x, where=x > 0, out=np.zeros_like(x)),
                          np.exp, np.sqrt, lambda x: x, lambda x: x*x, lambda x: x*x*x])
    functions_name = np.array(['sin(x)', 'cos(x)', 'ln(x)', 'e^x', 'sqrt(x)', 'x', 'x^2', 'x^3'])

    # learning process
    all_W = []
    all_func_set = []
    all_train_MSE = []
    all_val_MSE = []
    for k in range(1, 4):  # combinations for 1, 2, 3 items
        func_index_sets = combinations(np.arange(len(functions)), k)

        for index_set in func_index_sets:
            basis_functions = [functions[i] for i in index_set]  # not use [index] to have python.list
            basis_functions.insert(0, lambda x: np.ones_like(x))  # fi_0 = 1

            # training, get W (+ метод max правдоподобия)
            DM = design_mtrx_func(train_x, basis_functions)
            W = count_W(DM, train_t)
            y_ = count_regress(W, DM)
            train_MSE = 1/len(train_x) * np.sum(np.power(train_t - y_, 2))

            # validation, get best basis_func
            DM = design_mtrx_func(val_x, basis_functions)
            y_ = count_regress(W, DM)
            val_MSE = 1 / len(val_x) * np.sum(np.power(val_t - y_, 2))

            all_func_set.append([i for i in index_set])  # add list, not tuple
            all_W.append(W)
            all_train_MSE.append(train_MSE)
            all_val_MSE.append(val_MSE)

    count_best = 3
    best_model_indexes = np.argsort(all_val_MSE)[:count_best]   # take 3 min errors

    # make plt.bar
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    bar_error_train_val(best_model_indexes, all_W, all_func_set, all_train_MSE, all_val_MSE, functions_name)

    # testing the best model
    basis_functions = [functions[i] for i in all_func_set[best_model_indexes[0]]]
    basis_functions.insert(0, lambda x: np.ones_like(x))  # fi_0 = 1
    DM = design_mtrx_func(test_x, basis_functions)
    W = count_W(DM, test_t)
    y_ = count_regress(W, DM)
    test_MSE = 1/len(test_x) * np.sum(np.power(test_t - y_, 2))
    print("Best model error on test_data =", test_MSE)
