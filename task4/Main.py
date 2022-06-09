import numpy as np
import matplotlib.pyplot as plt

from task4.Operations.get_variables import equidist_var, some_formula, random_var, distribute_data
from task4.Operations.regression import design_mtrx_polinom, gradient, gradient_descent, count_regress, loss, MSE, \
    count_W
from task4.Visualization.plt_graphics import intermed_regression, loss_check, lambda_compare_with_MSE, \
    regression_compare

if __name__ == '__main__':
    N = 50
    poly_deg = 7  # basis-func

    x_ = equidist_var(0,1,N)
    ground_truth = some_formula(x_)
    error_ = 2 * random_var(N)
    # special anomaly
    error_[8] -= 80
    error_[25] += 250
    error_[41] -= 200
    t_ = ground_truth + error_

    # distribute with anomaly in train
    anomaly_index = [8, 25, 41]
    x_cut = np.delete(x_, anomaly_index)
    t_cut = np.delete(t_, anomaly_index)
    # len(test) <= len(val) (max difference = 1)
    train_x, train_t, test_x, test_t, val_x, val_t = distribute_data(x_cut, t_cut, 0.74, 0.13)
    train_x = np.insert(train_x, [4, 11, 22], x_[anomaly_index])
    train_t = np.insert(train_t, [4, 11, 22], t_[anomaly_index])
    # train_x = np.append(train_x, x_[anomaly_index])  # add to end
    # train_t = np.append(train_t, t_[anomaly_index])

    # data for regularization + gradient
    lambdas = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 5, 10, 20, 30, 40, 50, 60, 70])
    w_start = 10000 * np.random.rand(poly_deg + 1)
    gamma = 0.005  # step size for W
    eps = 0.00001
    eps0 = 0.000001

    # TRAINING
    train_DM = design_mtrx_polinom(train_x, poly_deg + 1)
    all_loss = []
    all_train_MSE = []
    all_W = []
    y_for_graph = []  # 10 graphics for lambda=0
    for lamb in lambdas:
        w_now = w_start
        w_next = []
        this_loss = []
        counter = 0
        while True:
            y_ = count_regress(w_now, train_DM)
            this_loss.append(loss(train_t, y_))
            if lamb == 0 and (counter == 0 or (counter % 500 == 0 and counter <= 4500)):
                y_for_graph.append(y_)

            now_gradient = gradient(train_t, train_DM, w_now, lamb)
            w_next = gradient_descent(w_now, gamma, now_gradient)

            if np.linalg.norm(w_next - w_now) < eps * (np.linalg.norm(w_next) + eps0):  # too little step
                all_train_MSE.append(MSE(train_t, y_))  # final error
                all_W.append(w_now)  # final weights
                break
            w_now = w_next  # change weights
            counter += 1
        # loss_check(this_loss)
    intermed_regression(y_for_graph, train_x, train_t)  # 10 graphics for lambda=0

    # VALIDATION
    val_DM = design_mtrx_polinom(val_x, poly_deg + 1)
    all_val_MSE = []
    for i, lamb in enumerate(lambdas):
        y_ = count_regress(all_W[i], val_DM)
        all_val_MSE.append(MSE(val_t, y_))

    lambda_compare_with_MSE(all_train_MSE, all_val_MSE, lambdas)

    # Regression without regularization
    simple_regres_DM_train = design_mtrx_polinom(train_x, poly_deg + 1)
    simple_regres_DM_final = design_mtrx_polinom(x_, poly_deg + 1)
    simple_regres_W = count_W(simple_regres_DM_train, train_t, 0)  # lambda = 0, count weights with train data
    simple_regress_y_final = count_regress(simple_regres_W, simple_regres_DM_final)

    # Regression with regularization (best model from validation)
    best_model_index = np.argsort(all_val_MSE)[0]
    regul_DM_final = design_mtrx_polinom(x_, poly_deg + 1)
    regul_y_final = count_regress(all_W[best_model_index], regul_DM_final)

    # TEST
    DM_test = design_mtrx_polinom(test_x, poly_deg + 1)
    simple_regress_y_test = count_regress(simple_regres_W, DM_test)
    simple_regress_MSE_test = MSE(test_t, simple_regress_y_test)

    regul_y_test = count_regress(all_W[best_model_index], DM_test)
    regul_MSE_test = MSE(test_t, regul_y_test)

    regression_compare(x_, t_, simple_regress_y_final, regul_y_final, simple_regress_MSE_test, regul_MSE_test)
