import numpy as np
import matplotlib.pyplot as plt

from task2.Operations.get_variables import normal_var, equidist_var, some_formula
from task2.Operations.regression import design_mtrx_polinom, count_W, polinom_regres


if __name__ == '__main__':
    N = 1000
    max_degree = 20

    x_ = equidist_var(0, 2 * np.pi, N)
    ground_truth_ = some_formula(x_)
    error_ = normal_var(0, np.sqrt(100), N)
    t_ = ground_truth_ + error_
    DM = design_mtrx_polinom(x_, max_degree + 1) # + 1 - because begin with ^0

    RMSY_all = []
    fig, axs = plt.subplots(4, 5, figsize=(15, 9))
    for degree in range(1, max_degree + 1):
        DM_slice = DM[:,:degree + 1]
        W = count_W(DM_slice, t_)
        y = polinom_regres(W, DM_slice)
        RMSY = np.sqrt(1/N * np.sum((t_ - y)**2))
        RMSY_all.append(RMSY)

        this_axs = axs[int((degree - 1) / 5), int((degree - 1) % 5)]
        # axs[int((degree-1) / 5), int((degree-1) % 5)].plot(t_, 'm-', y, 'k--')  # линия
        this_axs.scatter(x_, t_, color = 'm', s = 0.4)  # точки
        this_axs.plot(x_, y, 'k--')
        this_axs.title.set_text('deg = %s' % degree)
        if degree == 1:
            this_axs.legend(['Regression', 'Target variables'], bbox_to_anchor = (0.2, 1.8), fontsize = 'large')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    plt.plot([i for i in range(1, max_degree + 1)], RMSY_all)
    plt.xlabel('max degree')
    plt.ylabel('error')
    plt.title("Зависимость ошибки от max степени полинома")
    plt.show()