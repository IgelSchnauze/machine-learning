import matplotlib.pyplot as plt
import numpy as np


def intermed_regression(y_pack, x_, t_):
    fig, axs = plt.subplots(2, 5, figsize=(16, 6))
    for i in range(len(y_pack)):
        this_axs = axs[int(i / 5), int(i % 5)]
        this_axs.plot(np.sort(x_), y_pack[i][np.argsort(x_)], 'm-')
        this_axs.scatter(x_, t_, color='k', s=0.8)
        this_axs.title.set_text('n = %s' % (i * 500))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle("Промежуточная регрессия на весах W_n", fontsize = 14)
    plt.show()


def lambda_compare_with_MSE(train_MSE, val_MSE, lambdas):
    plt.figure(figsize=(16, 7))
    column_pos_x = np.arange(len(lambdas))
    width = 0.4
    graph_tr = plt.bar(column_pos_x - width / 2, train_MSE, width, label="train_MSE", color='olive')
    graph_val = plt.bar(column_pos_x + width / 2, val_MSE, width, label="val_MSE", color='brown')
    plt.ylabel('MSE')
    plt.xlabel('Lambda')
    plt.title('Compare model with regularization parameter', fontsize='x-large')
    plt.xticks(column_pos_x, lambdas)
    plt.legend(loc=3)
    plt.grid()

    for rect in graph_tr + graph_val:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize='small')
    plt.show()


def regression_compare(x_, t_, y_simple, y_with_regul, MSE_simple, MSE_with_regul):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i in [0,1]:
        axs[i].scatter(x_, t_, color='k', s=0.8, label="target_data")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
    axs[0].plot(x_, y_with_regul)
    axs[0].title.set_text("С регуляризацией\n Ошибка на тестовой = " + f"{MSE_with_regul:.4f}")
    axs[0].legend(loc=3)
    axs[1].plot(x_, y_simple)
    axs[1].title.set_text("Без регуляризации\n Ошибка на тестовой = " + f"{MSE_simple:.4f}")

    plt.subplots_adjust(wspace=0.2)
    plt.suptitle("Итоговая регрессия на исходных данных", fontsize=14)
    plt.show()


def loss_check(vec_loss):
    plt.figure()
    plt.plot(vec_loss)
    plt.show()
