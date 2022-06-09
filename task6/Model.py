import pickle

import numpy as np

from task6.Operations.metrics import calc_accuracy


class ModelTrainer:
    def __init__(self, M, K_classes, batch_size=4, learning_rate=0.06):
        self.M = M
        self.K_classes = K_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate  # gradient step

        self.W = np.random.normal(0, 1, (self.K_classes, self.M)) * 0.01
        self.b = np.random.normal(0, 1, self.K_classes) * 0.01

        self.best_W = self.W
        self.best_b = self.b
        self.best_val_acc = 0

    def regress_object(self, vec_x):
        return np.dot(self.W, vec_x) + self.b

    def regress_all_objects(self, X):
        return np.array([self.regress_object(x_i) for x_i in X])

    def softmax_stabil(self, Z):
        D = - np.max(Z, axis=1, keepdims=True)  # для каждой х max по классам
        exps = np.exp(Z + D)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def loss_stabil(self, t, predict):
        K = -1 * np.max(predict, axis=1, keepdims=True)
        sum_exp = np.sum(np.exp(predict + K), axis=1, keepdims=True)
        return (1 / t.shape[0]) * np.sum(-1 * np.sum(t * (predict + K - sum_exp), axis=1))

    def gradient_wb(self, x, t, y):
        grad_w = np.dot((y - t).T, x)
        grad_b = np.sum((y - t), axis=0)
        return (1 / t.shape[0]) * grad_w, (1 / t.shape[0]) * grad_b

    def gradient_descent_wb(self, gradient_W, gradient_b, step):
        self.W = self.W - step * gradient_W
        self.b = self.b - step * gradient_b

    def training(self, train_x, train_t, val_x, val_t, step_count, step_for_save_and_val):
        all_loss = []
        val_accuracy = []
        batch_now = 0
        for step in range(step_count):
            batch_end = (batch_now + 1) * self.batch_size  # если больше размера, возьмет остаток
            train_x_batch = train_x[batch_now * self.batch_size: batch_end]
            train_t_batch = train_t[batch_now * self.batch_size: batch_end]

            Z = self.regress_all_objects(train_x_batch)
            Z_softmax = self.softmax_stabil(Z)
            all_loss.append(self.loss_stabil(train_t_batch, Z_softmax))
            gradient_W, gradient_b = self.gradient_wb(train_x_batch, train_t_batch, Z_softmax)
            self.gradient_descent_wb(gradient_W, gradient_b, self.learning_rate)

            batch_now += 1
            if batch_end >= len(train_x):
                batch_now = 0

            # validation and sawing weights
            if step % step_for_save_and_val == 0:
                val_acc_now = self.validation(val_x, val_t)
                val_accuracy.append(val_acc_now)

                if val_acc_now > self.best_val_acc:
                    self.best_val_acc = val_acc_now
                    self.best_W = self.W
                    self.best_b = self.b

        return all_loss, val_accuracy

    def validation(self, val_x, val_t):
        Z = self.regress_all_objects(val_x)
        Z_softmax = self.softmax_stabil(Z)
        return calc_accuracy(val_t, Z_softmax)

    def save_weights_to(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump((self.best_W, self.best_b), f)

    def load_weights_from(self, file_name):
        with open(file_name, 'rb') as f:
            self.W, self.b = pickle.load(f)
            self.best_W = self.W
            self.best_b = self.b
