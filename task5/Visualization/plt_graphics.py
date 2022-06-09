import matplotlib.pyplot as plt
import numpy as np


def prec_rec_graph(vec_prec, vec_rec, vec_tao, aver_prec):
    plt.plot(vec_rec, vec_prec, marker='.')
    plt.title(f"Precision-recall curve depending on height, S = {aver_prec:.3}", fontsize=14, pad=15)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.xticks(np.arange(0, 1.2, 0.1))
    plt.yticks(np.arange(0, 1.2, 0.1))
    for x, y, tao, i in zip(vec_rec, vec_prec, vec_tao, range(0, len(vec_tao))):
        sign = 1 if i%2 == 0 else -12
        plt.annotate(f"{tao}", (x, y), (x + 0.004 * sign, y), fontsize=8, fontweight="bold")
    plt.show()
