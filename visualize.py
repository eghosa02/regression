import matplotlib.pyplot as plt
import numpy as np


def plot_reg(x, y, model):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(model.predict(x), dtype=float)

    fig, ax = plt.subplots()
    ax.scatter(y_hat, y, alpha=0.4, label='Data points')
    lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    ax.plot(lims, lims, color='red', label='y = ŷ')
    ax.set_xlabel('Predicted systolic blood pressure (mmHg)')
    ax.set_ylabel('Observed systolic blood pressure (mmHg)')
    ax.legend()
    fig.savefig('reg.png', dpi=150)
    plt.close(fig)


def plot_chow_stats(secquence, treshold):
    f_vals = [secquence[i][0] for i in range(len(secquence))]
    p_vals = [secquence[i][1] for i in range(len(secquence))]
    x = [i for i in range(len(secquence))]
    plt.plot(x, f_vals, color='red', label='F line')
    plt.plot(x, p_vals, color='blue', label='P line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Stats')
    plt.savefig('stats.png')
    plt.close()
