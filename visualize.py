import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt


def plot_reg(x, y, model):
    plt.scatter(x, y, label='Data points')
    plt.plot(x, model.predict(x), color='red', label='REG line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression')
    plt.savefig('regression.png')

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
