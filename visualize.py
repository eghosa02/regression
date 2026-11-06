import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
from model import CAPMModel

def plot_reg(x, y, model:CAPMModel):
    plt.scatter(x, y, label='Data points')
    plt.plot(x, model.predict(x), color='red', label='CAPM line')
    plt.xlabel('Market Excess Return (RM - RF)')
    plt.ylabel('Stock Excess Return (Ri - RF)')
    plt.title('CAPM Regression')
    plt.savefig('capm_regression.png')