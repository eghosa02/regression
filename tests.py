import numpy as np
from scipy import stats
from model import CAPMModel


class T_test:
    def __init__(self, model):
        self.model = model

    def test(self, Y, alpha=0.05):
        Y = np.array(Y)
        var = self.model.get_stats()['parameter_variance']
        t_stat = self.model.beta / np.sqrt(np.diag(var))
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(Y) - self.model.beta.shape[0]))
        accept_null = p_values > alpha
        return t_stat, p_values, accept_null


class F_test:
    def __init__(self, model_R, model_U):
        self.model_R = model_R
        self.model_U = model_U

    def test(self, Y, alpha=0.05):
        Y = np.array(Y)
        res_R = self.model_R.get_stats()['residuals']
        res_U = self.model_U.get_stats()['residuals']
        RSS_R = res_R.T @ res_R
        RSS_U = res_U.T @ res_U
        q = self.model_U.beta.shape[0] - self.model_R.beta.shape[0]
        F_stat = ((RSS_R - RSS_U) / q) / (RSS_U / (len(Y) - self.model_U.beta.shape[0]))
        p_value = 1 - stats.f.cdf(F_stat, q, len(Y) - self.model_U.beta.shape[0])
        accept_null = p_value > alpha
        return F_stat, p_value, accept_null


class 




if __name__ == "__main__":
    from dataset import load_dataset

    dataset = load_dataset("dataset")
    X, Y = dataset['RM'] - dataset['RF'], dataset['Ri_AAPL'] - dataset['RF']
    model = CAPMModel(beta=None)
    model.train(X, Y)
    t_test = T_test(model)
    t_stat, p_values, accept_null = t_test.test(Y)
    print("T-statistics:", t_stat)
    print("P-values:", p_values)
    print("Accept null hypothesis:", accept_null)