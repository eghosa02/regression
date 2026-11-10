import numpy as np
from scipy import stats
from model import LinearRegressionModel
from sklearn.preprocessing import PolynomialFeatures


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


class chi2_test:
    def __init__(self, model):
        self.model = model

    def test(self, Y, R, r, alpha=0.05):
        Y = np.array(Y)
        chi2_num = (R@self.model.beta - r)
        chi2_denum = R@self.model.get_stats()['parameter_variance']@R.T
        chi2_test = chi2_num.T@np.linalg.inv(chi2_denum)@chi2_num
        p_values = stats.chi2.sf(chi2_test, R.shape[0])
        accept_null = p_values > alpha
        return chi2_test, p_values, accept_null
    

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


class Reset_test:
    def __init__(self, model):
        self.model = model
        
    
    def test(self, p=2, alpha=0.05):
        aux_model = LinearRegressionModel(beta=None)
        X_new = self.model.X
        for i in range(2, p + 1):
            X_new = np.column_stack((X_new, self.model.fitted**i))
        aux_model.train(X_new, self.model.Y)
        f_test = F_test(self.model, aux_model)
        test_res = f_test.test(self.model.Y, alpha)
        return test_res


class Chow_test:
    def __init__(self, model):
        self.model = model

    def test_given_t(self, t, alpha=0.05):
        X1 = self.model.X[:t]
        Y1 = self.model.Y[:t]
        aux_model_1 = LinearRegressionModel(beta=None)
        aux_model_1.train(X1, Y1)
        X2 = self.model.X[t:]
        Y2 = self.model.Y[t:]
        aux_model_2 = LinearRegressionModel(beta=None)
        aux_model_2.train(X2, Y2)
        concatenated_model = aux_model_1 + aux_model_2
        f_test = F_test(self.model, concatenated_model)
        test_res = f_test.test(self.model.Y, alpha)
        return test_res

    def test(self, trim=0.1, alpha=0.05):
        test_res_list = []
        le_y = len(self.model.Y)
        start = int(trim * le_y) #10
        end = le_y - start + 1   #91, l'ultimo è escluso
        for t in range(start, end):
            test_res_list.append(self.test_given_t(t, alpha))
        return test_res_list, alpha


class White_test:
    def __init__(self, model):
        self.model = model


    def test(self, deg=2, alpha=0.05):
        res = self.model.get_stats()['residuals']**2
        X = self.model.X[:, 1:]
        X_U = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X)
        model = LinearRegressionModel(beta=None)
        model.train(X_U, res)
        model_r = LinearRegressionModel(beta=None)
        model_r.train(np.ones((len(res), 1)), res)
        f_test = F_test(model_r, model)
        test_res = f_test.test(res, alpha)
        return test_res


class White_test_for_dummyes:
    def __init__(self, model):
        self.model = model

    def _build_white_matrix(self, X, deg):
        _, n_features = X.shape
        
        is_dummy = np.array([
            set(np.unique(X[:, j])).issubset({0, 1, 0.0, 1.0}) 
            for j in range(n_features)
        ])
        
        cols = [X]
        
        for j in range(n_features):
            if not is_dummy[j]:
                cols.append((X[:, j] ** 2).reshape(-1, 1))
                
        if deg >= 2:
            interactions = []
            for j in range(n_features):
                for m in range(j + 1, n_features):
                    if not (is_dummy[j] and is_dummy[m]):
                        interactions.append((X[:, j] * X[:, m]).reshape(-1, 1))
                        
            if interactions:
                cols.append(np.hstack(interactions))
                
        X_U = np.hstack(cols)
        #_, unique_indices = np.unique(X_U, axis=1, return_index=True)
        #X_U = X_U[:, sorted(unique_indices)]
        return X_U


    def test(self, deg=2, alpha=0.05):
        res = self.model.get_stats()['residuals']**2
        X = self.model.X[:, 1:]
        X_U = self._build_white_matrix(X, deg)
        model = LinearRegressionModel(beta=None)
        model.train(X_U, res)
        model_r = LinearRegressionModel(beta=None)
        model_r.train(np.ones((len(res), 1)), res)
        f_test = F_test(model_r, model)
        test_res = f_test.test(res, alpha)
        return test_res


class DW_test: #DA CONSULTARE I VALORI TABELLATI PER CONFERMARE O MENO H0
    def __init__(self, model):
        self.model = model

    def test(self, alpha=0.05):
        res = self.model.get_stats()['residuals']
        num = np.sum((res[1:] - res[:len(res)-1])**2)
        den = res.T@res
        return num/den


class BG_test:
    def __init__(self, model):
        self.model = model
    
    def test(self, p, alpha=0.05):
        res = self.model.get_stats()['residuals']
        model = LinearRegressionModel(beta=None)
        model.train(self.model.X, res)
        aux_model = LinearRegressionModel(beta=None)

        lags = np.column_stack([
            np.concatenate((np.zeros(j), res[:-j]))       #giustamente la matrice dei residui di base senza lag ha lo stesso numero di righe di X, quindi ad ogni lag, bisogna anticipare con numero di zeri necessari per eguagliarne la dimensione
                for j in range(1, p + 1)
        ])
        X_U = np.hstack((self.model.X, lags)) 
        aux_model.train(X_U, res)

        f_test = F_test(model, aux_model)
        test_res = f_test.test(res, alpha)
        return test_res


    


if __name__ == "__main__":
    from dataset import load_dataset
    from visualize import plot_chow_stats

    dataset = load_dataset("dataset").reset_index()
    X, Y = dataset[['age', 'BMI', 'sodium', 'smoker', 'gender']], dataset['arterial_pressure']
    model = LinearRegressionModel(beta=None)
    model.train(X, Y)
    test = Chow_test(model)
    sec, alpha = test.test()
    plot_chow_stats(sec, alpha)