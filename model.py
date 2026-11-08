import numpy as np
from helpers import with_design_matrix
from Statistics import Stats


class LinearRegressionModel:
    def __init__(self, beta):
        self.beta:np.ndarray = beta
        self.X = None
        self.Y = None
        self.fitted = None
        self.stats = Stats()

    
    def get_stats(self):
        stats = self.stats.compute_stats(self.beta.shape[0],self.X, self.Y, self.fitted, self.X[-1])
        return stats


    @with_design_matrix
    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model has not been trained yet. Please call the 'train' method first.")
        return X @ self.beta


    @with_design_matrix
    def train(self, X, Y):
        Y = np.array(Y)
        self.beta = np.linalg.inv(X.T@X) @ X.T@Y
        self.X = X
        self.Y = Y
        self.fitted = X @ self.beta
        return self.beta


    def __add__(self, other):

        if self.beta is None or other.beta is None:
            raise ValueError("Both models must be trained before being combined.")
        if self.X.shape[1] != other.X.shape[1]:
            raise ValueError("Both models must have the same regressors.")

        k = self.X.shape[1]
        n1, n2 = self.X.shape[0], other.X.shape[0]
        X = np.block([
            [self.X,            np.zeros((n1, k))],
            [np.zeros((n2, k)), other.X          ],
        ])                                                  

        combined = LinearRegressionModel(beta=np.concatenate((self.beta, other.beta)))
        combined.X = X
        combined.Y = np.concatenate((self.Y, other.Y))
        combined.fitted = X @ combined.beta            
        return combined


if __name__ == "__main__":
    from dataset import load_dataset
    from visualize import plot_reg

    dataset = load_dataset("dataset")
    X, Y = dataset['RM'] - dataset['RF'], dataset['Ri_AAPL'] - dataset['RF']
    model = LinearRegressionModel(beta=None)
    model.train(X, Y)
    plot_reg(X, Y, model)
    print(model.get_stats())