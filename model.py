import numpy as np
from helpers import with_design_matrix
from Statistics import Stats


class RegressionModel:
    def __init__(self, beta):
        self.beta:np.ndarray = beta
        self.X = None
        self.Y = None
        self.predictions = None
        self.stats = Stats()

    
    def get_stats(self):
        stats = self.stats.compute_stats(self.beta.shape[0],self.X, self.Y, self.predictions, self.X[-1])
        return stats


    @with_design_matrix
    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model has not been trained yet. Please call the 'train' method first.")
        self.predictions = X @ self.beta
        return self.predictions


    @with_design_matrix
    def train(self, X, Y):
        Y = np.array(Y)
        self.beta = np.linalg.inv(X.T@X) @ X.T@Y
        self.X = X
        self.Y = Y
        self.predict(self.X)
        return self.beta


if __name__ == "__main__":
    from dataset import load_dataset
    from visualize import plot_reg

    dataset = load_dataset("dataset")
    X, Y = dataset['RM'] - dataset['RF'], dataset['Ri_AAPL'] - dataset['RF']
    model = RegressionModel(beta=None)
    model.train(X, Y)
    plot_reg(X, Y, model)
    print(model.get_stats())