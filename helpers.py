import functools
import numpy as np

def with_design_matrix(method):
    @functools.wraps(method)
    def wrapper(self, X, *args, **kwargs):
        X = np.array(X)
        if X.ndim == 1:                  
            X = X.reshape(-1, 1)
        if not np.all(X[:, 0] == 1): 
            one = np.ones((X.shape[0], 1))
            X = np.hstack((one, X))                  
        return method(self, X, *args, **kwargs)
    return wrapper