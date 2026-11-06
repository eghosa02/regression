import numpy as np


def R_squared(res, Y):
    RSS = res.T@res
    T = Y - np.mean(Y)
    TSS = T.T@T
    R_squared = 1 - (RSS / TSS)
    return R_squared

def R_squared_real(res, Y):
    R = res - np.mean(res)
    RSS = R.T@R
    T = Y - np.mean(Y)
    TSS = T.T@T
    R_squared = 1 - (RSS / TSS)
    return R_squared


def adjusted_R_squared(res, Y, k):
    adj_R_squared = 1 - (1 - R_squared(res, Y)) * ((len(Y) - 1) / (len(Y) - k - 1))
    return adj_R_squared


def AIC(res, Y, k):
    AIC = np.log((res.T@res)/len(Y)) + (2 * k) / len(Y)
    return AIC


def BIC(res, Y, k):
    BIC = np.log((res.T@res)/len(Y)) + (k * np.log(len(Y))) / len(Y)
    return BIC


def VIF(X_1, X_2):
    U = X_2 - X_1
    R_squared_U_X1 = R_squared(U, X_1)
    VIF = 1 / (1 - R_squared_U_X1)
    return VIF
