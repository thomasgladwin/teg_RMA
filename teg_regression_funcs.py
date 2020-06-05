import numpy as np
from scipy import stats

#
### tegregr
#

def get_F_regr(X, y, coeffs):
    N, k = np.shape(X)
    # Calculate statistics
    pred = np.matmul(X, coeffs)
    res = y - pred
    R2 = np.var(pred) / np.var(y)
    # F-test of overall model fit
    df1 = k - 1
    df2 = N - 1 - df1
    SSM = sum((pred - np.mean(pred))**2)
    MSM = SSM / df1
    SSE = sum((res - np.mean(res))**2)
    MSE = SSE / df2
    F = MSM/MSE
    F_p = 1 - stats.f.cdf(F, df1, df2)
    ErrVar = np.var(res)
    resss = sum(res**2)
    return F, F_p, R2, df1, df2, ErrVar, resss

def get_coeff_test(X, resss, coeffs):
    N, k_off = np.shape(X)
    k = k_off - 1 # k is number of redictors
    t_vec = []
    p_vec = []
    se_vec = []
    MSE = resss / (N - k - 1)
    invXtX = np.linalg.inv(np.matmul(X.T, X))
    for ik in range(len(X.T)):
        se = np.sqrt(MSE * invXtX[ik][ik])
        if se > 0:
            se_vec.append(se)
            t = coeffs[ik]/se
            t_vec.append(t)
            df_t = N - k - 1
            p = 1 - stats.t.cdf(t, df_t)
            p = 2 * np.min([p, 1-p])
            p_vec.append(p)
        else:
            t_vec.append(0)
            p_vec.append(1)
    return t_vec, p_vec, df_t, se_vec

def create_correlated_variable(X, r):
    if len(np.shape(X)) == 1:
        X = np.reshape(X, (len(X), 1))
    N, k = np.shape(X)
    Y_init = np.random.randn(N)
    Res = teg_regression(X, Y_init)
    Y_pred = np.matmul(np.hstack([X, np.ones((N, 1))]), Res['b'])
    Y_resid = Y_init - Y_pred
    Y_pred = results.zscore(Y_pred)
    Y_resid = results.zscore(Y_resid)
    Y_correlated = r * Y_pred + np.sqrt(1 - r**2) * Y_resid
    return Y_correlated
