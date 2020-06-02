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

def hierarchical(baseline_X, y, df1, ErrVar):
    N, k_baseline = np.shape(baseline_X)
    Res_baseline = teg_regression(baseline_X, y)
    F_baseline = Res_baseline['F']
    df1_baseline = Res_baseline['df1']
    df2_baseline = Res_baseline['df2']
    ErrVar_baseline = Res_baseline['ErrVar']
    Delta_df1 = df1 - df1_baseline
    Delta_df2 = N - df1
    Delta_F = ((ErrVar_baseline * (N-1) - ErrVar * (N-1)) / Delta_df1) / (ErrVar * (N-1) / Delta_df2)
    Delta_p = 1 - stats.f.cdf(Delta_F, Delta_df1, Delta_df1)
    return Delta_F, Delta_p, Delta_df1, Delta_df2

def teg_regression(X, y, baseline_X = []):
    # X is a (N, k) NumPy array
    # y is a (N,) NumPy array
    # baseline_X is an optional baseline model
    #
    # X and baseline_X should not include a ones column.
    if len(np.shape(X)) == 1:
        X = np.reshape(X, (len(X), 1))
    if len(baseline_X) > 0:
        X = np.hstack([X, baseline_X])
    N, k = np.shape(X)
    Inter = np.ones((N, 1))
    X = np.hstack([X, Inter])
    # Get least-squares coefficients
    Fit = np.linalg.lstsq(X, y, rcond=None)
    coeffs = Fit[0]
    # Get model fit
    F, F_p, R2, df1, df2, ErrVar, resss = get_F_regr(X, y, coeffs)
    # T-tests per predictor
    t_vec, p_vec, df_t, se_vec = get_coeff_test(X, resss, coeffs)
    # Hierarchical
    if len(baseline_X) > 0:
        Delta_F, Delta_p, Delta_df1, Delta_df2 = hierarchical(baseline_X, y, df1, ErrVar)
    else:
        Delta_F = 0
        Delta_p = 1
        Delta_df1 = 0
        Delta_df2 = 0
    # Return results
    return ({'b':coeffs, 'R2':R2, 
        'df1':df1, 'df2':df2, 'F':F, 'F_p':F_p,
        't':t_vec, 't_p':p_vec, 'df_t':df_t, 'se_vec':se_vec, 'Delta_F':Delta_F, 'Delta_p':Delta_p, 'Delta_df1':Delta_df1, 'Delta_df2':Delta_df2, 'ErrVar':ErrVar})

def teg_report_regr(Res):
    print('R2 = ' + str(np.around(Res['R2'], 3)) + ', F(' + str(np.around(Res['df1'], 3)) + ', ' + str(np.around(Res['df2'], 3)) + ') = ' + str(np.around(Res['F'], 3)) + ', p = ' + str(np.around(Res['F_p'], 3)))
    if Res['Delta_df1'] > 0:
        print('Delta F(' + str(np.around(Res['Delta_df1'], 3)) + ', ' + str(np.around(Res['Delta_df2'], 3)) + ') = ' + str(np.around(Res['Delta_F'], 3)) + ', p = ' + str(np.around(Res['Delta_p'], 3)))
    for ik in range(len(Res['b']) - 1):
        print('b[' + str(ik) + '] = ' + str(np.around(Res['b'][ik], 3)) + ', se(b) = ' + str(np.around(Res['se_vec'][ik], 3)) + ', t(' + str(np.around(Res['df_t'], 3)) + ') = ' + str(np.around(Res['t'][ik], 3)) + ', p = ' + str(np.around(Res['t_p'][ik], 3)))
    print('Offset = ' + str(np.around(Res['b'][-1], 3)))

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

#
### teg_RMA
#

def level_combinations(levels, level_vec = [], combinations = [], current_factor = 0):
    if len(level_vec) == 0:
        level_vec = [0 for n in range(len(levels))]
        combinations = []
    N_levels = levels[current_factor]
    if current_factor < (len(level_vec) - 1):
        for i_level in range(N_levels):
            level_vec[current_factor] = i_level
            combinations = level_combinations(levels, level_vec, combinations, current_factor + 1)
    else:
        for i_level in range(N_levels):
            level_vec[current_factor] = i_level
            combinations.append(list(level_vec))
    return combinations

def effect_code_factor(factor_levels):
    effect_coder_array = []
    for i_level in range(1, max(factor_levels) + 1):
        effect_coder_vec = [0 for n in range(len(factor_levels))]
        for i_comb in range(len(factor_levels)):
            if factor_levels[i_comb] == 0:
                effect_coder_vec[i_comb] = -1
            if factor_levels[i_comb] == i_level:
                effect_coder_vec[i_comb] = 1
        effect_coder_array.append(effect_coder_vec)
    return effect_coder_array

def effect_coding(levels):
    comb = level_combinations(levels);
    effect_coders = []
    # Effect coding vectors per base factor
    for i_factor in range(len(levels)):
        factor_levels = [r[i_factor] for r in comb]
        effect_coding_M = effect_code_factor(factor_levels)
        effect_coders.append(effect_coding_M)
    # Generate effect codes for all main effects and interactions
    max_way = len(levels)
    tuple_vec = [2 for n in range(max_way)]
    tuples = level_combinations(tuple_vec)
    effect_coding_arrays = []
    effect_factors = []
    for tuple in tuples:
        if np.sum(tuple) == 0:
            continue
        effect_tuple = [[1 for n in range(len(comb))]]
        for i_factor in range(len(tuple)):
            if tuple[i_factor] == 1:
                effect_coder_to_add = effect_coders[i_factor]
                effect_tuple_current = list(effect_tuple)
                # For first vector into_add:
                for i_vec in range(len(effect_tuple_current)):
                    for i_vec_to_add in range(len(effect_coder_to_add)):
                        new_effect_vector = [effect_tuple_current[i_vec][n] * effect_coder_to_add[i_vec_to_add][n] for n in range(len(effect_tuple_current[i_vec]))]
                        if i_vec_to_add == 0:
                            effect_tuple[i_vec] = new_effect_vector
                        else:
                            effect_tuple.append(new_effect_vector)
        effect_coding_arrays.append(effect_tuple)
        effect_factors.append([n for n in range(len(tuple)) if tuple[n] == 1])
    return effect_coding_arrays, effect_factors

def within_subject_regression(effect_coding_array, Y):
    X = np.matrix(effect_coding_array).T
    # Create reduced matrix containing only individual variations in the relevant within-subject effect
    Y_red = np.zeros(Y.shape)
    for i_data_line in range(len(Y)):
        y_line = Y[i_data_line]
        y_line = y_line - np.mean(y_line)
        Fit = np.linalg.lstsq(X, y_line.T, rcond=None)
        pred = np.matmul(X, Fit[0]).T
        Y_red[i_data_line] = pred
    # Linear regression for initial fit and SS
    Y_red_long = np.reshape(Y_red, (np.prod(Y_red.shape), 1))
    X_rep = np.vstack([X for n in range(len(Y))])
    Fit = np.linalg.lstsq(X_rep, Y_red_long, rcond=None)
    pred = np.matmul(X_rep, Fit[0])
    SSM = np.sum(np.square(pred))
    SSE = Fit[1][0]
    return X, Y_red, Fit, SSM, SSE

def within_subject_results(X, Y_red, Fit, SSM, SSE):
    # G-G epsilon adjustment
    dim0 = 1
    eps0 = 1
    if Y_red.shape[1] > 1:
        C = np.cov(Y_red.T)
        eigen_vals, v = np.linalg.eig(C)
        eigen_vals = np.real(eigen_vals)
        dim0 = len(eigen_vals[eigen_vals > 1e-8 * np.mean(eigen_vals)])
        eps0 = np.square(np.sum(eigen_vals)) / (dim0 * np.sum(np.square(eigen_vals)));
    df1 = X.shape[1]
    df2 = (Y_red.shape[0] - 1) * dim0
    MSM = SSM / (eps0 * df1)
    MSE = SSE / (eps0 * df2)
    F = MSM / MSE
    p = 1 - stats.f.cdf(F, eps0 * df1, eps0 * df2) 
    eta_p_2 = SSM / (SSM + SSE)
    results = [F, df1, df2, p, eta_p_2, eps0, SSM, SSE, MSM, MSE]
    return results

def calculations_per_test(effect_coding_array, Y):
    X, Y_red, Fit, SSM, SSE = within_subject_regression(effect_coding_array, Y)
    results = within_subject_results(X, Y_red, Fit, SSM, SSE)
    return results

def get_results(effect_coding_arrays, Y):
    results = []
    for i_effect in range(len(effect_coding_arrays)):
        effect_results = calculations_per_test(effect_coding_arrays[i_effect], Y)
        results.append(effect_results)
    return results

def report(results, effect_factors):
    for i_effect in range(len(results)):
        f = effect_factors[i_effect]
        if len(f) == 0:
            str0 = 'Offset'
        else:
            str0 = "F" + str(f[0])
            for i_f in range(1, len(f)):
                str0 = str0 + 'xF' + str(f[i_f])                
        s = np.around(results[i_effect], 3)
        str0 = str0 + '.\tF(' + "{:.0f}".format(s[1]) + ', ' + "{:.0f}".format(s[2]) + ') = ' + str(s[0])
        str0 = str0 + ', p = ' + str(s[3]) + ', eta_p^2 = ' + str(s[4])
        print(str0)
        str0 = '\t' + 'SSM = ' + str(s[6])  + ', SSE = ' + str(s[7])  + ', MSM = ' + str(s[9]) + ', MSE = ' + str(s[9]) + ', eps = ' + str(s[5])
        # print(str0)

def teg_RMA(Y, levels):
    effect_coding_arrays, effect_factors = effect_coding(levels)
    results = get_results(effect_coding_arrays, Y)
    report(results, effect_factors)
    return results
