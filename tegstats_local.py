import numpy as np
from scipy import stats
import teg_RMA_funcs
import teg_regression_funcs

#
### teg_regression
#

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
    F, F_p, R2, df1, df2, ErrVar, resss = teg_regression_funcs.get_F_regr(X, y, coeffs)
    # T-tests per predictor
    t_vec, p_vec, df_t, se_vec = teg_regression_funcs.get_coeff_test(X, resss, coeffs)
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

def teg_report_regr(Res):
    print('R2 = ' + str(np.around(Res['R2'], 3)) + ', F(' + str(np.around(Res['df1'], 3)) + ', ' + str(np.around(Res['df2'], 3)) + ') = ' + str(np.around(Res['F'], 3)) + ', p = ' + str(np.around(Res['F_p'], 3)))
    if Res['Delta_df1'] > 0:
        print('Delta F(' + str(np.around(Res['Delta_df1'], 3)) + ', ' + str(np.around(Res['Delta_df2'], 3)) + ') = ' + str(np.around(Res['Delta_F'], 3)) + ', p = ' + str(np.around(Res['Delta_p'], 3)))
    for ik in range(len(Res['b']) - 1):
        print('b[' + str(ik) + '] = ' + str(np.around(Res['b'][ik], 3)) + ', se(b) = ' + str(np.around(Res['se_vec'][ik], 3)) + ', t(' + str(np.around(Res['df_t'], 3)) + ') = ' + str(np.around(Res['t'][ik], 3)) + ', p = ' + str(np.around(Res['t_p'][ik], 3)))
    print('Offset = ' + str(np.around(Res['b'][-1], 3)))

#
### teg_RMA
#

def teg_RMA(Y, levels, B = [], randomization_tests = 0):
    # Y is numpy matrix, levels is list, B is numpy matrix
    # [condition, observations] orientation
    effect_coding_arrays, effect_factors = teg_RMA_funcs.effect_coding(levels)
    if len(B) > 0:
        B_effect_coding_arrays, B_effect_factors = teg_RMA_funcs.create_between(B)
    else:
        B_effect_coding_arrays = []
        B_effect_factors = []
    [results, factors_per_result] = teg_RMA_funcs.get_results(effect_coding_arrays, B_effect_coding_arrays, Y, randomization_tests)
    teg_RMA_funcs.report(results, effect_factors, B_effect_factors, factors_per_result)
    return results
