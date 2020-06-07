import numpy as np
from scipy import stats

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

def vector_set_combinations(vector_sets, vector_per_set = [], combinations = [], current_set = 0):
    if len(vector_per_set) == 0:
        vector_per_set = [0 for n in range(len(vector_sets))]
        combinations = []
    N_vectors = len(vector_sets[current_set])
    if current_set < len(vector_sets) - 1:
        for i_vector in range(len(vector_sets[current_set])):
            vector_per_set[current_set] = i_vector
            combinations = vector_set_combinations(vector_sets, vector_per_set, combinations, current_set + 1)
    else:
        for i_vector in range(len(vector_sets[current_set])):
            vector_per_set[current_set] = i_vector
            current_vecs = np.vstack([vector_sets[n][vector_per_set[n]] for n in range(len(vector_per_set))])
            new_vector = np.prod(current_vecs, 0)
            if len(new_vector.shape) > 1:
                new_vector = np.reshape(np.array(new_vector), (new_vector.shape[1]))
            combinations.append(new_vector)
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
                # For first vector in to_add:
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

def recode_B_factors_as_effects(B):
    factor_coding_arrays = []
    # B is [variable][observations] oriented
    for iB in range(B.shape[0]):
        bvec = B[iB]
        levels = np.unique(bvec)
        for i_level in range(len(levels)):
            bvec[bvec == levels[i_level]] = i_level
        this_effect_coder = effect_code_factor(np.unique(bvec))
        B_effect_coding_array = np.zeros((len(this_effect_coder), B.shape[1]))
        for i_obs in range(len(bvec)):
            i_level = bvec[i_obs]
            for i_effect_coder in range(len(this_effect_coder)):
                B_effect_coding_array[i_effect_coder][i_obs] = this_effect_coder[i_effect_coder][i_level]
        factor_coding_arrays.append(B_effect_coding_array)
    return factor_coding_arrays

def create_between(B):
    factor_coding_arrays = recode_B_factors_as_effects(B)
    # Matrices for main effects and interactions
    max_way = B.shape[0]
    tuple_vec = [2 for n in range(max_way)]
    tuples = level_combinations(tuple_vec)
    tuples = np.array(tuples)
    B_effect_coding_arrays = []
    B_effect_factors = []
    for tuple in tuples:
        if np.sum(tuple) == 0:
            continue
        factors = [x[0] for x in np.argwhere(tuple == 1)] 
        factor_coding_array_set = [factor_coding_arrays[n] for n in factors]
        this_effect_coding_array = vector_set_combinations(factor_coding_array_set)
        B_effect_coding_arrays.append(np.array(this_effect_coding_array))
        B_effect_factors.append(factors)
    return B_effect_coding_arrays, B_effect_factors

def within_subject_regression(effect_coding_array, Y_original, B_effect_coding_array_original):
    Y = Y_original.copy().T
    X = np.matrix(effect_coding_array).T
    # Create reduced matrix containing only individual variations in the relevant within-subject effect
    Y_red = np.zeros(Y.shape)
    for i_data_line in range(len(Y)):
        y_line = Y[i_data_line]
        y_line = y_line - np.mean(y_line)
        Fit = np.linalg.lstsq(X, y_line.T, rcond=None)
        pred = np.matmul(X, Fit[0]).T
        Y_red[i_data_line] = pred.T
    # Linear regression for initial fit and SS
    Y_red_long = np.reshape(Y_red, (np.prod(Y_red.shape), 1))
    X_rep = np.vstack([X for n in range(len(Y))])
    # Adjust X_rep to represent the interaction with a B-S factor(s)
    # B_effect_coding_array_original has [effect coder][observation] orientation
    if len(B_effect_coding_array_original) > 0:
        this_B_effect_coding_array = B_effect_coding_array_original.copy().T
        X_B = []
        for b_line in this_B_effect_coding_array:
            repeated_per_WS_condition = np.vstack([b_line for n in range(Y_red.shape[1])])
            X_B.append(repeated_per_WS_condition)
        X_B = np.array(X_B)
        X_B = np.reshape(X_B, (X_rep.shape[0], this_B_effect_coding_array.shape[1]))
        for i_col in range(X_B.shape[1]):
            X_B.T[i_col] = X_B.T[i_col] - np.mean(X_B.T[i_col])
        # 
        new_X_rep = []
        for iW in range(X_rep.shape[1]):
            for iB in range(X_B.shape[1]):
                new_X_rep.append(np.multiply(X_rep.T[iW], X_B.T[iB]))
        new_X_rep = np.array(new_X_rep)
        X_rep = np.matrix(new_X_rep).T
    Fit = np.linalg.lstsq(X_rep, Y_red_long, rcond=None)
    pred = np.matmul(X_rep, Fit[0])
    SSM = np.sum(np.square(pred))
    SSE = Fit[1][0]
    return X_rep, Y_red, Fit, SSM, SSE

def within_subject_results(X_rep, Y_red, Fit, SSM, SSE, B_effect_coding_array):
    # G-G epsilon adjustment
    dim0 = 1
    eps0 = 1
    if Y_red.shape[1] > 1:
        C = np.cov(Y_red.T)
        eigen_vals, v = np.linalg.eig(C)
        eigen_vals = np.real(eigen_vals)
        dim0 = len(eigen_vals[eigen_vals > 1e-8 * np.mean(eigen_vals)])
        eps0 = np.square(np.sum(eigen_vals)) / (dim0 * np.sum(np.square(eigen_vals)));
    df1 = X_rep.shape[1]
    if len(B_effect_coding_array) == 0:
        df2 = (Y_red.shape[0] - 1) * dim0
    else:
        df2 = (Y_red.shape[0] - 1 - (len(B_effect_coding_array))) * dim0
    MSM = SSM / (eps0 * df1)
    MSE = SSE / (eps0 * df2)
    F = MSM / MSE
    p = 1 - stats.f.cdf(F, eps0 * df1, eps0 * df2) 
    eta_p_2 = SSM / (SSM + SSE)
    results = [F, df1, df2, p, eta_p_2, eps0, SSM, SSE, MSM, MSE]
    return results

def calculations_per_test(effect_coding_array, Y, B_effect_coding_array, randomization_tests):
    X_rep, Y_red, Fit, SSM, SSE = within_subject_regression(effect_coding_array, Y, B_effect_coding_array)
    results = within_subject_results(X_rep, Y_red, Fit, SSM, SSE, B_effect_coding_array)
    return results

def get_results(effect_coding_arrays, B_effect_coding_arrays, Y, randomization_tests):
    results = []
    factors_per_result = []
    for i_effect in range(len(effect_coding_arrays)):
        effect_results = calculations_per_test(effect_coding_arrays[i_effect], Y, [], randomization_tests)
        results.append(effect_results)
        factors_per_result.append([i_effect, -1])
        for i_effect_B in range(len(B_effect_coding_arrays)):
            effect_results = calculations_per_test(effect_coding_arrays[i_effect], Y, B_effect_coding_arrays[i_effect_B], randomization_tests)
            results.append(effect_results)
            factors_per_result.append([i_effect, i_effect_B])
    if randomization_tests == 1:
        nIts = 50000
        F_rnd = []
        print('Running randomization tests.')
        for iIt in range(nIts):
            if iIt > 0 and np.mod(iIt, np.floor(nIts/10)) == 0:
                perc = np.around(100 * iIt / nIts)
                print(str(perc) + '% ', end = "", flush=True)
            # Flip and permute
            flipper0 = -1 + 2 * np.random.randint(0, 2, size=(1, Y.shape[1]))
            flipperM = np.matmul(np.ones((Y.shape[0], 1)), flipper0)
            Y_flip = np.multiply(Y, flipperM)
            results_rnd, dum0 = get_results(effect_coding_arrays, B_effect_coding_arrays, Y_flip, 0)
            F_list = [results_rnd[n][0] for n in range(len(results_rnd))]
            F_rnd.append(F_list)
        print('Randomization tests complete.\n')
        F_rnd = np.array(F_rnd)
        # Replace original results p with randomization-based p
        for i_test in range(len(results)):
            F_rnd_vec = np.array(F_rnd.T[i_test])
            n_higher = len(np.where(F_rnd_vec > results[i_test][0])[0])
            p_rand = n_higher / len(F_rnd)
            results[i_test][3] = p_rand
    return results, factors_per_result

def report(results, effect_factors, B_effect_factors, factors_per_result):
    for i_effect in range(len(results)):
        # Within factors
        f = effect_factors[factors_per_result[i_effect][0]]
        if len(f) == 0:
            str0 = 'Offset'
        else:
            str0 = "W" + str(f[0])
            for i_f in range(1, len(f)):
                str0 = str0 + 'xW' + str(f[i_f])                
        # Between factors
        if factors_per_result[i_effect][1] >= 0:
            f = B_effect_factors[factors_per_result[i_effect][1]]
            str0 = str0 + "xB" + str(f[0])
            for i_f in range(1, len(f)):
                str0 = str0 + 'xB' + str(f[i_f])                
        s = np.around(results[i_effect], 3)
        str0 = str0 + '.\tF(' + "{:.0f}".format(s[1]) + ', ' + "{:.0f}".format(s[2]) + ') = ' + str(s[0])
        str0 = str0 + ', p = ' + str(s[3]) + ', eta_p^2 = ' + str(s[4])
        print(str0)
        str0 = '\t' + 'SSM = ' + str(s[6])  + ', SSE = ' + str(s[7])  + ', MSM = ' + str(s[9]) + ', MSE = ' + str(s[9]) + ', eps = ' + str(s[5])
        # print(str0)
