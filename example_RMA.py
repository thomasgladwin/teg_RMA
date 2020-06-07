import numpy as np
# import tegstats_local as tegstats
import tegstats

# Note: Use Pythonic orientation: row-wise data matrices, conditions x samples
#
# Randomization testing uses 50000 iterations; takes a little while.
#
# Be aware this implementation does effect-wise testing as expained in the paper, so between-within interactions aren't identical to, e.g., SPSS.
#
# Gladwin TE (2020). An implementation of repeated measures ANOVA: effect coding, automated exploration of interactions, and randomization testing. MethodsX, doi:10.1016/j.mex.2020.100947.

# Pure within
N = 60
levels = [2, 5]
noise_fac = 1
effects = np.ones((np.prod(levels), 1))
Y = np.matmul(effects, np.ones((1, N))) + noise_fac * np.random.randn(len(effects), N)
stats = tegstats.teg_RMA(Y, levels)
# stats = tegstats.teg_RMA(Y, levels, randomization_tests=1)

# Within x between
levels_per_B = [3, 4]
B = np.vstack([np.random.randint(1, n + 1, size=(1, N)) for n in levels_per_B])
stats = tegstats.teg_RMA(Y, levels, B)
# stats = tegstats.teg_RMA(Y, levels, B, randomization_tests=1)

# Save to text for comparison: adjust parameters as needed
np.savetxt('test.txt', np.hstack((B.T, Y.T))) 
# D = dlmread('test.txt'); B = D(:, 1:2); Y = D(:, 3:end);
# O = teg_RMA(Y, [4, 4], {'v1', 'v2'}, B, {'b1', 'b2'});
