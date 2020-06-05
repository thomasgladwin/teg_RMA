import numpy as np
# import tegstats_local as tegstats
import tegstats

# Note: Use Pythonic orientation: row-wise data matrices

# Pure within

N = 80
levels = [2, 3, 2]
noise_fac = 1
effects = np.ones((np.prod(levels), 1))
Y = np.matmul(effects, np.ones((1, N))) + noise_fac * np.random.randn(len(effects), N)

stats = tegstats.teg_RMA(Y, levels)

# Within x between

levels_per_B = [3, 3, 2]
B = np.vstack([np.random.randint(1, n + 1, size=(1, N)) for n in levels_per_B])

stats = tegstats.teg_RMA(Y, levels, B)

# Save to text for comparison
np.savetxt('test.txt', np.hstack((B.T, Y.T))) 
# D = dlmread('test.txt'); B = D(:, 1:2); Y = D(:, 3:end);
# O = teg_RMA(Y, [4, 4], {'v1', 'v2'}, B, {'b1', 'b2'});
