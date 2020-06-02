import numpy as np
import tegstats

# Create arrays
# Random + effect:
#N = 45
#levels = [2, 3]
#effects = np.matrix([3, 2, 1, 4, 3, 2])
#N_cond = len(effects)
#noise_fac = 4
#Y = np.matmul(np.ones((N, 1)), effects) + noise_fac * np.random.randn(N, N_cond)

# For comparison with Matlab
#   Matlab code:
#       Y = 1:((2 * 3 * 4)*45); Y = Y - 1; Y = mod(Y, 11); Y = reshape(Y, 45, (2 * 3 * 4));
#       O = teg_RMA(Y, [2, 3, 4], {'v1', 'v2', 'v3'});
levels = [2, 3, 4]
Y = np.matrix([np.mod(n, 11) for n in range((2 * 3 * 4) * 45)]) 
Y = np.reshape(Y, (45, (2 * 3 * 4)))

stats = tegstats.teg_RMA(Y, levels)
