import numpy as np
import tegstats

# Params for simulated data
N = 100
k = 3
k_baseline = 6
coeffs_true = np.array([0.1 * n for n in range(k)])
offset_t = 20

# Create arrays
X = np.array([np.random.randn(N) for n in range(k)]).T
X_baseline = np.array([np.random.randn(N) for n in range(k_baseline)]).T
y = np.matmul(X, coeffs_true) + offset_t + np.random.randn(N)

# Run multiple regression
print('Multiple')
Res = tegstats.teg_regression(X, y)
tegstats.teg_report_regr(Res)
print()

# Run hierarchical regression
print('Hierarchical')
Res = tegstats.teg_regression(X, y, X_baseline)
tegstats.teg_report_regr(Res)
print()

# Multicollinearity
print('Multicolinearity exploration')
N = 100
indep_fac = 0.2
x1 = np.random.randn(N)
x2 = np.random.randn(N)
x3 = x1 + x2 + indep_fac * np.random.randn(N)
Res = tegstats.teg_regression(np.vstack([x2, x3]).T, x1)
VIF = 1/(1 - Res['R2'])
print('VIF x1 = ' + str(VIF))
Res = tegstats.teg_regression(np.vstack([x1, x3]).T, x2)
VIF = 1/(1 - Res['R2'])
print('VIF x2 = ' + str(VIF))
Res = tegstats.teg_regression(np.vstack([x1, x2]).T, x3)
VIF = 1/(1 - Res['R2'])
print('VIF x3 = ' + str(VIF))

y = x1 + np.random.randn(N)
print('x1')
Res = tegstats.teg_regression(x1, y)
tegstats.teg_report_regr(Res)
print('x2')
Res = tegstats.teg_regression(x2, y)
tegstats.teg_report_regr(Res)
print('x3')
Res = tegstats.teg_regression(x2, y)
tegstats.teg_report_regr(Res)
print('x1, x2')
Res = tegstats.teg_regression(np.vstack([x1, x2]).T, y)
tegstats.teg_report_regr(Res)
print('x1, x2, x3')
Res = tegstats.teg_regression(np.vstack([x1, x2, x3]).T, y)
tegstats.teg_report_regr(Res)

p1v = []
p2v = []
VIFv = []
for iIt in range(500):
    N = 100
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    x3 = x1 + x2 + indep_fac * np.random.randn(N)
    Res = tegstats.teg_regression(np.vstack([x1, x3]).T, x2)
    VIF = 1/(1 - Res['R2'])
#   Uncomment to make sure only samples with a VIF < 5 are included.
#    if VIF > 5:
#        continue
    VIFv.append(VIF)
    y = x1 + np.random.randn(N)
    Res = tegstats.teg_regression(x2, y)
    p1 = Res['t_p'][0]
    Res = tegstats.teg_regression(np.vstack([x1, x2, x3]).T, y)
    p2 = Res['t_p'][1]
    p1v.append(p1)
    p2v.append(p2)
p1v = np.array(p1v)
p2v = np.array(p2v)
VIFv = np.array(VIFv)
posrate1 = np.count_nonzero(p1v < 0.05) / len(p1v)
posrate2 = np.count_nonzero(p2v < 0.05) / len(p2v)
print('Significance rate for x2 when sole predictor: ' + str(np.around(posrate1, 3)))
print('Significance rate for x2 when in model with x1 and x3: ' + str(np.around(posrate2, 3)))
print('Mean variance inflation factor was ' + str(np.around(np.mean(VIFv), 3)))
