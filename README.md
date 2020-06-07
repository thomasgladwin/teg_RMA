# teg_RMA

Repeated measures ANOVA in Matlab and (work in progress, with fewer bells and whistles) Python.

For use in Python: pip install tegstats (or use the tegstats_local.py file), and see example_RMA.py for usage. The Python version so far only does pure within-subject analyses and within x between-factor interactions (with an arbitrary number of factors and levels per factor), and has the option to use randomization tests (hope you're not in a hurry though). The package includes the multiple and hierarchical regression function teg_regression (see example_regression.py for usage).

For Matlab, the basic usage is: O = teg_RMA(M, levels, varnames)

with M an observation x nested variable-combinations matrix; levels a vector with the number of levels per variable (from highest to lowest level of nesting); and varnames a cell array of strings.

Interactions are explored by recursively testing the lower-level effects per level of the final factor.

Categorical and continuous between-subject factors can be added. Note that in this implementation, tests are performed per effect separately. So, including a between-subject factor does not affect within-subject tests. Be aware that this differs from the tests in SPSS.

The F-tests for significant effects are reported, together with other statistical measures including partial eta squared estimates of effect size. Various descriptive statistics are given. Note that the p-values printed with means reflects the difference of the raw scores from zero when there is no within-subject factor, and the differences of the scores after subtraction of the subject-mean from zero when there is a within-subject factor.

Set perm_test = 1 in the settings variables at the top of teg_RMA to use randomization and permutation testing to determine p-values (slow but nicely avoids assumptions); set perm_test = 0 otherwise. Set nIts_perm to adjust the number of iterations. 

To use the program, you need to download the latest teg_RMA.zip, teg_basic_stats.zip and teg_basic_funcs.zip, unpack them to their own directories, and add them all to the Matlab path.

Please cite as:
Thomas Edward Gladwin (2020). An implementation of repeated measures ANOVA: effect coding, automated exploration of interactions, and randomization testing. MethodsX, doi:10.1016/j.mex.2020.100947. https://www.sciencedirect.com/science/article/pii/S2215016120301679?via%3Dihub.
