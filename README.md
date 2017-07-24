# teg_RMA

<a href="https://doi.org/10.5281/zenodo.826750"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.826750.svg" alt="DOI"></a>

M-files that organize data for analyses using, e.g., repeated measures ANOVA.

Basic usage: O = teg_RMA(M, levels, varnames)

with M an observation x nested variable-combinations matrix; levels a vector with the number of levels per variable (from highest to lowest level of nesting); and varnames a cell array of strings.

Categorical and continuous between-subject factors can be added. Note that in this implementation, tests are performed per effect separately. So, including a between-subject factor does not affect within-subject tests. Be aware that this differs from the tests in SPSS.

The F-tests for significant effects are reported, together with other statistical measures including partial eta squared estimates of effect size.

Set perm_test = 1 in the settings variables at the top of teg_repeated_measures_ANOVA to use permutation testing to determine p-values (slow but nicely avoids assumptions); set perm_test = 0 otherwise. Set nIts_perm to adjust the number of iterations. 

To use the program, you need to download the latest teg_RMA.zip and teg_basic_funcs.zip, unpack them to their own directories, and add both to the Matlab path.
