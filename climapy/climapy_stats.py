"""
climapy.climapy_stats:
    Functions to support statistical analysis.

Author:
    Benjamin S. Grandey, 2017
"""

import numpy as np


___all__ = ['stats_fdr', ]


def stats_fdr(p_values, alpha=0.10):
    """
    Control the false discovery rate (FDR). Useful when applying multiple hypothesis tests.

    Args:
        p_values: a numpy array containing p-values (e.g. calculated from Welch's t-test).
        alpha: the significance level (default 0.10).

    Returns:
        p_fdr: the p-value threshold for controlling the false discovery rate.
        
    Note:
        Here, in contrast to Wilks (2016), p_fdr is defined as the largest (i/n)*alpha that
        satisfies p(i) <= (i/n)*alpha, as opposed to the largest p(i) value that satisfies the
        inequality. When using p_fdr to find where p_values <= p_fdr, both approaches should give
        the same result.
    
    References:
        Wilks (2016), doi:10.1175/BAMS-D-15-00267.1
        Benjamini and Hochberg (1995), J. R. Stat. Soc. B, 57, 289–300
    """
    # Check input arguments
    if not isinstance(p_values, np.ndarray):
        raise ValueError('Invalid input. p_values should be a numpy array.')
    if not isinstance(alpha, float):
        raise ValueError('Invalid input. alpha should be a float.')
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('Invalid input. alpha should be in range 0.0 < alpha < 1.0.')
    # Calculated p-value threshold
    p_sorted = np.sort(p_values.flatten())  # sort the p-values
    n = p_sorted.size  # number of p-values
    p_fdr = 0.0  # initialise p-value threshold - to be updated
    for i in range(1, n+1):  # loop over p-values
        p_temp = p_sorted[i-1]  # next p-value to check
        if p_temp <= (i * 1. / n) * alpha:
            p_fdr = (i * 1. / n) * alpha  # update p-value threshold
        else:
            break
    return p_fdr
