"""
test_climapy_stats:
    Test the climapy_stats part of climapy.

Usage:
    Designed for use with pytest.

Author:
    Benjamin S. Grandey, 2017
"""

import climapy
import numpy as np
import pytest


class TestFdr:
    """Test stats_fdr()"""

    def test_invalid_p_values(self):
        with pytest.raises(ValueError):
            climapy.stats_fdr([0.0, 0.1, 0.2])

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError):
            climapy.stats_fdr(np.arange(10), alpha='0.1')

    def test_invalid_alpha_two(self):
        with pytest.raises(ValueError):
            climapy.stats_fdr(np.arange(10), alpha=-0.01)

    def test_invalid_alpha_three(self):
        with pytest.raises(ValueError):
            climapy.stats_fdr(np.arange(10), alpha=1.1)

    def test_example_bh95(self):
        # Test data from Benjamini and Hochberg (1995)
        p_values = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000])
        np.random.shuffle(p_values)  # in-place shuffling, so that p-values are out of order
        p_fdr = climapy.stats_fdr(p_values, alpha=0.05)
        assert p_fdr == 4/15 * 0.05  # check correct answer

    def test_example_bh95_2d(self):
        # Test data from Benjamini and Hochberg (1995)
        p_values = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000])
        np.random.shuffle(p_values)  # in-place shuffling, so that p-values are out of order
        p_values = p_values.reshape([3, 5])  # test 2D array
        p_fdr = climapy.stats_fdr(p_values, alpha=0.05)
        assert p_fdr == 4/15 * 0.05

    def test_input_unchanged(self):
        # Test data from Benjamini and Hochberg (1995)
        p_values = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000])
        np.random.shuffle(p_values)  # in-place shuffling, so that p-values are out of order
        p_values_copy = p_values.copy()
        climapy.stats_fdr(p_values, alpha=0.05)
        assert np.array_equal(p_values, p_values_copy)
