"""
test_climapy_stats:
    Test the climapy_cesm part of climapy.

Usage:
    Designed for use with pytest.

Author:
    Benjamin S. Grandey, 2017
"""

import climapy
import numpy as np
import os
import pytest
import xarray as xr


# Load test data
data01 = xr.open_dataset(os.path.dirname(__file__)+'/data/data01.nc', decode_times=False,
                         autoclose=True)
# Note: this CESM output data has already undergone some post-processing using CDO.


class TestTimeFromBnds:
    """Test cesm_time_from_bnds()"""

    def test_one(self):
        data = climapy.cesm_time_from_bnds(data01)  # by default, min_year=1701
        assert np.array_equal(data['time'].values[[0, -1]],
                              np.array(['1703-01-16T11:00:00', '1707-12-16T11:00:00'],
                                       dtype='datetime64'))

    def test_two(self):
        data01_mod = data01.copy()
        data = climapy.cesm_time_from_bnds(data01_mod, min_year=2001)
        assert np.array_equal(data['time'].values[[0, -1]],
                              np.array(['2003-01-16T11:00:00', '2007-12-16T11:00:00'],
                                       dtype='datetime64'))


class TestDataUnchanged:
    """Check that data01 has not been changed inplace."""

    def test_data_unchanged(self):
        data01_orig = xr.open_dataset(os.path.dirname(__file__)+'/data/data01.nc',
                                      decode_times=False, autoclose=True)
        assert data01.identical(data01_orig)
