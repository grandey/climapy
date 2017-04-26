"""
Test the climapy_xr part of climapy.

Designed for use with pytest.

Contents:
1. Prerequisite functions
2. Loading of test data into dictionaries
3. The test classes

History:
- 2017-04 - Benjamin S. Grandey
"""

import climapy
from copy import deepcopy
import numpy as np
import os
#import pytest
import xarray as xr


# ------------------------------
# Part 1: prerequisite functions
# ------------------------------

def prepare_test_data():
    """
    Load data01.nc and manipulate to create additional test data.
    Used to load data into data_dict below.
    """
    # Dictionary in which to store data
    data_dict = {}
    # Load data01.nc Dataset
    data01 = xr.open_dataset(os.path.dirname(__file__)+'/data/data01.nc',
                             decode_times=False, autoclose=True)
    data_dict['data01'] = data01.copy()
    # Extract two *DataArrays* - to test functions with DataArrays
    da_ts = data01['TS'].copy()
    da_precl = data01['PRECL'].copy()
    data_dict['da_ts'] = da_ts.copy()
    data_dict['da_precl'] = da_precl.copy()
    # Dataset with *shifted* longitudes
    ds_shift_lon = climapy.xr_shift_lon(data01.copy())
    data_dict['ds_shift_lon'] = ds_shift_lon.copy()
    # Datasets with *reversed* lon/lat coordinates
    ds_rev_lon = data01.copy()
    ds_rev_lon['lon'].values = ds_rev_lon['lon'].values[::-1]
    ds_rev_lat = data01.copy()
    ds_rev_lat['lat'].values = ds_rev_lat['lat'].values[::-1]
    ds_rev_both = data01.copy()
    ds_rev_both['lat'].values = ds_rev_both['lat'].values[::-1]
    ds_rev_both['lon'].values = ds_rev_both['lon'].values[::-1]
    data_dict['ds_rev_lon'] = ds_rev_lon.copy()
    data_dict['ds_rev_lat'] = ds_rev_lat.copy()
    data_dict['ds_rev_both'] = ds_rev_both.copy()
    # Dataset with *transposed* lon/lat coords
    ds_transposed = data01.copy()
    ds_transposed = ds_transposed.transpose()
    data_dict['ds_transposed'] = ds_transposed.copy()
    # Dataset with *renamed* longitude and latitude coords
    ds_renamed = data01.copy()
    ds_renamed = ds_renamed.rename({'lon': 'longitude', 'lat': 'latitude'})
    data_dict['ds_renamed'] = ds_renamed.copy()
    # Datasets with slightly *irregular* lon/lat coords, yet still monotonic
    nx, ny = data01['lon'].size, data01['lat'].size
    lon_irr = (data01['lon'].values +
               np.random.uniform(low=-0.5, high=0.5, size=nx))  # add small amount of noise
    lon_irr[[0, -1]] = data01['lon'].values[[0, -1]]  # keep end values unchanged
    lat_irr = (data01['lat'].values +
               np.random.uniform(low=-0.5, high=0.5, size=ny))
    lat_irr[[0, -1]] = data01['lat'].values[[0, -1]]
    ds_irr_lon = data01.copy()
    ds_irr_lon['lon'].values = lon_irr.copy()
    ds_irr_lat = data01.copy()
    ds_irr_lat['lat'].values = lat_irr.copy()
    ds_irr_both = data01.copy()
    ds_irr_both['lon'].values = lon_irr.copy()
    ds_irr_both['lat'].values = lat_irr.copy()
    data_dict['ds_irr_lon'] = ds_irr_lon.copy()
    data_dict['ds_irr_lat'] = ds_irr_lat.copy()
    data_dict['ds_irr_both'] = ds_irr_both.copy()
    # Dataset with *strange* lon/lat coords - very irregular and not monotonic
    lon_strange = (data01['lon'].values +
                   np.random.uniform(low=-10, high=10, size=nx))  # add large amount of noise
    lon_strange[[0, -1]] = data01['lon'].values[[0, -1]]  # keep end values unchanged
    lat_strange = (data01['lat'].values + np.random.uniform(low=-10, high=10, size=ny))
    lat_strange[[0, -1]] = data01['lat'].values[[0, -1]]  # keep end values unchanged
    ds_strange = data01.copy()
    ds_strange['lon'].values = lon_strange.copy()
    ds_strange['lat'].values = lat_strange.copy()
    data_dict['ds_strange'] = ds_strange.copy()
    # Return dictionary of data
    return data_dict


def load_region_bounds_dict():
    """
    Load dictionary of region bounds, used during testing.
    Used by load_cdo_results(), TestMaskBounds, and TestAreaWeightedStat below.
    """
    region_bounds_dict = {'EAs': [(94, 156), (20, 65)],  # longitude tuple, latitude tuple
                          'SEAs': [(94, 161), (-10, 20)],
                          'ANZ': [(109, 179), (-50, -10)],
                          'SAs': [(61, 94), (0, 35)],
                          'AfME': [(-21, 61), (-40, 35)],
                          'Eur': [(-26, 31), (35, 75)],
                          'CAs': [(31, 94), (35, 75)],
                          'NAm': [(-169, -51), (15, 75)],
                          'SAm': [(266, 329), (-60, 15)],
                          'Zon': [None, (-75.5, -65.5)],
                          'Mer': [(175.5, 185.5), None],
                          'Glb': [None, None]}
    return region_bounds_dict


def load_cdo_results():
    """
    Load CDO-calculated 'truths'.
    Used to load data into copy_dict below.
    """
    # Location of data files
    cdo_dir = os.path.dirname(__file__)+'/data/cdo_results/'
    # Dictionary in which to store data
    cdo_dict = {}
    # Load gridcell area data
    cdo_dict['gridarea'] = xr.open_dataset(cdo_dir+'data01_gridarea.nc',
                                           decode_times=False, autoclose=True)
    # Load data for regions
    for region in load_region_bounds_dict().keys():
        for suffix in ['', '_area', '_fldmean']:
            key = region + suffix
            if key != 'Glb':  # data01_Glb.nc does not exist
                cdo_dict[key] = xr.open_dataset(cdo_dir+'data01_'+key+'.nc',
                                                decode_times=False, autoclose=True)
    # Return dictioary of data
    return cdo_dict


def check_data_dict_identical(data_dict_1, data_dict_2):
    """
    Check whether data_dicts are identical.
    Used by TestDataUnchanged below.
    """
    result = True  # assume True, unless proven otherwise
    if data_dict_1.keys() != data_dict_2.keys():
        result = False
    for key in data_dict_1.keys():
        if data_dict_1[key].identical(data_dict_2[key]) is not True:
            result = False
    return result


# -------------------------------------------------------
# Part 2: load test data, and "truths", into dictionaries
# -------------------------------------------------------

data_dict = prepare_test_data()  # Input data for tests
copy_dict = deepcopy(data_dict)  # Copy of input data - used by TestDataUnchanged
cdo_dict = load_cdo_results()  # CDO-derived "truths"


# --------------------
# Part 3: test classes
# --------------------

class TestDataDict:
    """Check that input data disctionary has been prepared successfully."""

    def test_keys(self):
        assert set(data_dict.keys()) == set(['data01', 'da_ts', 'da_precl', 'ds_shift_lon',
                                             'ds_rev_lon', 'ds_rev_lat', 'ds_rev_both',
                                             'ds_transposed', 'ds_renamed',
                                             'ds_irr_lon', 'ds_irr_lat', 'ds_irr_both',
                                             'ds_strange'])

    def test_type(self):
        for key, data in data_dict.items():
            if key[0:3] == 'da_':
                assert isinstance(data, xr.core.dataarray.DataArray)
            else:
                assert isinstance(data, xr.core.dataset.Dataset)

    def test_lon_ends(self):
        for key, data in data_dict.items():
            # Get start and final longitudes of data
            if key == 'ds_renamed':
                lon_ends = data['longitude'].values[[0, -1]].tolist()
            else:
                lon_ends = data['lon'].values[[0, -1]].tolist()
            # Compare against correct answer
            if key == 'ds_shift_lon':
                assert lon_ends == [-180, 177.5]
            elif key in ['ds_rev_lon', 'ds_rev_both']:
                assert lon_ends == [357.5, 0]
            else:
                assert lon_ends == [0, 357.5]

    def test_lat_ends(self):
        for key, data in data_dict.items():
            # Get start and final latitudes of data
            if key == 'ds_renamed':
                lat_ends = data['latitude'].values[[0, -1]].tolist()
            else:
                lat_ends = data['lat'].values[[0, -1]].tolist()
            # Compare against correct answer
            if key in ['ds_rev_lat', 'ds_rev_both']:
                assert lat_ends == [90, -90]
            else:
                assert lat_ends == [-90, 90]


class TestCheckLonLatMatch:
    """Test xr_check_lon_lat_match()"""
    pass


class TestShiftLon:
    """Test xr_shift_lon()"""
    pass


class TestArea:
    """Test xr_area()"""
    pass


class TestMaskBounds:
    """Test xr_mask_bounds()"""
    pass


class TestAreaWeightedStat:
    """Test xr_area_weighted_stat()"""
    pass


class TestDataUnchanged:
    """Check that data_dict has not been changed inplace."""
    def test_one(self):
        assert check_data_dict_identical(data_dict, copy_dict)
