"""
test_climapy_xr:
    Test the climapy_xr part of climapy.

Usage:
    Designed for use with pytest.

Contents:
    1. Prerequisite functions
    2. Loading of test data into dictionaries
    3. The test classes

Author:
    Benjamin S. Grandey, 2017
"""

import climapy
from copy import deepcopy
import numpy as np
import os
import pytest
import xarray as xr


# ------------------------------
# Part 1: prerequisite functions
# ------------------------------

np_rand = np.random.RandomState(2435)  # seed random numbers, for reproducibility


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
    # Datasets with *reversed* lon/lat coordinates and data
    ds_rev_lon = data01.copy()
    ds_rev_lon['lon'].values = ds_rev_lon['lon'].values[::-1]
    for var_name in ['TS', 'PRECL']:  # array order: time, lat, lon
        ds_rev_lon[var_name].values = ds_rev_lon[var_name].values[:, :, ::-1]
    ds_rev_lat = data01.copy()
    ds_rev_lat['lat'].values = ds_rev_lat['lat'].values[::-1]
    for var_name in ['TS', 'PRECL']:
        ds_rev_lat[var_name].values = ds_rev_lat[var_name].values[:, ::-1, :]
    ds_rev_both = data01.copy()
    ds_rev_both['lat'].values = ds_rev_both['lat'].values[::-1]
    ds_rev_both['lon'].values = ds_rev_both['lon'].values[::-1]
    for var_name in ['TS', 'PRECL']:
        ds_rev_both[var_name].values = ds_rev_both[var_name].values[:, ::-1, ::-1]
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
               np_rand.uniform(low=-0.5, high=0.5, size=nx))  # add small amount of noise
    lon_irr[[0, -1]] = data01['lon'].values[[0, -1]]  # keep end values unchanged
    lat_irr = (data01['lat'].values +
               np_rand.uniform(low=-0.5, high=0.5, size=ny))
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
                   np_rand.uniform(low=-10, high=10, size=nx))  # add large amount of noise
    lon_strange[[0, -1]] = data01['lon'].values[[0, -1]]  # keep end values unchanged
    lat_strange = (data01['lat'].values + np_rand.uniform(low=-10, high=10, size=ny))
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
            if key == 'Glb':  # for globe is data01.nc
                cdo_dict[key] = xr.open_dataset(cdo_dir+'../data01.nc',
                                                decode_times=False, autoclose=True)
            else:
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
    """Check that input data dictionary has been prepared successfully."""

    def test_keys(self):
        assert set(data_dict.keys()) == {'data01', 'da_ts', 'da_precl', 'ds_shift_lon',
                                         'ds_rev_lon', 'ds_rev_lat', 'ds_rev_both',
                                         'ds_transposed', 'ds_renamed',
                                         'ds_irr_lon', 'ds_irr_lat', 'ds_irr_both',
                                         'ds_strange'}

    def test_type(self):
        for key, data in data_dict.items():
            if key[0:3] == 'da_':
                assert isinstance(data, xr.DataArray)
            else:
                assert isinstance(data, xr.Dataset)

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

    def test_against_data01(self):
        ref_data = data_dict['data01']
        for key, data in data_dict.items():
            if key == 'ds_renamed':
                with pytest.raises(KeyError):
                    climapy.xr_check_lon_lat_match(ref_data, data)
            elif key in ['data01', 'da_ts', 'da_precl', 'ds_transposed']:
                assert climapy.xr_check_lon_lat_match(ref_data, data)
            else:
                assert climapy.xr_check_lon_lat_match(ref_data, data) is False

    def test_against_shift_lon(self):
        ref_data = data_dict['ds_shift_lon']
        for key, data in data_dict.items():
            if key == 'ds_renamed':
                with pytest.raises(KeyError):
                    climapy.xr_check_lon_lat_match(ref_data, data)
            elif key == 'ds_shift_lon':
                assert climapy.xr_check_lon_lat_match(ref_data, data)
            else:
                assert climapy.xr_check_lon_lat_match(ref_data, data) is False

    def test_against_irr_both(self):
        ref_data = data_dict['ds_irr_both']
        for key, data in data_dict.items():
            if key == 'ds_renamed':
                with pytest.raises(KeyError):
                    climapy.xr_check_lon_lat_match(ref_data, data)
            elif key == 'ds_irr_both':
                assert climapy.xr_check_lon_lat_match(ref_data, data)
            else:
                assert climapy.xr_check_lon_lat_match(ref_data, data) is False

    def test_against_renamed(self):
        ref_data = data_dict['ds_renamed']
        for key, data in data_dict.items():
            if key == 'ds_renamed':
                assert climapy.xr_check_lon_lat_match(ref_data, data,
                                                      lon_name='longitude', lat_name='latitude')
            else:
                with pytest.raises(KeyError):
                    climapy.xr_check_lon_lat_match(ref_data, data,
                                                   lon_name='longitude', lat_name='latitude')


class TestShiftLon:
    """Test xr_shift_lon()"""

    def test_incorrect_lon_name(self):
        with pytest.raises(KeyError):
            climapy.xr_shift_lon(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_shift_lon(data_dict['data01'], lon_name='longitude')

    def test_non_monotonic(self):
        with pytest.raises(ValueError):
            climapy.xr_shift_lon(data_dict['ds_strange'])

    def test_default_shift(self):
        for key, data in data_dict.items():
            if key != 'ds_strange':
                if key == 'ds_renamed':
                    new_lon = climapy.xr_shift_lon(data, lon_name='longitude')['longitude'].values
                elif key in ['ds_rev_lon', 'ds_rev_both']:
                    new_lon = climapy.xr_shift_lon(data)['lon'].values[::-1]
                else:
                    new_lon = climapy.xr_shift_lon(data)['lon'].values
                assert new_lon.min() == new_lon[0]
                assert new_lon.max() == new_lon[-1]
                if key in ['ds_irr_lon', 'ds_irr_both']:
                    assert -180 <= new_lon[0] <= -177  # allow some leeway for irreg longitudes
                    assert 177 <= new_lon[-1] <= 180
                else:
                    assert new_lon[0] == -180, AssertionError(key)
                    assert new_lon[-1] == 177.5, AssertionError(key)

    def test_shift_of_m90(self):
        for key, data in data_dict.items():
            if key != 'ds_strange':
                if key == 'ds_renamed':
                    new_lon = climapy.xr_shift_lon(data, lon_min=-90,
                                                   lon_name='longitude')['longitude'].values
                elif key in ['ds_rev_lon', 'ds_rev_both']:
                    new_lon = climapy.xr_shift_lon(data, lon_min=-90)['lon'].values[::-1]
                else:
                    new_lon = climapy.xr_shift_lon(data, lon_min=-90)['lon'].values
                assert new_lon.min() == new_lon[0]
                assert new_lon.max() == new_lon[-1]
                if key in ['ds_irr_lon', 'ds_irr_both']:
                    assert -90 <= new_lon[0] <= -87  # allow some leeway for irreg longitudes
                    assert 267 <= new_lon[-1] <= 270
                else:
                    assert new_lon[0] == -90, AssertionError(key)
                    assert new_lon[-1] == 267.5, AssertionError(key)

    def test_shift_back(self):
        for key, data in data_dict.items():
            if key not in ['ds_strange', 'ds_renamed']:
                orig_lon_min = data['lon'].values.min()  # original min longitude
                for lon_min in [-180., -270., 0., 90.]:  # try different lon_min values
                    new_data = climapy.xr_shift_lon(data, lon_min=lon_min)
                    new_data = climapy.xr_shift_lon(new_data,
                                                    lon_min=orig_lon_min)  # shift back
                    # Correct small diffs in coords before comparing
                    new_data = new_data.reindex_like(data, method='nearest', tolerance=1e-3)
                    assert new_data.equals(data)  # compare to original


class TestArea:
    """Test xr_area()"""

    def test_incorrect_lon_name(self):
        with pytest.raises(KeyError):
            climapy.xr_area(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_area(data_dict['data01'], lon_name='longitude')

    def test_incorrect_lat_name(self):
        with pytest.raises(KeyError):
            climapy.xr_area(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_area(data_dict['data01'], lat_name='latitude')

    def test_non_monotonic(self):
        with pytest.raises(ValueError):
            climapy.xr_area(data_dict['ds_strange'])

    def test_area_values(self):
        cdo_values = cdo_dict['gridarea']['cell_area'].values
        area_values = climapy.xr_area(data_dict['data01']).values
        rel_diff = (area_values - cdo_values) / cdo_values  # relative difference
        assert np.abs(rel_diff).max() < 1e-3  # check that diffs are small

    def test_global_sum(self):
        cdo_sum = cdo_dict['gridarea']['cell_area'].values.sum()
        for key, data in data_dict.items():
            if key != 'ds_strange':
                if key == 'ds_renamed':
                    area_sum = climapy.xr_area(data_dict[key], lon_name='longitude',
                                               lat_name='latitude').values.sum()
                else:
                    area_sum = climapy.xr_area(data_dict[key]).values.sum()
                rel_diff = (area_sum - cdo_sum) / cdo_sum  # relative difference
                if key in ['ds_irr_lon', 'ds_irr_lat',
                           'ds_irr_both']:  # more leeway allowed for irregular coords
                    assert abs(rel_diff) < 1e-3, AssertionError(key)
                else:
                    assert abs(rel_diff) < 1e-6, AssertionError(key)


class TestMaskBounds:
    """Test xr_mask_bounds()"""

    def test_incorrect_lon_name(self):
        with pytest.raises(KeyError):
            climapy.xr_mask_bounds(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_mask_bounds(data_dict['data01'], lon_name='longitude')

    def test_incorrect_lat_name(self):
        with pytest.raises(KeyError):
            climapy.xr_mask_bounds(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_mask_bounds(data_dict['data01'], lat_name='latitude')

    def test_non_monotonic(self):
        with pytest.raises(ValueError):
            climapy.xr_mask_bounds(data_dict['ds_strange'])

    def test_inside_values(self):
        for region, bounds in load_region_bounds_dict().items():
            cdo_data = cdo_dict[region]
            lon_bounds, lat_bounds = bounds
            mask_data = climapy.xr_mask_bounds(data_dict['data01'],
                                               lon_bounds=lon_bounds, lat_bounds=lat_bounds,
                                               select_how='inside')
            mask_data = mask_data.dropna(dim='lon',  # drop NaN rows/columns, like CDO
                                         how='all').dropna(dim='lat', how='all')
            mask_data = climapy.xr_shift_lon(mask_data,  # shift lons for consistency with cdo_data
                                             lon_min=cdo_data['lon'].min())
            rel_diff = ((mask_data['TS'].values - cdo_data['TS'].values) /
                        cdo_data['TS'].values)  # relative difference in 'TS' variable
            assert np.abs(rel_diff).max() < 1e-12  # check that differences very small

    def test_outside_plus_inside(self):
        """Test how='outside' by checking that input data can be reconstructed."""
        for region, bounds in load_region_bounds_dict().items():
            lon_bounds, lat_bounds = bounds
            for key in ['data01', 'ds_shift_lon', 'ds_rev_both', 'ds_irr_both']:
                outside_data = climapy.xr_mask_bounds(data_dict[key],
                                                      lon_bounds=lon_bounds, lat_bounds=lat_bounds,
                                                      select_how='outside')['PRECL']
                inside_data = climapy.xr_mask_bounds(data_dict[key],
                                                     lon_bounds=lon_bounds, lat_bounds=lat_bounds,
                                                     select_how='inside')['PRECL']
                outside_plus_inside = (np.nan_to_num(outside_data.values) +
                                       np.nan_to_num(inside_data.values))
                diff_from_input = outside_plus_inside - data_dict[key]['PRECL'].values
                assert np.abs(diff_from_input).max() == 0

    # Implicit further testing of xr_mask_bounds() by TestAreaWeightedStat below...


class TestAreaWeightedStat:
    """Test xr_area_weighted_stat()"""

    def test_incorrect_lon_name(self):
        with pytest.raises(KeyError):
            climapy.xr_area_weighted_stat(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_area_weighted_stat(data_dict['data01'], lon_name='longitude')

    def test_incorrect_lat_name(self):
        with pytest.raises(KeyError):
            climapy.xr_area_weighted_stat(data_dict['ds_renamed'])
        with pytest.raises(KeyError):
            climapy.xr_area_weighted_stat(data_dict['data01'], lat_name='latitude')

    def test_non_monotonic(self):
        with pytest.raises(ValueError):
            climapy.xr_area_weighted_stat(data_dict['ds_strange'])

    def test_sum_of_area(self):
        for region, bounds in load_region_bounds_dict().items():
            lon_bounds, lat_bounds = bounds
            cdo_area = cdo_dict[region+'_area']['cell_area'].values.flatten()[0]  # "truth"
            for key, data in data_dict.items():
                if key not in ['ds_strange', 'da_ts', 'da_precl']:
                    if key == 'ds_renamed':
                        lon_name, lat_name = 'longitude', 'latitude'
                    else:
                        lon_name, lat_name = 'lon', 'lat'
                    ds_ones = data_dict[key]['TS'] * 0 + 1  # set values to 1
                    area = climapy.xr_area_weighted_stat(ds_ones, stat='sum',
                                                         lon_bounds=lon_bounds,
                                                         lat_bounds=lat_bounds,
                                                         lon_name=lon_name,
                                                         lat_name=lat_name).values[0]  # time 0
                    rel_diff = (cdo_area - area) / cdo_area  # relative diff
                    if key in ['ds_irr_lon', 'ds_irr_lat', 'ds_irr_both']:
                        if region in ['Zon', 'Mer']:  # irregular coords need more leeway
                            assert abs(rel_diff) < 0.5, AssertionError(region, key)
                        else:
                            assert abs(rel_diff) < 0.1, AssertionError(region, key)
                    else:  # stricter for regular coords
                        assert abs(rel_diff) < 1e-3, AssertionError(region, key)

    def test_area_weighted_mean(self):
        for region, bounds in load_region_bounds_dict().items():
            lon_bounds, lat_bounds = bounds
            cdo_awm = cdo_dict[region+'_fldmean']['PRECL'].values.flatten()
            for key, data in data_dict.items():
                if key not in ['ds_strange', 'da_precl', 'da_ts',
                               'ds_irr_lon', 'ds_irr_lat', 'ds_irr_both']:
                    if key == 'ds_renamed':
                        lon_name, lat_name = 'longitude', 'latitude'
                    else:
                        lon_name, lat_name = 'lon', 'lat'
                    awm = climapy.xr_area_weighted_stat(data_dict[key],
                                                        lon_bounds=lon_bounds,
                                                        lat_bounds=lat_bounds,
                                                        lon_name=lon_name,
                                                        lat_name=lat_name)['PRECL'].values
                    rel_diff = (awm - cdo_awm) / cdo_awm  # relative diff at each timestep
                    max_rel_diff = np.abs(rel_diff).max()
                    assert abs(max_rel_diff) < 1e-3, AssertionError(region, key)


class TestDataUnchanged:
    """Check that data_dict has not been changed inplace."""
    def test_data_unchanged(self):
        assert check_data_dict_identical(data_dict, copy_dict)
