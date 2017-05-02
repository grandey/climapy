# climapy

## Purpose:
Support data analysis of climate model data.

## Status:
Work in progress.

## Installation:
Since I anticipate that I will be the only user of **climapy**, I have not created a setup.py file. Instead, I just include a symbolic link to a local copy of **climapy** in a directory in my $PYTHONPATH.

## Testing:
Tests, for use with pytest, are contained in tests/. The tests can be run from the root directory in the repository:
```
pytest
```

## Functions contained in climapy:
```
    dt_convert_to_datetime64(data, units='days since 1-1-1 00:00:00', calendar='365_day'):
        Convert numbers to array of numpy.datetime64 objects.

    stats_fdr(p_values, alpha=0.10):
        Control the false discovery rate (FDR). Useful when applying multiple hypothesis tests.

    xr_check_lon_lat_match(xr_data_1, xr_data_2, lon_name='lon', lat_name='lat'):
        Check whether longitude and latitude coordinates of xarray Datasets/DataArrays are equal.

    xr_shift_lon(xr_data, lon_min=-180., lon_name='lon'):
        Shift longitudes of an xarray Dataset or DataArray.

    xr_area(xr_data, lon_name='lon', lat_name='lat'):
        Calculate grid-cell areas of an xarray Dataset or DataArray.

    xr_mask_bounds(xr_data, lon_bounds=(-180, 180), lat_bounds=(-90, 90), select_how='inside',
                   lon_name='lon', lat_name='lat'):
        Select inside/outside specified region bounds, and mask elsewhere.

    xr_area_weighted_stat(xr_data, stat='mean', lon_bounds=None, lat_bounds=None,
                          lon_name='lon', lat_name='lat'):
        Calculate area-weighted mean or sum across globe (default) or a specified region.
```

## Author:
Benjamin S. Grandey, 2017
