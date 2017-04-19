"""
climapy.climapy_xr:
Functions that act on xarray Datasets and DataArrays.

Author:
Benjamin S. Grandey, 2017
"""

import numpy as np
import warnings
import xarray as xr


__all__ = ['xr_check_lon_lat_match', 'xr_shift_lon', 'xr_area', 'xr_mask_bounds',
           'xr_area_weighted_stat']


def xr_check_lon_lat_match(xr_data_1, xr_data_2, lon_name='lon', lat_name='lat'):
    """
    Check whether longitude and latitude coordinates are equal.

    Keyword arguments:
    xr_data_1 -- first xarray Dataset or DataArray, with longitude and latitude dimensions
    xr_data_2 -- second xarray Dataset or DataArray, with longitude and latitude dimensions
    lon_name -- the name of the longitude dimension and coordinate (default 'lon')
    lat_name -- the name of the longitude dimension and coordinate (default 'lon')

    Returns:
    True or False
    """
    result = True  # start by assuming True; modify to False if data fails tests
    if (xr_data_1[lon_name].values != xr_data_2[lon_name].values).any():
        warnings.warn('Longitudes not equal')
        result = False
    if (xr_data_1[lat_name].values != xr_data_2[lat_name].values).any():
        warnings.warn('Latitudes not equal')
        result = False
    return result


def xr_shift_lon(xr_data, lon_min=-180., lon_name='lon'):
    """
    Shift longitudes of an xarray Dataset or DataArray.

    Keyword arguments:
    xr_data -- an xarray Dataset or DataArray, with a longitude dimension
    lon_min -- the minimum longitude requested (default -180.)
    lon_name -- the name of the longitude dimension (default 'lon')

    Returns:
    Copy of input object, with longitudes shifted to the range lon_min to lon_min+360.
    """
    data = xr_data.copy()
    lon = data[lon_name]  # input longitude data
    if np.all(np.diff(lon) > 0):  # check for mononotonic increase ...
        increasing = True
    elif np.all(np.diff(lon) < 0):  # ... or monotonic decrease
        increasing = False
    else:
        raise ValueError('Input longitudes must increase or decrease monotonicically')
    add_360 = np.where(lon < lon_min, 360, 0)  # any longitudes outside valid range?
    sub_360 = np.where(lon >= (lon_min+360), -360, 0)
    if (add_360 + sub_360).any():
        data[lon_name] = lon + add_360 + sub_360  # correct longitudes to be within valid range
        if increasing:  # find location where minimum longitude value occurs
            x = np.where(data[lon_name] == data[lon_name].min())[0][0]
        else:  # if decreasing, find location where maximum longitude value occurs
            x = np.where(data[lon_name] == data[lon_name].max())[0][0]
        args = {lon_name: -x}
        data = data.roll(**args)  # shift data
    return data


def xr_area(xr_data, lon_name='lon', lat_name='lat'):
    """
    Calculate grid-cell areas of an xarray Dataset or DataArray.

    Keyword arguments:
    xr_data -- an xarray Dataset or DataArray, with lon and lat dimensions
    lon_name -- the name of the longitude dimension and coordinate (default 'lon')
    lat_name -- the name of the longitude dimension and coordinate (default 'lon')

    Returns:
    xarray DataArray named 'area', containing grid-cell areas.
    """
    radius = 6371 * 1e3  # Earth's radius in m
    lon = xr_data[lon_name].values
    lat = xr_data[lat_name].values
    nx, ny = len(lon), len(lat)
    # Longitude boundaries of grid cells, using linear interpolation
    lon_extended = np.concatenate([lon[[0]]-(lon[1]-lon[0]),  # extrapolate end elements
                                   lon[:],
                                   lon[[-1]]+(lon[-1]-lon[-2])])
    for x in range(len(lon_extended)-1):
        if lon_extended[x+1] < lon_extended[x]:
            lon_extended[x+1] += 360  # correct lon_extended to increase monotonically
    if np.diff(lon_extended).min() <= 0:
        warnings.warn('Longitudes not increasing monotonically!')
    lon_bounds = np.interp(np.arange(nx+1)+0.5, np.arange(nx+2),
                           lon_extended)  # longitude boundaries
    # Zonal width of grid cells in terms of longitude
    x_width = np.diff(lon_bounds)
    if x_width.max() > (2*x_width.min()):
        warnings.warn('Max longitude width ({}) > '
                      '2x min longitude width ({})'.format(x_width.max(), x_width.min()))
    # Latitude boundaries of grid cells, using linear interpolation
    if np.diff(lat).min() <= 0:
        warnings.warn('Latitudes not increasing monotonically!')
    lat_bounds = np.interp(np.arange(ny+1)-0.5, np.arange(ny), lat)
    if lat_bounds.min() < -90.:
        lat_bounds[np.where(lat_bounds < -90.)[0]] = -90.  # set min latitude bound to -90
    if lat_bounds.max() > 90.:
        lat_bounds[np.where(lat_bounds > 90.)[0]] = 90.  # set max latitude bound to 90
    # Meridional width of grid cells in terms of sin(latitude)
    y_width = np.diff(np.sin(lat_bounds/180*np.pi))
    # Convert x_width and y_width into xarray DataArrays for automatic broadcasting
    x_width = xr.DataArray(x_width, coords={lon_name: lon}, dims=(lon_name, ))
    y_width = xr.DataArray(y_width, coords={lat_name: lat}, dims=(lat_name, ))
    # Calculate surface area of grid cells
    # Ref: https://badc.nerc.ac.uk/help/coordinates/cell-surf-area.html
    area = radius**2 * y_width * (x_width / 180 * np.pi)
    # Name area DataArray and add units
    area = area.rename('area')
    area.attrs['units'] = 'm2'
    # Check that longitude and latitude coords same as input
    if xr_check_lon_lat_match(xr_data, area) is not True:
        warnings.warn('Longitudes and/or latitudes not equal.')
    # Sanity check: compare sum to surface area of a sphere
    correct_answer = radius**2 * 4 * np.pi
    perc_diff = 100 * (area.values.sum() - correct_answer) / correct_answer
    if abs(perc_diff) > 1e-4:
        warnings.warn('Total area calculated differs from '
                      'spherical Earth by {}%'.format(perc_diff))
    # Return result
    return area


def xr_mask_bounds(xr_data, lon_bounds=(-180, 180), lat_bounds=(-90, 90), select_how='inside',
                   lon_name='lon', lat_name='lat'):
    """
    Select inside/outside specified region bounds, and mask elsewhere.

    Keyword arguments:
    xr_data -- an xarray Dataset or DataArray, with longitude and latitude dimensions
    lon_bounds -- tuple/list containing longitude bounds (default (-180, 180))
    lat_bounds -- tuple/list containing latitude bounds (default (-90, 90))
    select_how -- select data either 'inside' region (ie mask outside; default) or 'outside' region
    lon_name -- the name of the longitude dimension and coordinate (default 'lon')
    lat_name -- the name of the longitude dimension and coordinate (default 'lon')

    Returns:
    Copy of input object, with masking applied.
    """
    data = xr_data.copy()
    # If lon_bounds or lat_bounds are set to None, then use defaults
    if lon_bounds is None:
        lon_bounds = (-180, 180)
    if lat_bounds is None:
        lat_bounds = (-90, 90)
    # Shift longitudes for consistency with lon_bounds specification
    orig_lon_min = data[lon_name].values.min()  # current minimum longitude used later
    data = xr_shift_lon(data, lon_min=lon_bounds[0])
    # Selecting inside bounds - lon and lat can be masked separately
    if select_how == 'inside':
        data = data.where((data[lon_name] >= lon_bounds[0]) &
                          (data[lon_name] <= lon_bounds[1]))
        data = data.where((data[lat_name] >= lat_bounds[0]) &
                          (data[lat_name] <= lat_bounds[1]))
    # Selecting outside bounds - lon and lat should be masked together
    elif select_how == 'outside':
        data = data.where((data[lon_name] < lon_bounds[0]) |
                          (data[lon_name] > lon_bounds[1]) |
                          (data[lat_name] < lat_bounds[0]) |
                          (data[lat_name] > lat_bounds[1]))
    # Warning if mask_how is not recognised
    else:
        warnings.warn('select_how="{}" is unrecognised option, so returning -1'.format(select_how))
        return -1
    # Shift longitudes back to original array
    data = xr_shift_lon(data, lon_min=orig_lon_min)
    # Check that longitude and latitude coords same as input
    if xr_check_lon_lat_match(xr_data, data) is not True:
        warnings.warn('Longitudes and/or latitudes not equal.')
    return data


def xr_area_weighted_stat(xr_data, stat='mean', lon_bounds=None, lat_bounds=None,
                          lon_name='lon', lat_name='lat'):
    """
    Calculate area-weighted mean or sum across globe (default) or a specified region.

    Keyword arguments:
    xr_data -- an xarray Dataset or DataArray, with longitude and latitude dimensions
    stat -- statistic to calculate, either 'mean' (default) or 'sum'
    lon_bounds -- tuple/list containing longitude bounds (default None)
    lat_bounds -- tuple/list containing latitude bounds (default None)
    lon_name -- the name of the longitude dimension and coordinate (default 'lon')
    lat_name -- the name of the longitude dimension and coordinate (default 'lon')

    Returns:
    Area-weighted mean or sum
    """
    # Apply region masking?
    if (lon_bounds is not None) or (lat_bounds is not None):
        data = xr_mask_bounds(xr_data, lon_bounds=lon_bounds, lat_bounds=lat_bounds,
                              select_how='inside', lon_name=lon_name, lat_name=lat_name)
    else:
        data = xr_data.copy()
    # Get grid cell area and double-check that longitudes and latitudes match data exactly
    area = xr_area(xr_data, lon_name=lon_name, lat_name=lat_name)
    if xr_check_lon_lat_match(xr_data, area) is not True:
        warnings.warn('Longitudes and/or latitudes not equal.')
    # Calculation depends on requested statistic
    if stat == 'sum':
        data = (data * area).sum(dim=[lon_name, lat_name])
    elif stat == 'mean':
        data = (data * area).sum(dim=[lon_name, lat_name]) / area.sum(dim=[lon_name, lat_name])
    else:
        warnings.warn('stat="{}" is unrecognised option, so returning -1'.format(stat))
        return -1
    data = float(data.values)  # get value
    return data
