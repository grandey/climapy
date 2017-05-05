"""
climapy.climapy_cesm:
    Functions to support analysis of Community Earth System Model (CESM) output.
    
Author:
    Benjamin S. Grandey, 2017
"""

import climapy


__all__ = ['cesm_time_from_bnds', ]


def cesm_time_from_bnds(xr_data, min_year=1701):
    """
    Use mid-points from time_bnds in CESM output data to populate time dimension with
    numpy.datetime64 values.
    
    Args:
        xr_data: xarray Dataset containing CESM output data.
        min_year: integer specifying minimum year to accept for reference date, derived from units.
            If the reference year is less than min_year, then the reference year will be changed to
            min_year. This allows dates to be shifted to be greater than e.g. 1678. (default 1701)

    Returns:
        Copy of xr_data, with modified time and time_bnds dimensions.
    """
    data = xr_data.copy()
    try:
        time_bnds_mid = data['time_bnds'].mean(dim='bnds')
    except ValueError:
        time_bnds_mid = data['time_bnds'].mean(dim='nb2')
    data['time'].values = climapy.dt_convert_to_datetime64(time_bnds_mid,
                                                           units=data['time'].units,
                                                           calendar=data['time'].calendar,
                                                           min_year=min_year)
    data['time_bnds'].values = climapy.dt_convert_to_datetime64(data['time_bnds'],
                                                                units=data['time'].units,
                                                                calendar=data['time'].calendar,
                                                                min_year=min_year)
    return data
