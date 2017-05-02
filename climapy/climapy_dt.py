"""
climapy.climapy_dt:
    Functions associated with the dates and times, especially numpy.datetime64 objects.

Author:
    Benjamin S. Grandey, 2017
"""

import collections
import numpy as np


__all__ = ['dt_convert_to_datetime64', ]


def dt_convert_to_datetime64(data, units='days since 1-1-1 00:00:00', calendar='365_day'):
    """
    Convert numbers to array of numpy.datetime64 objects.
    
    Args:
        data: single float/int or an array (or other iterable) of floats/ints to be converted.
            Negative values cannot be accepted.
        units: string describing units of input data (default: 'days since 1-1-1 00:00:00').
        calendar: string describing calendar of input data.  Valid calendar options are
            '365_day' (default), 'no_leap' (equivalent to default), and 'gregorian'.

    Returns:
        single datetime64 object or numpy.array of dtype=datetime64.
    """
    # Check calendar
    if calendar == 'no_leap':
        calendar = '365_day'  # no_leap synonym for 365_day
    if calendar not in ['365_day', 'gregorian']:
        raise ValueError('Invalid calendar. Try "365_day" or "gregorian".')
    # Decompose units string
    if units.split()[0:2] == ['days', 'since']:
        delta = np.timedelta64(24*60*60, 's')  # allow partial days if data contains non-integers
    else:
        raise ValueError('Invalid units. Failed to decompose units string.')
    try:
        year, month, day = [int(s) for s in units.split()[2].split('-')]
        hour, minute, second = [int(s) for s in units.split()[3].split(':')]
        # Reference datetime64
        ref = np.datetime64('{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}'.format(year, month, day,
                                                                               hour, minute,
                                                                               second))
    except ValueError:
        raise ValueError('Invalid units. Failed to identify reference date and time.')
    # Convert input data to array of floats
    try:
        if isinstance(data, collections.Iterable):
            data2 = np.array(data, dtype='float64')
        else:
            data2 = np.array([data, ], dtype='float64')
    except ValueError:
        raise ValueError('Invalid data. Unable to convert to numpy.array of floats.')
    # Check that data array does not contain any negative values
    if data2.min() < 0.:
        raise ValueError('Invalid data. Negative value(s) detected. Data must be non-negative.')
    # If calendar is 365_day, correct data2 to skip Feb-29 in leap years.
    if calendar == '365_day':
        # Search for Feb-29 locations of Feb-29 between reference and a little final date-times.
        for i in range(1+int(data2.max()*1.01)):  # search beyond since correction not yet applied
            if str(ref + (i * delta))[5:10] == '02-29':
                if units.split()[0] == 'days':
                    data2[np.where(data2 > i)] += 1  # apply correction by adding a day
    # Convert data to datetime64, according to Gregorian calendar
    result = ref + (data2 * delta)
    # If input data was not an iterable, convert result to a single datetime64 object
    if not isinstance(data, collections.Iterable):
        result = result[0]
    return result
