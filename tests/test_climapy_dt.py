"""
test_climapy_dt:
    Test the climapy_dt part of climapy.
    
Usage:
    Designed for use with pytest.

Author:
    Benjamin S. Grandey, 2017
"""

import climapy
import numpy as np
import pytest


class TestConvertToDatetime64:
    """Test dt_convert_to_datetime()"""

    def test_invalid_calendar(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64(1, calendar='my_calendar')

    def test_invalid_units_one(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64(1, units='my_calendar')

    def test_invalid_units_two(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64(1, units='eons since 1-1-1 0:0:0')

    def test_invalid_units_three(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64(1, units='days since the start')

    def test_invalid_data_one(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64('string')

    def test_invalid_data_two(self):
        with pytest.raises(ValueError):
            climapy.dt_convert_to_datetime64(-1)

    def test_gregorian_with_int(self):
        result = climapy.dt_convert_to_datetime64(365*10+2, units='days since 1-1-1 00:00:00',
                                                  calendar='gregorian')
        correct = np.datetime64('0011-01-01T00:00:00')  # two leap years in period
        assert result == correct

    def test_365_day_with_int(self):
        result = climapy.dt_convert_to_datetime64(365*10, units='days since 1-1-1 00:00:00',
                                                  calendar='365_day')
        correct = np.datetime64('0011-01-01')
        assert result == correct

    def test_365_day_with_float(self):
        result = climapy.dt_convert_to_datetime64(365.*1970+1.5, units='days since 1-1-1 00:00:00',
                                                  calendar='365_day')
        correct = np.datetime64('1971-01-02T12:00:00')
        assert result == correct

    def test_365_day_with_list(self):
        result = climapy.dt_convert_to_datetime64([3648, 3650, 3652],
                                                  units='days since 1-1-1 00:00:00',
                                                  calendar='365_day')
        correct = np.array(['0010-12-30', '0011-01-01', '0011-01-03'],
                           dtype='datetime64')
        assert np.array_equal(result, correct)

    def test_365_day_with_array(self):
        result = climapy.dt_convert_to_datetime64(np.array([3648., 3650., 3652.]),
                                                  units='days since 2000-01-01 00:00:00',
                                                  calendar='365_day')
        correct = np.array(['2009-12-30', '2010-01-01', '2010-01-03'],
                           dtype='datetime64')
        assert np.array_equal(result, correct)
