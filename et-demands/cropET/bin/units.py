"""Units.py
Functions for unit conversions
NOT CURRENTLY USED - PLACEHOLDER TO MOVE UNIT DEFS AND UNIT CONVERSIONS
"""

import math


def _deg2rad(deg):
    return deg * math.pi / 180.0

def _rad2deg(rad):
    return rad * 180.0 / math.pi

def _c2f(c):
    return c * (9.0 / 5) + 32

def _f2c(f):
    return (f - 32) * (5.0 / 9)


# TAKEN FROM WSWUP/REFET CODE - REPLACE UNIT CHECKING / CONVERSIONS WITH THIS CODE
        # Unit conversion
        for variable, unit in input_units.items():
            # print('  {}: {}'.format(variable, unit))

            # Check input unit types
            if unit == '':
                continue
            elif unit.lower() in [
                    'c', 'celsius',
                    'mj m-2 day-1', 'mj m-2 d-1',
                    'kpa',
                    'm s-1', 'm/s',
                    'm', 'meter', 'meters',
                    'deg', 'degree', 'degrees']:
                continue
            elif unit.lower() not in [
                    'k', 'kelvin', 'f', 'fahrenheit',
                    'pa',
                    'langleys', 'w m-2', 'w/m2',
                    'mph',
                    'ft', 'feet',
                    'rad', 'radian', 'radians']:
                raise ValueError('unsupported unit conversion for {} {}'.format(
                    variable, unit))

            # Convert input values to expected units
            if variable == 'tmax':
                if unit.lower() in ['f', 'fahrenheit']:
                    self.tmax -= 32
                    self.tmax *= (5.0 / 9)
                elif unit.lower() in ['k', 'kelvin']:
                    self.tmax -= 273.15
            elif variable == 'tmin':
                if unit.lower() in ['f', 'fahrenheit']:
                    self.tmin -= 32
                    self.tmin *= (5.0 / 9)
                elif unit.lower() in ['k', 'kelvin']:
                    self.tmin -= 273.15
            elif variable == 'ea':
                if unit.lower() in ['pa']:
                    self.ea /= 1000.0
            elif variable == 'rs':
                if unit.lower() in ['langleys']:
                    self.rs *= 0.041868
                elif unit.lower() in ['w m-2', 'w/m2']:
                    self.rs *= 0.0864
            elif variable == 'uz':
                if unit.lower() in ['mph']:
                    self.uz *= 0.44704
            elif variable == 'zw':
                if unit.lower() in ['ft', 'feet']:
                    self.zw *= 0.3048
            elif variable == 'elev':
                if unit.lower() in ['ft', 'feet']:
                    self.elev *= 0.3048
            elif variable == 'lat':
                if unit.lower() in ['rad', 'radian', 'radians']:
                    # This is a little backwards but convert to degrees so that
                    # it can be converted to radians below.  This is done so
                    # that not setting the value will default to degrees.
                    self.lat *= (180.0 / math.pi)
