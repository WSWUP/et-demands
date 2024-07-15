.. _model-description:

=================
Model Description
=================
The ET-Demands package

.. _model-description-refet:

RefET
-----

Reference evapotranspiration is calculated according to the ASCE Standardized
Reference Evapotranspiration Equation `(ASCE-EWRI, 2005) <https://doi.org/10.1061/9780784408056>`_.


Gap Filling and QA/QC
^^^^^^^^^^^^^^^^^^^^^

Missing Data
""""""""""""
Missing values of maximum air temperature (T\ :sub:`max`), minimum air temperature
(T\ :sub:`min`), and mean wind speed (u\ :sub:`x`), up to six timesteps, are first
filled through linear interpolation. Additional missing values not handled by the linear
interpolation are filled using mean monthly values. Missing values of precipitation
(P\ :sub:`r`), snow, (S\ :sub:`n`), and snow depth (S\ :sub:`d`) are set to 0.

Maximum and Minimum Air Temperature
"""""""""""""""""""""""""""""""""""
Maximum air temperature (T\ :sub:`max`) values greater than 120°F are set to 120°F.
Minimum air temperature (T\ :sub:`min`) values greater than 90°F are set to 90°F.
Maximum air temperature is checked against minimum air temperature at every time step.
If minimum air temperature is greater than maximum air temperature, maximum air
temperature is set to minimum air temperature.




ASCE Standardized Reference Evapotranspiration Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Daily
"""""
.. math::

   ET_{sz} =\frac{0.408 \Delta (R_n-G) + \gamma \frac{C_n}{T_{mean} + 273}u_2
      (e_s-e_a)}{\Delta + \gamma(1+C_d u_2)}

where:

ET\ :sub:`sz` = standardized reference crop evapotranspiration for short
ET\ :sub:`os` or tall ET\ :sub:`rs` surfaces [mm d\ :sup:`-1` for daily time
steps or mm h\ :sup:`-1` for hourly time steps]

R\ :sub:`n` = calculated net radiation at the crop surface [MM m\ :sup:`-2`
d\ :sup:`-1` for daily time steps or MM m\ :sup:`-2` h\ :sup:`-1` for hourly
time steps]

G = soil heat flux density at the soil surface [MM m\ :sup:`-2` d\ :sup:`-1` for
daily time steps or MM m\ :sup:`-2` h\ :sup:`-1` for hourly time steps]

T\ :sub:`mean` = mean daily or hourly air temperature at 1.5 to 2.5-m height [°C]

u\ :sub:`2` = mean daily or hourly wind speed at 2-m height [m s\ :sup:`-1`]

e\ :sub:`s` = saturation vapor pressure at 1.5 to 2.5-m height [kPa]

e\ :sub:`a` = mean actual vapor pressure at 1.5 to 2.5-m height [kPa]

Δ = slope of the saturation vapor pressure-temperature curve [kPa °C\ :sup:`-1`]

γ = psychrometric constant [kPa °C\ :sup:`-1`]

C\ :sub:`n` = numerator constant that changes with reference type and calculation
time step [K mm s\ :sup:`3` Mg\ :sub:`-1` d\ :sup:`-1` for daily time steps or
K mm s\ :sup:`3` Mg\ :sub:`-1` h\ :sup:`-1` for hourly time steps]

C\ :sub:`d` = denominator constant that changes with reference type and
calculation time step [s m\ :sup:`-1`]

For a grass reference surface (ET\ :sub:`o`),

C\ :sub:`n` = 900

C\ :sub:`d` = 0.34

For an alfalfa reference surface (ET\ :sub:`r`),

C\ :sub:`n` = 1600

C\ :sub:`d` = 0.38

As soil heat flux density is positive when the soil is warming and negative when
the soil is cooling, over a day period it is relatively small compared to daily
R\ :sub:`n`. For daily calculations it is ignored,

G = 0

Hourly
""""""
The equation for ET\ :sub:`sz` is the same as daily, with

For a grass reference surface (ET\ :sub:`o`),

C\ :sub:`n` = 37.0

At night, when R\ :sub:`n`< 0,

C\ :sub:`d` = 0.96

G = 0.5

For an alfalfa reference surface (ET\ :sub:`r`),

C\ :sub:`n` = 66.0

At night, when R\ :sub:`n`< 0,

C\ :sub:`d` = 1.7

G = 0.2


# UNIT CONVERSION

# CALCULATE MEAN AIR TEMPERATURE

Mean Air Temperature (T\ :sub:`mean`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ASCE-EWRI (2005) advises to use the mean of daily minimum and daily maximum
\temperature to calculate mean daily temperature as opposed to the mean of
hourly temperatures.

.. math::

   T_{mean} = \frac{T_{min} + T_{max}}{2}

where:

T\ :sub:`mean` = mean daily air temperature [°C]

T\ :sub:`min` = minimum daily air temperature [°C]

T\ :sub:`max` = maximum daily air temperature [°C]

Ultimately, the ET\ :sub:`sz` equation requires actual vapor pressure
(e\ :sub:`a`). This can be calculated from dew point temperature (T\ :sub:`d`),
specific humidity (q), or relative humidity (RH). If needed, dew point
temperature can be calculated from minimum air temperature (T\ :sub:`min`) and
mean monthly dew point depression values (K\ :sub:`0`).

# CALCULATE DEW POINT TEMPERATURE FROM MINIMUM TEMPERATURE AND DEW POINT DEPRESSION

Dew Point Temperature
^^^^^^^^^^^^^^^^^^^^^

.. math::

   T_{d} = T_{min} - K_0

where:

T\ :sub:`d` = mean hourly or daily dew point temperature [°C]

T\ :sub:`min` = mean hourly or daily minimum daily air temperature [°C]

K\ :sub:`0` =  mean monthly dew point depression [°C]

# CALCULATE ACTUAL VAPOR PRESSURE FROM DEW POINT TEMPERATURE

Actual Vapor Pressure (e\ :sub:`a`) from Dew Point Temperature (T\ :sub:`d`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   e_a = 0.6108 \cdot \exp{\frac{17.27 \cdot T_{d}}{T_{d} + 237.3}}

where:

e\ :sub:`a` = actual vapor pressure [kPa]

T\ :sub:`d` = mean hourly or daily dew point temperature [°C]

# CALCULATE ACTUAL VAPOR PRESSURE FROM RELATIVE HUMIDITY

Actual Vapor Pressure (e\ :sub:`a`) from Relative Humidity (RH)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   e_a = \frac{RH}{100} \cdot e_{s}

where:

e\ :sub:`a` = actual vapor pressure [kPa]

RH = relative humidity [%]

e\ :sub:`s` = saturation vapor pressure [kPa]

# CALCULATE ACTUAL VAPOR PRESSURE FROM SPECIFIC HUMIDITY

Actual Vapor Pressure (e\ :sub:`a`) from Specific Humidity (q)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _actual_vapor_pressure(q, pair):
    """"Actual vapor pressure from specific humidity
    Parameters
    ----------
    q : scalar or array_like of shape(M, )
        Specific humidity [kg/kg].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].
    Returns
    -------
    ea : ndarray
        Actual vapor pressure [kPa].
    Notes
    -----
    ea = q * pair / (0.622 + 0.378 * q)
    """
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q
    return ea

Atmospheric Pressure (P)
^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _air_pressure(elev, method='asce'):
    """Mean atmospheric pressure at station elevation (Eqs. 3 & 34)
    Parameters
    ----------
    elev : scalar or array_like of shape(M, )
        Elevation [m].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
    Returns
    -------
    pair : ndarray
        Air pressure [kPa].
    Notes
    -----
    The current calculation in Ref-ET:
        101.3 * (((293 - 0.0065 * elev) / 293) ** (9.8 / (0.0065 * 286.9)))
    Equation 3 in ASCE-EWRI 2005:
        101.3 * (((293 - 0.0065 * elev) / 293) ** 5.26)
    Per Dr. Allen, the calculation with full precision:
        101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** (9.80665 / (0.0065 * 286.9)))
    """
    pair = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    if method == 'asce':
        pair += 293
        pair /= 293
        np.power(pair, 5.26, out=pair)
    elif method == 'refet':
        pair += 293
        pair /= 293
        np.power(pair, 9.8 / (0.0065 * 286.9), out=pair)
    # np.power(pair, 5.26, out=pair)
    pair *= 101.3
    return pair


Psychrometric Constant (γ)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \gamma = .0 000665 \cdot P

where:

γ = psychrometric constant [kPa °C\ :sup:`-1`]

P = mean atmospheric pressure at station elevation [kPa]

Slope of the Saturation Vapor Pressure-Temperature Curve (Δ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _es_slope(tmean, method='asce'):
    """Slope of the saturation vapor pressure-temperature curve (Eq. 5)
    Parameters
    ----------
    tmean : ndarray
        Mean air temperature [C].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
    Returns
    -------
    es_slope : ndarray
        Slope [kPa C-1].
    Notes
    -----
    4098 * 0.6108 * exp(17.27 * T / (T + 237.3)) / ((T + 237.3) ** 2))
    """
    if method == 'refet':
        es_slope = (
            4098.0 * _sat_vapor_pressure(tmean) / np.power(tmean + 237.3, 2))
    elif method == 'asce':
        es_slope = (
            2503.0 * np.exp(17.27 * tmean / (tmean + 237.3)) /
            np.power(tmean + 237.3, 2))
    return es_slope

Saturation Vapor Pressure (e\ :sub:`s`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _sat_vapor_pressure(temperature):
    """Saturation vapor pressure from temperature (Eq. 7)
    Parameters
    ----------
    temperature : scalar or array_like of shape(M, )
        Air temperature [C].
    Returns
    -------
    e : ndarray
        Saturation vapor pressure [kPa].
    Notes
    -----
    es = 0.6108 * exp(17.27 * temperature / (temperature + 237.3))
    """
    e = np.array(temperature, copy=True, ndmin=1).astype(np.float64)
    e += 237.3
    np.reciprocal(e, out=e)
    e *= temperature
    e *= 17.27
    np.exp(e, out=e)
    e *= 0.6108
    return e

# CALCULATE VAPOR PRESSURE DEFECIT FROM ES AND EA

Vapor Pressure Deficit (VPD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _vpd(es, ea):
    """Vapor pressure deficit
    Parameters
    ----------
    es : scalar or array_like of shape(M, )
        Saturated vapor pressure [kPa].
    ea : scalar or array_like of shape(M, )
        Actual vapor pressure [kPa].
    Returns
    -------
    ndarray
        Vapor pressure deficit [kPa].
    """

    return np.maximum(es - ea, 0)

Extraterrestrial Radiation (R\ :sub:`a`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The calculations for hourly and daily extraterrestrial radiation (R\ :sub:`a`)
differ slightly as the hourly calculations require hourly solar time angles (ω)
in addition to the sunset hour angle  (ω\ :sub:`s`) while the daily calculations
just require the sunset hour angle.


Hourly and daily calculations require solar declination (δ), sunset hour angle
(ω\ :sub:`s`), and inverse square of the earth-sun distance (d\ :sub:`r`).

**Solar Declination (δ)**

::

  def _delta(doy, method='asce'):
    """Earth declination (Eq. 51)
    Parameters
    ----------
    doy : scalar or array_like of shape(M, )
        Day of year.
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
    Returns
    -------
    ndarray
        Earth declination [radians].
    Notes
    -----
    Original equation in Duffie & Beckman (1980) (with conversions to radians):
        23.45 * (pi / 180) * sin(2 * pi * (doy + 284) / 365)
    Equation 24 in ASCE-EWRI (2005):
        0.409 * sin((2 * pi * doy / 365) - 1.39)
    """
    if method == 'asce':
        return 0.409 * np.sin(_doy_fraction(doy) - 1.39)
    elif method == 'refet':
        return 23.45 * (math.pi / 180) * np.sin(2 * math.pi * (doy + 284) / 365)

**Sunset Hour Angle (ω\ :sub:`s`)**

::

  def _omega_sunset(lat, delta):
    """Sunset hour angle (Eq. 59)
    Parameters
    ----------
    lat : scalar or array_like of shape(M, )
        Latitude [radians].
    delta : scalar or array_like of shape(M, )
        Earth declination [radians].
    Returns
    -------
    ndarray
        Sunset hour angle [radians].
    """
    return np.arccos(-np.tan(lat) * np.tan(delta))

**Inverse Square of the Earth-Sun Distance (d\ :sub:`r`)**

::

  def _dr(doy):
    """Inverse square of the Earth-Sun Distance (Eq. 50)
    Parameters
    ----------
    doy : scalar or array_like of shape(M, )
        Day of year.
    Returns
    -------
    ndarray
    Notes
    -----
    This function returns 1 / d^2, not d, for direct use in radiance to
      TOA reflectance calculation
    pi * L * d^2 / (ESUN * cos(theta)) -> pi * L / (ESUN * cos(theta) * d)
    """
    return 1.0 + 0.033 * np.cos(_doy_fraction(doy))

Daily Extraterrestrial Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _ra_daily(lat, doy, method='asce'):
    """Daily extraterrestrial radiation (Eq. 21)
    Parameters
    ----------
    lat : scalar or array_like of shape(M, )
        latitude [radians].
    doy : scalar or array_like of shape(M, )
        Day of year.
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
    Returns
    -------
    ra : ndarray
        Daily extraterrestrial radiation [MJ m-2 d-1].
    Notes
    -----
    Equation in ASCE-EWRI 2005 uses a solar constant of ~1366.666... W m-2
    Equation in Duffie & Beckman (?) uses a solar constant of 1367 W m-2
    """
    delta = _delta(doy, method)
    omegas = _omega_sunset(lat, delta)
    theta = (omegas * np.sin(lat) * np.sin(delta) +
             np.cos(lat) * np.cos(delta) * np.sin(omegas))

    if method == 'asce':
        ra = (24. / math.pi) * 4.92 * _dr(doy) * theta
    elif method == 'refet':
        ra = (24. / math.pi) * (1367 * 0.0036) * _dr(doy) * theta
    return ra

Hourly Extraterrestrial Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hourly calculations also require the calculation hourly solar time angles (ω),
which requires the calculation of solar time (t\ :sub:`s`).

**Seasonal Correction (sc)**

::

  def _seasonal_correction(doy):
    """Seasonal correction for solar time (Eqs. 57 & 58)
    Parameters
    ----------
    doy : scalar or array_like of shape(M, )
        Day of year.
    Returns
    ------
    ndarray
        Seasonal correction [hour]
    """
    b = 2 * math.pi * (doy - 81.) / 364.
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.0250 * np.sin(b)

**Solar Time (t\ :sub:`s`)**

::

  def _solar_time_rad(lon, time_mid, sc):
    """Solar time (i.e. noon is 0) (Eq. 55)
    Parameters
    ----------
    lon : scalar or array_like of shape(M, )
        Longitude [radians].
    time_mid : scalar or array_like of shape(M, )
        UTC time at midpoint of period [hours].
    sc : scalar or array_like of shape(M, )
        Seasonal correction [hours].
    Returns
    -------
    ndarray
        Solar time [hours].
    Notes
    -----
    This function could be integrated into the _omega() function since they are
    always called together (i.e. _omega(_solar_time_rad()).  It was built
    independently from _omega to eventually support having a separate
    solar_time functions for longitude in degrees.
    """
    return time_mid + (lon * 24 / (2 * math.pi)) + sc - 12

**Solar Time Angle (ω)**

::

  def _omega(solar_time):
    """Solar hour angle (Eq. 55)
    Parameters
    ----------
    solar_time : scalar or array_like of shape(M, )
        Solar time (i.e. noon is 0) [hours].
    Returns
    -------
    omega : ndarray
        Hour angle [radians].
    """
    omega = (2 * math.pi / 24.0) * solar_time

    # Need to adjust omega so that the values go from -pi to pi
    # Values outside this range are wrapped (i.e. -3*pi/2 -> pi/2)
    omega = _wrap(omega, -math.pi, math.pi)
    return omega

    def _wrap(x, x_min, x_max):
    """Wrap floating point values into range
    Parameters
    ----------
    x : ndarray
        Values to wrap.
    x_min : float
        Minimum value in output range.
    x_max : float
        Maximum value in output range.
    Returns
    -------
    ndarray
    """
    return np.mod((x - x_min), (x_max - x_min)) + x_min

**Hourly Extraterrestrial Radiation**

::

  def _rso_hourly(ra, ea, pair, doy, time_mid, lat, lon, method='asce'):
    """Full hourly clear sky solar radiation formulation (Appendix D)
    Parameters
    ----------
    ra : scalar or array_like of shape(M, )
        Extraterrestrial radiation [MJ m-2 h-1].
    ea : scalar or array_like of shape(M, )
        Actual vapor pressure [kPa].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].
    doy : scalar or array_like of shape(M, )
        Day of year.
    time_mid : scalar or array_like of shape(M, )
        UTC time at midpoint of period [hours].
    lat : scalar or array_like of shape(M, )
        Latitude [rad].
    lon : scalar or array_like of shape(M, )
        Longitude [rad].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
        Passed through to declination calculation (_delta()).
    Returns
    -------
    rso : ndarray
        Hourly clear sky solar radiation [MJ m-2 h-1].
    """
    sc = _seasonal_correction(doy)
    omega = _omega(_solar_time_rad(lon, time_mid, sc))

    # sin of the angle of the sun above the horizon (D.6 and Eq. 62)
    delta = _delta(doy, method)
    sin_beta = (
        np.sin(lat) * np.sin(delta) +
        np.cos(lat) * np.cos(delta) * np.cos(omega))

    # Precipitable water
    w = _precipitable_water(pair, ea)

    # Clearness index for direct beam radiation (Eq. D.2)
    # Limit sin_beta >= 0.01 so that KB does not go undefined
    kt = 1.0
    kb = 0.98 * np.exp(
        (-0.00146 * pair) / (kt * np.maximum(sin_beta, 0.01)) -
        0.075 * np.power((w / np.maximum(sin_beta, 0.01)), 0.4))

    # Transmissivity index for diffuse radiation (Eq. D.4)
    kd = np.minimum(-0.36 * kb + 0.35, 0.82 * kb + 0.18)

    rso = ra * (kb + kd)
    return rso

Clear-Sky Radiation (R\ :sub:`so`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Daily Clear-Sky Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^

::

  def _ra_daily(lat, doy, method='asce'):
      """Daily extraterrestrial radiation (Eq. 21)
      Parameters
      ----------
      lat : scalar or array_like of shape(M, )
          latitude [radians].
      doy : scalar or array_like of shape(M, )
          Day of year.
      method : {'asce' (default), 'refet'}, optional
          Calculation method:
          * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
          * 'refet' -- Calculations will follow RefET software.
      Returns
      -------
      ra : ndarray
          Daily extraterrestrial radiation [MJ m-2 d-1].
      Notes
      -----
      Equation in ASCE-EWRI 2005 uses a solar constant of ~1366.666... W m-2
      Equation in Duffie & Beckman (?) uses a solar constant of 1367 W m-2
      """
      delta = _delta(doy, method)
      omegas = _omega_sunset(lat, delta)
      theta = (omegas * np.sin(lat) * np.sin(delta) +
               np.cos(lat) * np.cos(delta) * np.sin(omegas))

      if method == 'asce':
          ra = (24. / math.pi) * 4.92 * _dr(doy) * theta
      elif method == 'refet':
          ra = (24. / math.pi) * (1367 * 0.0036) * _dr(doy) * theta
      return ra

Hourly Clear-Sky Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Several calculations, including the sin of the angle of the sun above the
horizon (sin\ :sub:`β`) and the clearness index for direct beam radiation
(k\ :sub:`b`) change when calculating hourly clear-sky radiation.

::

  def _rso_hourly(ra, ea, pair, doy, time_mid, lat, lon, method='asce'):
    """Full hourly clear sky solar radiation formulation (Appendix D)
    Parameters
    ----------
    ra : scalar or array_like of shape(M, )
        Extraterrestrial radiation [MJ m-2 h-1].
    ea : scalar or array_like of shape(M, )
        Actual vapor pressure [kPa].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].
    doy : scalar or array_like of shape(M, )
        Day of year.
    time_mid : scalar or array_like of shape(M, )
        UTC time at midpoint of period [hours].
    lat : scalar or array_like of shape(M, )
        Latitude [rad].
    lon : scalar or array_like of shape(M, )
        Longitude [rad].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.
        Passed through to declination calculation (_delta()).
    Returns
    -------
    rso : ndarray
        Hourly clear sky solar radiation [MJ m-2 h-1].
    """
    sc = _seasonal_correction(doy)
    omega = _omega(_solar_time_rad(lon, time_mid, sc))

    # sin of the angle of the sun above the horizon (D.6 and Eq. 62)
    delta = _delta(doy, method)
    sin_beta = (
        np.sin(lat) * np.sin(delta) +
        np.cos(lat) * np.cos(delta) * np.cos(omega))

    # Precipitable water
    w = _precipitable_water(pair, ea)

    # Clearness index for direct beam radiation (Eq. D.2)
    # Limit sin_beta >= 0.01 so that KB does not go undefined
    kt = 1.0
    kb = 0.98 * np.exp(
        (-0.00146 * pair) / (kt * np.maximum(sin_beta, 0.01)) -
        0.075 * np.power((w / np.maximum(sin_beta, 0.01)), 0.4))

    # Transmissivity index for diffuse radiation (Eq. D.4)
    kd = np.minimum(-0.36 * kb + 0.35, 0.82 * kb + 0.18)

    rso = ra * (kb + kd)
    return rso

Cloudiness Fraction (fcd)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \beta = \arcsin(\sin(\textrm{lat}) \cdot \sin(\delta) + \cos(\textrm{lat})
   \cdot \cos(\delta) \cdot \cos(\omega))

where:

beta

lat = Latitude [radians]

δ = solar declination [radians]

ω = solar hour angle [radians]



# CALCULATE NET LONG-WAVE RADIATION

# CALCULATE NET RADIATION

# ADJST WINDSPEED FOR MEASUREMENT HEIGHT

Windspeed Adjustment
^^^^^^^^^^^^^^^^^^^^

The standardized reference crop evapotranspiration equation assumes a 2-m height
windspeed. Windspeed measured at different heights can be approximated as

.. math::

   u_2 = u_z + \frac{4.87}{\ln\left(67.8 z_w - 5.42 \right)}

where:

u\ :sub:`2` = wind speed at 2 m above ground surface [m s\ :sup:`-1`]

u\ :sub:`z` = measured wind speed at z\ :sub:`w` m above ground surface [m s\ :sup:`-1`]

z\ :sub:`w` = height of wind measurement about ground surface [m]

# CACLCULATE REFERENCE ET



Latent Heat of Vaporization (λ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The latent heat of vaporization is calculated from mean air temperature. This
differs from ASCE-EWRI (2005) which advises to use a constant value of
2.45 MJ kg\ :sup:`-1` as it varies only slightly over the ranges of air
temperature that occur in agricultural or hydrologic systems. The equation used
is from XXX.

.. math::

   \lambda = 2500000 - 2360 \cdot T_{mean}

where:

λ = latent heat of vaporization [MJ kg\ :sup:`-1`]

T\ :sub:`mean` = mean daily air temperature [°C]







.. _model-description-refet-tr:

Thornton and Running Solar Radiation Estimate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The calculation of ET\ :sub:`sz` requires net radiation at the crop surface.
When this is not available, net radiation can be estimated using the approach
presented by Thornton and Running
`(Thornton and Running, 1999) <https://doi.org/10.1016/S0168-1923(98)00126-9>`_.

The calculation of clear sky radiation

.. math::

   inc_{Rs} =cs_{Rso} \cdot (1 - 0.9)


incRs = csRSo * (1 - 0.9 * math.exp(-BTR * dt ** 1.5))


This equation requires the parameter B, which

.. math::

   B = b_0 + b_1 \cdot \exp(-b_2 \cdot \overline{\Delta T})

For arid stations, [REFERENCE FOR THESE COEFFICIENTS]

b\ :sub:`0` = 0.023

b\ :sub:`1` = 0.1

b\ :sub:`2` = 0.2

b\ :sub:`0`, b\ :sub:`1`, and b\ :sub:`2` are provided by the user. [DISCUSSION OF THESE PARAMETERS, AND HOW TO GET THEM]




Other Potential ET Estimates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The RefET module code can also calculate potential evapotranspiration using
several different approaches. This provides a comparison with reference ET.


Penman
""""""

.. math::

   ET_o = W \cdot R_n + (1-W) \cdot f(ur) \cdot (e_a - e_d)

where:

ET\ :sub:`o` = grass reference evapotranspiration [mm d\ :sup:`-1`]

W = weighting factor (depends on temperature and altitude)

R\ :sub:`n` = net radiation in equivalent evaporation [mm d\ :sup:`-1`]

f(ur) = wind-related function

(e\ :sub:`a` - e\ :sub:`d`) = difference between saturation vapor pressure at mean
air temperature and the mean actual vapor pressure of the air [hPa]

.. math::

   f(ur) = 0.27 (1+(ur_2 / 100))

where:

f(ur) = wind-related function

ur\ :sub:`2` = daily  wind run at 2-m height [km d\ :sup:`-1`]

`(Penman, 1948) <https://doi.org/10.1098/rspa.1948.0037>`_.

Kimberly Penman 1982
"""""""""""""""""""""

Hargreaves-Samani
"""""""""""""""""

`(Hargreaves and Samani, 1985) <https://doi.org/10.13031/2013.26773>`_.

Priestley-Taylor
""""""""""""""""

`(Priestley and Taylor, 1972) <https://doi.org/10.1175/1520-0493(1972)100//<0081:OTAOSH//>2.3.CO;2>`_ .

Blaney-Criddle
""""""""""""""
[THIS CURRENTLY ISN'T SUPPORTED]

`(Blaney and Criddle, 1950) <https://archive.org/details/determiningwater96blan>`_.

.. _model-description-cropet:

.. _model-description-cropet:

CropETPrep
----------

CropET
------

The CropET module of the ET Demands model is the FAO-56 dual crop coefficient model
`(Allen et al., 1998) <http://www.fao.org/docrep/X0490E/X0490E00.htm>`_ .

.. math::

   ET_{c} = (K_c K_{cb} + K_e)ET_o

ET\ :sub:`c` = crop evapotranspiration

K\ :sub:`c` = crop coefficient

K\ :sub:`cb` = Basal crop coefficient

K\ :sub:`e` = coefficient representing bare soil evaporation

ET\ :sub:`o` = reference crop evapotranspiration from a grass reference surface

.. _model-description-cropet-aridfctr:

Aridity Rating
^^^^^^^^^^^^^^
Allen and Brockway `(1983) <https://idwr.idaho.gov/files/publications/1983-MISC-Est-Consumptive-Use-08-1983.pdf>`_
estimated consumptive irrigation requirements for crops in Idaho, and developed an
aridity rating for each meteorological weather station used to adjust temperature data.
The aridity rating ranges from 0 (fully irrigated) to 100 (arid) and reflects conditions
affecting the aridity of the site. The aridity rating was based on station metadata
information, questionnaires, and phone conversations, and includes conditions
close to the station (within a 50m radius),the area around the station
(within a 1600m radius in the upwind direction), and the region around the station
(within a 48km radius in the upwind direction).

.. math::

   AR_{cum} = 0.4AR_{St} + 0.5AR_{Ar} + 0.1AR_{Reg}

Allen and Brockway (1983) used empirical data from Allen and Brockway `(1982) <http://digital.lib.uidaho.edu/cdm/ref/collection/idahowater/id/379>`_
to develop monthly aridity effect values (A\ :sub:`e`). These values were used
as adjustment factors for the temperature data based on the aridity rating.
Stations with an aridity rating of 100 applied the adjustment factor directly,
while stations with aridity ratings less than 100, weighted the adjustment factor
by the aridity rating.

.. math::

   T_{adj} = \frac{AR_{cum}}{100} \cdot A_{e}

The empirical temperature data and aridity effect values used are show in the table below.
These data are the average monthly departure of air temperatures over arid areas
from air temperatures over irrigated areas in southern Idaho during 1981, and the
aridity effect.


+-------------+---------------+---------------+---------------+-------------+
| Month       | T\ :sub:`max` | T\ :sub:`min` | T\ :sub:`mean`| A\ :sub:`e` |
+=============+===============+===============+===============+=============+
| April       | 2.7           | 2.4           | 2.5           | 1.0         |
+-------------+---------------+---------------+---------------+-------------+
| May         | 1.3           | 0.6           | 0.9           | 1.5         |
+-------------+---------------+---------------+---------------+-------------+
| June        | 2.4           | 1.8           | 2.1           | 2.0         |
+-------------+---------------+---------------+---------------+-------------+
| July        | 4.8           | 2.9           | 3.8           | 3.5         |
+-------------+---------------+---------------+---------------+-------------+
| August      | 5.2           | 4.3           | 4.7           | 4.5         |
+-------------+---------------+---------------+---------------+-------------+
| September   | 3.3           | 2.7           | 3.0           | 3.0         |
+-------------+---------------+---------------+---------------+-------------+
| October     | 0.3           | 1.6           | 0.9           | 0.0         |
+-------------+---------------+---------------+---------------+-------------+

HOW WAS THE ARIDITY EFFECT DETERMINED. ARE THESE DATA GENERAL ENOUGH TO USE
AT OTHER LOCATIONS IF AN ARIDITY RATING IS DEVELOPED? IF NOT, CAN WE GENERALIZE
THE APPROACH TO DEVELOPING AN ARIDITY RATING, AND ASSOCIATED ARIDITY EFFECT ADJUSTMENTS?
ALSO, THE 'CropET' MODULE HAS A WAY OF PULLING IN ARIDITY EFFECT VALUES, HOWEVER,
THE 'RefET' MODULE DOES NOT. THIS MEANS THAT WHILE TEMPERATURES USED IN THE
CropET MODULE ARE ADJUSTED, TEMPERATURES USED TO CALCUATE REFERENCE ET ARE NOT.
IF WE WANT TO CONTINUE TO SUPPORT THE ARIDITY RATING, THIS SHOULD BE ADDRESSED.
WOULD ALSO REQUIRE PASSING THE MODEL THE ARIDITY EFFECT ADJUSTMENT FACTORS.

AreaET
------

PostProcessing
--------------

References
-----------
Allen, R. G., & Brockway, C. E. (1982). Weather and Consumptive Use in the Bear
River Basin, Idaho During 1982.

Allen, R. G., & Brockway, C. E. (1983). Estimating Consumptive Irrigation
Requirements for Crops in Idaho.

Allen, R. G., Pereira, L. S., Smith, M., Raes, D., & Wright, J. L. (2005).
FAO-56 Dual Crop Coefficient Method for Estimating Evaporation from Soil and
Application Extensions. Journal of Irrigation and Drainage Engineering, 131(1),
2–13. https://doi.org/10.1061/(ASCE)0733-9437(2005)131:1(2)

Allen, R. G., & Robison, C. W. (2007). Evapotranspiration and Consumptive
Irrigation Water Requirements for Idaho.

ASCE-EWRI. (2005). The ASCE Standardized Reference Evapotranspiration Equation.

Blaney, H. F., & Criddle, W. D. (1950). Determining Water Requirements in
Irrigated Areas from Climatological and Irrigation Data. SCS-TP-96. Washington D.C.

Hargreaves, G. H., & A. Samani, Z. (1985). Reference Crop Evapotranspiration
from Temperature. Applied Engineering in Agriculture, 1(2), 96–99.
https://doi.org/https://doi.org/10.13031/2013.26773

Penman, H. L. (1948). Natural Evaporation from Open Water, Bare Soil and Grass.
Proceedings of the Royal Society A: Mathematical, Physical and Engineering
Sciences, 193(1032), 120–145. https://doi.org/10.1098/rspa.1948.0037

Priestley, C. H. B., & Taylor, R. J. (1972). On the Assessment of Surface Heat
Flux and Evaporation Using Large-Scale Parameters. Monthly Weather Review,
100(2), 81–92. https://doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2

Thornton, P. E., & Running, S. W. (1999). An improved algorithm for estimating
incident daily solar radiation from measurements of temperature, humidity, and
precipitation. Agricultural and Forest Meteorology, 93, 211–228.
https://doi.org/10.1016/S0168-1923(98)00126-9
