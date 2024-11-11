.. _model-description:

=================
Model Description
=================

The ETDemands model consists of three modules that develop daily estimates of crop irrigation water requirements using daily weather data (including reference evapotranspiration) and crop-specific crop growth curves. The three modules that make up the ETDemands model are RefET, CropET, and AreaET. Currently, only the CropET module, prep tools, and the post-processing tools are fully functional for ETDemands.

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

Daily Reference Evapotranspiration
""""""""""""""""""""""""""""""""""

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

Hourly Reference Evapotranspiration
"""""""""""""""""""""""""""""""""""

The equation for ET\ :sub:`sz` is the same as daily, with

For a grass reference surface (ET\ :sub:`o`),

C\ :sub:`n` = 37.0

At night, when R\ :sub:`n` < 0,

C\ :sub:`d` = 0.96

G = 0.5

For an alfalfa reference surface (ET\ :sub:`r`),

C\ :sub:`n` = 66.0

At night, when R\ :sub:`n` < 0,

C\ :sub:`d` = 1.7

G = 0.2


# UNIT CONVERSION

Mean Air Temperature (T\ :sub:`mean`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ASCE-EWRI (2005) advises to use the mean of daily minimum and daily maximum
\temperature to calculate mean daily temperature as opposed to the mean of
hourly temperatures.

.. math::

   T_{mean} = \frac{T_{max} + T_{min}}{2}

where:

T\ :sub:`mean` = mean daily air temperature [°C]

T\ :sub:`max` = maximum daily air temperature [°C]

T\ :sub:`min` = minimum daily air temperature [°C]

Ultimately, the ET\ :sub:`sz` equation requires actual vapor pressure
(e\ :sub:`a`). This can be calculated from dew point temperature (T\ :sub:`d`),
specific humidity (q), or relative humidity (RH). If needed, dew point
temperature can be calculated from minimum air temperature (T\ :sub:`min`) and
mean monthly dew point depression values (K\ :sub:`0`).

Dew Point Temperature
^^^^^^^^^^^^^^^^^^^^^

.. math::

   T_{d} = T_{min} - K_0

where:

T\ :sub:`d` = mean hourly or daily dew point temperature [°C]

T\ :sub:`min` = mean hourly or daily minimum daily air temperature [°C]

K\ :sub:`0` =  mean monthly dew point depression [°C]

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

Actual Vapor Pressure (e\ :sub:`a`) from Specific Humidity (q)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   e_a = \frac{q \cdot P}{0.622 + 0.378 \cdot q}

where:

e\ :sub:`a` = actual vapor pressure [kPa]

q = specific humidity [kg/kg]

P = mean atmospheric pressure at station elevation [kPa]

Atmospheric Pressure (P)
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   P = 101.3 \cdot \left(\frac{293.15 - 0.0065z}{ 293.15} \right)^{(9.80665 / (0.0065 \cdot 286.9)}

where:

P = mean atmospheric pressure at station elevation [kPa]

z = station elevation above mean sea level [m]

This equation differs slightly from ASCE 2005 as it reflects full precision per Dr. Allen (pers. comm.).

Psychrometric Constant (γ)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \gamma = .0 000665 \cdot P

where:

γ = psychrometric constant [kPa °C\ :sup:`-1`]

P = mean atmospheric pressure at station elevation [kPa]

Slope of the Saturation Vapor Pressure-Temperature Curve (Δ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   \Delta = 4098 \cdot \frac{0.6108 \cdot \exp \left( \frac{17.27T_{mean}}
   {T_{mean} + 237.3} \right)}{\left(T_{mean} + 237.3\right)^2}

where:

Δ = slope of the saturation vapor pressure-temperature curve (kPa °C\ :sup:`-1`]

T\ :sub:`mean` = mean daily air temperature [°C]

Saturation Vapor Pressure (e\ :sub:`s`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   e_s = 0.6108 \cdot \exp \left( \frac{17.27T_{mean}}{T_{mean} + 237.3} \right)

where:

e\ :sub:`s` = saturation vapor pressure

Tetens (1930)

Vapor Pressure Deficit (VPD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
   \textrm{VPD} = e_s - e_a

where:

VPD = vapor pressure deficit [kPa]

e\ :sub:`s` = saturation vapor pressure [kPa]

e\ :sub:`a` = actual vapor pressure [kPa]

Extraterrestrial Radiation (R\ :sub:`a`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calculations for hourly and daily extraterrestrial radiation (R\ :sub:`a`)
differ slightly as the hourly calculations require hourly solar time angles (ω)
in addition to the sunset hour angle  (ω\ :sub:`s`) while the daily calculations
just require the sunset hour angle.


Hourly and daily calculations require solar declination (δ), sunset hour angle
(ω\ :sub:`s`), and inverse square of the earth-sun distance (d\ :sub:`r`).

**Solar Declination (δ)**

.. math::

  \delta=23.45 \cdot \frac{\pi}{180} \cdot \sin\left(\frac{2\pi}{365}
  \cdot(\textrm{DOY} + 284)\right)

where:

δ = solar declination [radians]

DOY = day of year

**Sunset Hour Angle** (ω\ :sub:`s`)

.. math::

  \omega_{s} = \arccos(-\tan(\textrm{lat}) \cdot \tan(\delta))

where:

ω\ :sub:`s` = sunset hour angle [radians]

lat = Latitude [radians]

δ = solar declination [radians]

To calcuate the inverse quare of the earth-sun distance, the day-of-year
fraction (DOY\ :sub:`frac`) is needed

Day-of-Year Fraction (DOY\ :sub:`frac`)

.. math::

  \textrm{DOY}_{\textrm{frac}} = \textrm{DOY} \cdot \left(\frac{2\pi}{365}\right)

where:

DOY\ :sub:`frac` = day-of-year fraction

DOY = day-of-year

**Inverse Square of the Earth-Sun Distance** (d\ :sub:`r`)

.. math::

  d_{r} = 1 + 0.033 \cos(\textrm{DOY}_{\textrm{frac}})

where:

d\ :sub:`r` = inverse square of the earth-sun distance [d\ :sup:`-2`]

ω\ :sub:`s` = sunset hour angle [radians]

lat = Latitude [radians]

δ = solar declination [radians]

Daily Extraterrestrial Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

  \theta = \omega_{s} \cdot \sin(\textrm{lat}) \cdot \sin(\delta)
  + \cos(\textrm{lat})\cdot \cos(\delta) \cdot \sin(\omega_{s})

   R_{a} = \frac{24}{\pi} \cdot (1367 \cdot 0.0036) \cdot d_{r} \cdot \theta

where:

ω\ :sub:`s` = sunset hour angle [radians]

lat = Latitude [radians]

R\ :sub:`a` = daily extraterrestrial radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

δ = solar declination [radians]

d\ :sub:`r` = inverse square of the earth-sun distance [d\ :sup:`-2`]

Hourly Extraterrestrial Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hourly calculations also require the calculation hourly solar time angles (ω),
which requires the calculation of solar time (t\ :sub:`s`).

**Seasonal Correction (sc)**

.. math::

   b = \frac{2\pi}{364} \cdot (\textrm{DOY} - 81)

   sc = 0.1645 \cdot \sin(2b) - 0.1255 \cdot \cos(b) - 0.0250 \sin(b)

where:

sc = seasonal correction [hours]

DOY = day-of-year

**Solar Time (t\ :sub:`s`)**

.. math::

   t_{s} = t + (\textrm{lon} \cdot \frac{24}{2\pi} + sc - 12)

where:

t\ :sub:`s` = solar time (i.e. noon is 0) [hours]

lon = Longitude [radians]

t = UTC time at the midpoint of the period [hours]

sc = seasonal correction [hours]

**Solar Time Angle (ω)**

.. math::

   \omega = \frac{2\pi}{24} \cdot t_{s}

where:

ω = solar hour angle [radians]

t\ :sub:`s` = solar time (i.e. noon is 0) [hours]

**Hourly Extraterrestrial Radiation**

.. math::

   \omega_{1} = \omega - \frac{\pi}{24}\cdot t

   \omega_{2} = \omega + \frac{\pi}{24}\cdot t

Checks on ω\ :sub:`1` and ω\ :sub:`2`

.. math::
   \textrm{if } \omega_{1} < -\omega_{s} \textrm{ then } \omega_{1} = -\omega_{s}

   \textrm{if } \omega_{2} < -\omega_{s} \textrm{ then } \omega_{2} = -\omega_{s}

   \textrm{if } \omega_{1} > \omega_{s} \textrm{ then } \omega_{1} = \omega_{s}

   \textrm{if } \omega_{2} > \omega_{s} \textrm{ then } \omega_{2} = \omega_{s}

   \textrm{if } \omega_{1} > \omega_{2} \textrm{ then } \omega_{1} = \omega_{2}

.. math::

   \theta = (\omega_{2} - \omega_{1}) \cdot \sin(\textrm{lat}) \cdot \sin(\delta)
   + \cos(\textrm{lat}) \cdot \cos(\delta) \cdot \sin(\omega_{2} - \omega_{1})

.. math::

   R_{a} = \frac{24}{\pi} \cdot (1367 \cdot 0.0036) \cdot d_{r} \cdot \theta

where:
ω\ :sub:`1` = solar time angle at the beginning of the period [radians]

ω\ :sub:`2` = solar time angle at the end of the period [radians]

ω = solar hour angle [radians]

t = UTC time at the midpoint of the period [hours]

ω\ :sub:`s` = sunset hour angle [radians]

lat = Latitude [radians]

δ = solar declination [radians]

R\ :sub:`a` = hourly extraterrestrial radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

d\ :sub:`r` = inverse square of the earth-sun distance [d\ :sup:`-2`]

Clear-Sky Radiation (R\ :sub:`so`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Sin of the Angle of the Sun above the Horizon (sin\ :sub:`β24`)**

.. math::

   \sin_{\beta24} = \sin(0.85 + 0.3 \cdot \textrm{lat} \cdot
   \sin(\textrm{DOY}_{\textrm{frac}})
    - 1.39)) - 0.42 \cdot \textrm{lat}^2

    \sin_{\beta24} = \max(\sin_{\beta24}, 0.1)

where:

sin\ :sub:`β24`= sine of the angle of the sun above the horizon [radians]

lat = Latitude [radians]

DOY\ :sub:`frac` = day-of-year fraction

**Precipitable Water (w)**

.. math::
   w = P \cdot 0.14 \cdot e_{a} + 2.1

where:

w = precipitable water [mm]

P = mean atmospheric pressure at station elevation [kPa]

e\ :sub:`a` = actual vapor pressure [kPa]


**Clearness Index for Direct Beam Radiation (k\ :sub:`b`)**

.. math::

   k_{b} = 0.98 \cdot \exp{\left(\frac{-0.00146P}{sin_{\beta24} - 0.0075}\right)}
   - 0.075\left(\frac{w}{\sin_{\beta24}}\right)^{0.4}

where:

k\ :sub:`b` = clearness index for direct beam radiation

P = mean atmospheric pressure at station elevation [kPa]

sin\ :sub:`β24`= sine of the angle of the sun above the horizon [radians]

w = precipitable water [mm]

**Transmissivity Index for Diffuse Radiation (k\ :sub:`d`)**

.. math::

   k_{d} = \min
   \begin{cases}
   -0.36 \cdot k_{b} + 0.35 \\
   0.82 \cdot k_{b} + 0.18
   \end{cases}

where:

k\ :sub:`d` = transmissivity index for diffuse radiation

k\ :sub:`b` = clearness index for direct beam radiation

Daily Clear-Sky Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   R_{so} = R_{a} \cdot (k_{b} + k_{d})

where:

R\ :sub:`so` = daily clear-sky radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`a` = daily extraterrestrial radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

k\ :sub:`b` = clearness index for direct beam radiation

k\ :sub:`d` = transmissivity index for diffuse radiation


Hourly Clear-Sky Radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Several calculations, including the sin of the angle of the sun above the
horizon (sin\ :sub:`β`) and the clearness index for direct beam radiation
(k\ :sub:`b`) change when calculating hourly clear-sky radiation.


**Sin of the Angle of the Sun above the Horizon (sin\ :sub:`β`)**

.. math::

   \sin_{\beta} = \sin(\textrm{lat}) \cdot \sin(\delta)+\cos(\textrm{lat}) \cdot
   \cos(\delta) \cdot \cos(\omega)

   \sin_{\beta,c} = \max
   \begin{cases}
   \sin_{\beta} \\
   0.01
   \end{cases}

where:

sin\ :sub:`β`= sine of the angle of the sun above the horizon [radians]

sin\ :sub:`β,c`= sin\ :sub:`β` limited to 0.01 so that k\ :sub:`b` does not go
undefined

lat = Latitude [radians]

δ = solar declination [radians]

ω = solar hour angle [radians]

**Clearness Index for Direct Beam Radiation (k\ :sub:`b`)**

.. math::

   k_{t} = 1.0

   k_{b} = 0.98 \cdot \exp \left(\frac{-0.00146P}{k_{t} \cdot \sin_{\beta,c}}\right) -
   0.075  \left(\frac{w}{\sin_{\beta,c}}\right)^{0.4}

where:

k\ :sub:`t` = atmospheric turbidity coefficient

k\ :sub:`b` = clearness index for direct beam radiation

P = mean atmospheric pressure at station elevation [kPa]

sin\ :sub:`β,c`= sine of the angle of the sun above the horizon, limited to 0.01 [radians]

w = precipitable water [mm]

**Transmissivity Index for Diffuse Radiation (k\ :sub:`d`)**

.. math::

   k_{d} = \min
   \begin{cases}
   -0.36 \cdot k_{b} + 0.35 \\
   0.82 \cdot k_{b} + 0.18
   \end{cases}

where:

k\ :sub:`d` = transmissivity index for diffuse radiation

k\ :sub:`b` = clearness index for direct beam radiation

**Hourly Clear-Sky Radiation**

.. math::

   R_{so} = R_{a} \cdot (k_{b} + k_{d})

where:

R\ :sub:`so` = hourly clear-sky radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

R\ :sub:`a` = hourly extraterrestrial radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

k\ :sub:`b` = clearness index for direct beam radiation

k\ :sub:`d` = transmissivity index for diffuse radiation

Cloudiness Fraction (fcd)
^^^^^^^^^^^^^^^^^^^^^^^^^

**Daily Cloudiness Fraction**

.. math::

   \textrm{fcd} = 1.35 \cdot \frac{R_{s}}{R_{so}}-0.35

   0.3 < \frac{R_{s}}{R_{so}} \leq 1.0

where:

fcd = daily cloudiness fraction

R\ :sub:`s` = measured solar radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`so` = clear sky solar radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`s` / R\ :sub:`so` is limited to 0.3 < R\ :sub:`s` / R\ :sub:`so` ≤ 1.0


**Hourly Cloudiness Fraction**

At low sun angles (β), cloudiness fraction (fcd) is set to 1.

.. math::

   \beta = \arcsin(\sin(\textrm{lat}) \cdot \sin(\delta) + \cos(\textrm{lat})
   \cdot \cos(\delta) \cdot \cos(\omega))

   \textrm{fcd}[R_{so} > 0] = 1.35 \cdot \frac{R_{s}}{R_{so}}-0.35

   0.3 < \frac{R_{s}}{R_{so}} \leq 1.0

   \textrm{fcd}[\beta < 0.3] = 1

where:

β = angle of the sun above the horizon [radians]

lat = Latitude [radians]

δ = solar declination [radians]

ω = solar hour angle [radians]

fcd = hourly cloudiness fraction

R\ :sub:`s` = measured solar radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

R\ :sub:`so` = clear sky solar radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

Net Longwave Radiation (R\ :sub:`nl`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Daily Net Longwave Radiation**

.. math::

   R_{nl} = 4.901\textrm{e-9} \cdot \textrm{fcd} \cdot (0.34 - 0.14 \cdot \sqrt{e_{a}}
   \cdot 0.5 ((T_{max} + 273.16)^4 + (T_{min} + 273.16)^4)

where:

R\ :sub:`nl` = daily net longwave radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

fcd = daily cloudiness fraction

e\ :sub:`a` = actual vapor pressure [kPa]

T\ :sub:`max` = maximum daily air temperature [°C]

T\ :sub:`min` = minimum daily air temperature [°C]

**Hourly Net Longwave Radiation**

.. math::

   R_{nl} = 2.042\textrm{e-10} \cdot \textrm{fcd} \cdot (0.34 - 0.14 \cdot \sqrt{e_{a}}
   \cdot(T_{mean} + 273.16)^4

where:

R\ :sub:`nl` = hourly net longwave radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

fcd = daily cloudiness fraction

e\ :sub:`a` = actual vapor pressure [kPa]

T\ :sub:`mean` = mean hourly air temperature [°C]

Net Radiation (R\ :sub:`n`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Daily Net Radiation**

.. math::

   R_{n} = 0.77 \cdot R_{s} - R_{nl}

where:

R\ :sub:`n` = daily net radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`nl` = daily net longwave radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`s` = measured solar radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

**Hourly Net Radiation**

.. math::

   R_{n} = 0.77 \cdot R_{s} - R_{nl}

where:

R\ :sub:`n` = hourly net radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

R\ :sub:`nl` = hourly net longwave radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

R\ :sub:`s` = measured solar radiation [MJ m\ :sup:`-2` h\ :sup:`-1`]

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

## CACLULATE MIN AND MAX MONTHLY MEAN TEMPERATURES

.. _model-description-refet-tr:

Thornton and Running Solar Radiation Estimate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If measured solar radiation (R\ :sub:`s`) is not provided, it can be estimated
using the approach described in Thorton and Running
`(1999) <https://doi.org/10.1016/S0168-1923(98)00126-9>`_. This approach
requires three calibrated coefficients [LINK TO PAGE ON HOW TO DO THIS].

.. math::

   T_{diff} = T_{max} - T_{min}

   T_{mon,diff} = T_{mon,max} - T_{mon,min}

   B_{TR} = TR_{b0} + TR_{b1} \cdot \exp{(TR_{b2} \cdot{T_{mon,diff}})

   R_{s} = R_{so} \cdot (1 - 0.9 \exp{(-B_{TR} \cdot T_{diff}^{1.5})})

where:

T\ :sub:`diff` = temperature difference [°C]

T\ :sub:`max` = maximum daily air temperature [°C]

T\ :sub:`min` = minimum daily air temperature [°C]

T\ :sub:`mon,diff` = mean monthly temperature difference [°C]

T\ :sub:`mon,max` = mean monthly maximum air temperature [°C]

T\ :sub:`mon,min` = mean monthly minimum air temperature [°C]

TR\ :sub:`b0` = Thornton and Running b\ :sub:`0` coefficient

TR\ :sub:`b1` = Thornton and Running b\ :sub:`1` coefficient

TR\ :sub:`b2` = Thornton and Running b\ :sub:`2` coefficient

B\ :sub:`TR` = Thorton and Running parameter

R\ :sub:`s` = calculated solar radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]

R\ :sub:`so` = clear sky solar radiation [MJ m\ :sup:`-2` d\ :sup:`-1`]


For arid stations, [REFERENCE FOR THESE COEFFICIENTS]

TR\ :sub:`b0` = 0.023

TR\ :sub:`b1` = 0.1

TR\ :sub:`b2` = 0.2

[DISCUSSION OF THESE PARAMETERS, AND HOW TO GET THEM]


Other Potential ET Estimates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RefET module code can also calculate potential evapotranspiration using
several different approaches. This provides a comparison with reference ET.

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

`(Priestley and Taylor, 1972) <https://journals.ametsoc.org/mwr/article/100/2/81/60530>`_ .

Blaney-Criddle
""""""""""""""
[CURRENTLY NOT SUPPORTED]

`(Blaney and Criddle, 1950) <https://archive.org/details/determiningwater96blan>`_.

.. _model-description-cropet:

.. _model-description-cropet:

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


AreaET
------
The AreaET module is currently under development.

The AreaET module converts crop evapotranspiration and net irrigation water requirement rates estimates output by the 
CropET module to volume and flow estaimtes based on user supplied acreage information. Aggregations of daily, monthly, annual, and growing seaon volumes and flows are output along with crop percentages and ratios for each study cell. Users can define start and end dates to analyze specific time periods.

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
Irrigated Areas from Climatological and Irrigation Data. SCS-TP-96.
Washington D.C.

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
