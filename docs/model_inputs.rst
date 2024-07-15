.. _model-inputs:

Model Inputs
============

.. _model-inputs-refet:

RefET
-----
The RefET module calculates hourly or daily reference ET from meteorological data using the approach described in
ASCE-NWRI `(2005) <https://doi.org/10.1061/9780784408056>`_. If needed, the module
will estimate solar radiation and dew point temperature, and gap fill meteorological data.

Meteorology Metadata
^^^^^^^^^^^^^^^^^^^^

Meteorology Data
^^^^^^^^^^^^^^^^
The RefET module requires hourly or daily meteorological data. If the calculated reference ET will be used
in the CropET module, care should be taken to ensure this meteorological data is representative of agricultural conditions.
Meteorological data can be obtained from :ref:`agricultural weather networks <data-sources-ag-met>`, or :ref:`adjustments <model-description-cropet-aridfctr>`
reference ET can be made within the CropET module.

Timeseries Data
"""""""""""""""

Daily Variables
~~~~~~~~~~~~~~~

* Date [in *YYYY-MM-DD* format]
* T\ :sub:`max` = maximum daily air temperature
* T\ :sub:`min` = minimum daily air temperature
* T\ :sub:`d` = mean daily dew point temperature\ :sup:`a`
* e\ :sub:`a` = actual vapor pressure\ :sup:`a`
* u\ :sub:`x` = mean daily wind speed at known height\ :sup:`b`
* R\ :sub:`n` = calculated net radiation at the crop surface\ :sup:`c`
* P\ :sub:`r` = daily precipitation (optional)
* Q = specific humidity (optional)
* S\ :sub:`n` = daily accumulated snow (optional)
* S\ :sub:`d` = snow depth (optional)

\ :sup:`a` One of these is required. If both T\ :sub:`d` and e\ :sub:`a` are not provided, mean monthly dew point depression, K\ :sub:`0`, must be provided.

\ :sup:`b` Wind measurement height in meters must be provided in the *ini* file.

\ :sup:`c` If R\ :sub:`n` is not provided it will be estimated using the approach described in Thornton and Running
`(1998) <https://doi.org/10.1016/S0168-1923(98)00126-9>`_. The three Thornton and Running
coefficients must be provided in the *ini* file.

Hourly Variables
~~~~~~~~~~~~~~~~~

* Date [in *YYYY-MM-DD HH:MM* format]
* T\ :sub:`mean` = mean hourly air temperature
* T\ :sub:`d` = mean hourly dew point temperature\ :sup:`a`
* u\ :sub:`x` = mean hourly wind speed at known height\ :sup:`b`
* R\ :sub:`n` = calculated net radiation at the crop surface\ :sup:`c`
* P\ :sub:`r` = hourly precipitation (optional)
* Q = specific humidity (optional)
* S\ :sub:`n` = hourly accumulated snow (optional)
* S\ :sub:`d` = snow depth (optional)

\ :sup:`a` One of these is required. If both T\ :sub:`d` and e\ :sub:`a` are not provided, mean monthly dew point depression, K\ :sub:`0`, must be provided.

\ :sup:`b` Wind measurement height in meters must be provided in the *ini* file.

\ :sup:`c` If R\ :sub:`n` is not provided it will be estimated using the approach described in Thornton and Running
`(1998) <https://doi.org/10.1016/S0168-1923(98)00126-9>`_. The three Thornton and Running coefficients must be provided
in the *ini* file.

# HOW IS THE SNOW OR SNOW DEPTH USED?

File Format
~~~~~~~~~~~
RefET requires the timeseries weather data to be in delimited columns with header of column names. Column names and units are specified in the *ini* file. Files are allowed to have header rows, with the number of header rows specified in the *ini* file. The delimiter is also specified in the *INI* file.

+-----------+----------------+
| Delimiter | Model Notation |
+===========+================+
| Comma     | ,              |
+-----------+----------------+
| Tab       | /t             |
+-----------+----------------+

* Format: **.csv**, **.txt**, **.dat**

* File Name: (DESCRIBE WILDCARDS)

* Structure:

+--------------+---------------+---------------+-------------+-------------+-------------+
| Date         | Tmax          | Tmin          | ux          | Rn          | Tdew        |
+==============+===============+===============+=============+=============+=============+
| 2017-10-01   | 9.34          | 3.70          | 3.95        | 120.93      | 3.21        |
+--------------+---------------+---------------+-------------+-------------+-------------+
| 2017-10-02   | 5.52          | -2.12         | 7.54        | 59.10       | -3.18       |
+--------------+---------------+---------------+-------------+-------------+-------------+
| ...          | ...           | ...           | ...         | ...         | ...         |
+--------------+---------------+---------------+-------------+-------------+-------------+

* Units

+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Class           | Variables      | Units                                  | Model Notation                                                           |
+=================+================+========================================+==========================================================================+
| Temperature     | T\ :sub:`max`, | °C                                     | c                                                                        |
|                 |                |                                        |                                                                          |
|                 | T\ :sub:`min`, | °F                                     | f                                                                        |
|                 |                |                                        |                                                                          |
|                 | T\ :sub:`d`    | °K                                     | k                                                                        |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Wind Speed      | u\ :sub:`x`    | m s\ :sup:`-1`                         | m/s, mps                                                                 |
|                 |                |                                        |                                                                          |
|                 |                | m d\ :sup:`-1`                         | m/d, m/day                                                               |
|                 |                |                                        |                                                                          |
|                 |                | mi d\ :sup:`-1`                        | miles/d, miles/day, mpd                                                  |
|                 |                |                                        |                                                                          |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Solar Radiation | R\ :sub:`n`    | MJ m\ :sup:`-2` d\ :sup:`-1`           | mj/m2, mj/m^2, mj/m2/d, mj/m^2/d, mj/m2/day, mj/m^2/day                  |
|                 |                |                                        |                                                                          |
|                 |                | W m\ :sup:`-2` d\ :sup:`-1`            | w/m2, w/m^2                                                              |
|                 |                |                                        |                                                                          |
|                 |                | cal cm\ :sup:`-2` d\ :sup:`-1`         | cal/cm2', cal/cm2, cal/cm2/d, cal/cm^2/d, cal/cm2/day, cal/cm^2/day      |
|                 |                |                                        |                                                                          |
|                 |                | langley                                | langley                                                                  |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Precipitation   | P\ :sub:`r`    | in  d\ :sup:`-1`                       | in/d, in/day, inches/d, inches/day                                       |
|                 |                |                                        |                                                                          |
|                 |                | mm  d\ :sup:`-1`                       | mm/d, mm/day                                                             |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Humidity        | Q              | kg kg\ :sup:`-1`                       | kg/kg                                                                    |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Vapor Pressure  | e\ :sub:`a`    | kPa                                    | kPa                                                                      |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Snow            | S\ :sub:`n`    | in  d\ :sup:`-1`                       | in/d, in/day, inches/d, inches/day                                       |
|                 |                |                                        |                                                                          |
|                 |                | mm  d\ :sup:`-1`                       | mm/d, mm/day                                                             |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+
| Snow Depth      | S\ :sub:`g`    | in                                     | in, inches, `in*100`                                                     |
|                 |                |                                        |                                                                          |
|                 |                | mm                                     | mm                                                                       |
+-----------------+----------------+----------------------------------------+--------------------------------------------------------------------------+

Mean Monthly Data
"""""""""""""""""
Mean monthly data are used to calculate a dew point temperature timeseries or gap fill the timeseries data if needed.

* T\ :sub:`max` = mean monthly maximum daily air temperature
* T\ :sub:`min` = mean monthly minimum daily air temperature
* u\ :sub:`x` = mean monthly wind speed at known height
* K\ :sub:`0` =  mean monthly dew point depression

File Format
~~~~~~~~~~~

* Delimiter:

See *Timeseries Data - Delimiter*

* Format: **.csv**, **.txt**, **.dat**

* Structure:

+-------------+---------------+-----+-----+-----+------+------+------+------+------+------+------+-----+-----+
| Met Node ID | Met Node Name | Jan | Feb | Mar | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov | Dec |
+-------------+---------------+-----+-----+-----+------+------+------+------+------+------+------+-----+-----+
| BFAM        | Blackfeet, MT | 2.6 | 3.0 | 7.3 | 11.6 | 16.8 | 21.5 | 28.3 | 23.5 | 17.9 | 13.6 | 7.3 | 1.7 |
+-------------+---------------+-----+-----+-----+------+------+------+------+------+------+------+-----+-----+

*T\ :sub:`max`* example shown. File structure will be the same for T\ :sub:`min`, u\ :sub:`x`, and K\ :sub:`0`. Individual files are provided for each variable.

* Units

See *Timeseries Data - Units*

Ancillary Data
""""""""""""""

* Thornton and Running Coefficients
  * TR\ :sub:`b0`
  * TR\ :sub:`b1`
  * TR\ :sub:`b2`

Thornton and Running coefficients are used to estimate solar radiation from meteorological data. These coefficients
are location-specific and should be calibrated using measured solar radiation data from a representative location. The
calibration approach is described in detail :ref:`here <model-calibration-refet-tr>`



* Wind Measurement Height (in meters)

.. _model-inputs-cropet:

CropET
------

Weather Data
^^^^^^^^^^^^

Timeseries Data
"""""""""""""""
The user must provide daily weather and reference ET data for each ET zone. This includes:

* Date [in *YYYY-MM-DD* format]
* T\ :sub:`max` = maximum daily air temperature
* T\ :sub:`min` = minimum daily air temperature
* T\ :sub:`d` = mean daily dew point temperature\ :sup:`a`
* u\ :sub:`x` = mean daily or hourly wind speed at known height\ :sup:`b`
* R\ :sub:`n` = calculated net radiation at the crop surface\ :sup:`c`
* Q = specific humidity (optional)
* S\ :sub:`n` = daily accumulated snow (optional)
* S\ :sub:`d` = snow depth (optional)

and one of two reference ET values:

* ASCEr - Daily reference ET from Penman–Monteith
* ASCEg - Daily reference ET from Penman–Monteith

File Format
~~~~~~~~~~~

* Format: **.csv**

* Structure:

+--------+--------+--------+----------+----------+------+---------+--------+--------+------------------------+
| Date   | TMax   | TMin   | Precip   | Snow     | SDep | EstRs   | EsWind | EsTDew | *ETRef* [ASCEr; ASCEg] |
+--------+--------+--------+----------+----------+------+---------+--------+--------+------------------------+
| Units  | [C]    | [C]    | [In]     | [In]     | [In] | [MJ/m2] | [m/s]  | [C]    | [mm/day]               |
+--------+--------+--------+----------+----------+------+---------+--------+--------+------------------------+


Location Shapefile
""""""""""""""""""

A shapefile containing the locations of each weather station is also required and is used to generate the static input files. The shapefile must contain the following attributes:

* STATION_ID - Weather station ID
* *ETZONE_ID* - Zone ID. This can include HUC8, HUC10, COUNTRYNAME, OR GRIDMET_ID
* LAT - Weather station latitude
* LON - Weather station longitude
* [*optional*] *ELEV* [ELEV_FT; ELEV_M] - Weather station elevation in feet or meters. This field is optional and only required if running the RefET model to estimate reference ET.

File Format
~~~~~~~~~~~

* Format: **.shp**

* Attribute Table Structure:

+--------------+----------------------------------------------------+-------+-------+----------------------------+
| STATION_ID   | *ZONE_ID* [HUC8; HUC10; COUNTRYNAME; GRIDMET_ID]   | LAT   | LON   | *ELEV* [ELEV_FT; ELEV_M]   |
+--------------+----------------------------------------------------+-------+-------+----------------------------+


Study Area
^^^^^^^^^^
The user must provide a study area polygon shapefile with at least one feature.  Each feature in the study area shapefile will become a separate ET cell/unit.  Currently, only HUC8, HUC10, county, and gridmet cell shapefiles are fully supported by the prep tools.


Soils Data
^^^^^^^^^^
Shapefiles containing features (polygons) attributed with AWC and hydrologic soils group information are required for running zonal statistics on the  soil properties for each of the ETZones in the cell shapefile. The following attributes are required/used in the examples:

Available Water Capacity  Shapefile (AWC_WTA_0to152cm_statsgo.shp is the example)

* AWC = available water capacity (in/in)

Percent Clay  Shapefile (Clay_WTA_0to152cm_statsgo.shp is the example)
	
* Clay = percent clay (decimal)

Percent Sand Shapefile (Sand_WTA_0to152cm_statsgo.shp is the example)

* Sand = percent sand (decimal)

**File Formats**

* Format: **.shp**
* Attribute Table Structure (examples’ file structures shown):

  AREA SYMBOL, SPATIALVER, MUSYM, MUKEY, MUKEY_1, AWC (or Clay or Sand depending on which shapefile)


Crop Type Data
^^^^^^^^^^^^^^
Shapefiles containing features (polygons) attributed with a CDL code are required  for ETDemands. The features should only include agricultural areas within the ETZones; the example CDL shapefile has removed all non-agricultural areas within the respective ETZones. The shapefile should have the following field:

* CDL (Int)

**File Formats**

* Format: **.shp**
* Attribute Table Structure:
	
 CDL


Static Inputs
^^^^^^^^^^^^^
These files will be generated automatically by the CropETPrep module



CropCoefs
"""""""""
Crop coefficient curves for each crop. Generally, these values should not be modified. The crop coefficients are used in ETDemands to reflect changes of vegetation phenology of a particular crop or vegetation type over the growing season. The crop coefficient curves (60 count) represent changes in vegetation phenology, which can vary from year to year depending on the start, duration, and termination of the growing season, all of which are dependent on air temperature conditions during spring, summer, and fall. Three methods were used to define the shape and duration of the crop coefficient curves to allow curves to be scaled differently each year according to weather conditions based on relative time scales or thermal units. These methods are:  

* Normalized cumulative growing degree-days from planting to effective full cover, with this ratio extended until termination (1=NCGDD)
* Percent time from planting to effective full cover, with this ratio extended until termination (4=%PL-Term)
* Percent time from planting to effective full cover and then number of days after full cover to termination (3=%PL-EC+daysafter)

**File Format:**

* Format: **.txt**
* Structure: 

Curve no.: 1-60
Curve type: ‘1=NCGDD: 2=%PL-EC: 3=%PL-EC+daysafter: 4=%PL-Term
Percent PL-EC or PL-TM (type 1-2-4) and/or Percent PL-EC+ days after (type 3)

GDD Base C
GDD Type
CGDD Planting to FC
CGDD Planting to Terminate
CGDD Planting to Terminate-alt
Comment:
Comment 2:


CropParams
""""""""""
Crop parameters that can/should be modified during calibration.

* Format: **.txt**

* Structure:
Crop number and flag for crop type: negative is annual; positive in perennial
Irrigation flag: 1-yes, 2-reg., 3-required
Days after planting/green up for earliest irrigation: days
Fw: assume sprinkler
Winter surface cover class: 1-bare, 2-mulch, 3-sod
Kc max: max of value or Kcb+0.05
MAD during initial and development stage: percent
MAD during midseason and lateseason: percent
Initial rooting depth, m: On alfalfa, 2nd cycle, start at max
Maximum rooting depth, m: mrd
End of root growth, as a fraction of time from pl to EFC (or term if type 4)
Starting crop height, m: sch
Maximum crop height, m: mch
Crop curve number: ccn
Crop curve name: ccn
Crop curve type: 1=NCGDD, 2=%PL-EC, 3=%PL-EC,daysafter, 4=%PL-Term
Flag for means to estimate pl or gu: 1=CGDD, 2=T30, 3=date, 4 is on all the time
T30 for pl or gu or CGDD for pl or gu
Date of pl or gu (can be blank): A negative value is an offset to the prior row, pos is months (fraction)
For nCGDD based curves: Tbase: Temp Min. C (neg. For spec.)
	CGDD for EFC: cgdd efc
	CGDD for termination: cgdd term
	
For time based curves:
	Time for EFC: days after pl or gu
	Time for harvest (neg to extend until frost): Use as max length for CGDD crops
Killing frost temperature: C
Invoke Stress: 1-yes, 0-no, 2-yes and will wake up after severe stress (Ks<0.05)
Curve number:
	Coarse soil
	Medium soil
	Fine soil


ETCellsCrops
""""""""""""
Flags controlling which crops to simulate.  If using the prep workflow, the flags will initially be set based on the CDL acreage.

* Format: **.txt**
* Structure:
Number of Crops: XX,	Crop Number (CDL): XX...
ET Cell ID/ET Index,	ET Cell Name,	Ref ET ID/Met Node Id,	ET Cell Irrigation (0 is off; 1 is on)


EToRatiosMon
""""""""""""
Reference ET scale factors by month for each ET cell.  This file could be used to account for a seasonal bias in the input weather data.  This file is optional.

* Format: **.txt**
* Structure:
Met Node ID, Met Node, Month….

ETCellsProperties
"""""""""""""""""
Soil properties and weather station data for each ET cell.  This file links the stations and the ET cells.

* Format: **.txt**
* Structure:

+--------------+--------------+-----------------+----------------+-----------------+--------------------+--------------------------------------------+------------------------------------+-------------------------------------------------------------------------+-------------------------------------------------+----------------+------------------+
| ET Zone ID   | ET Zone Name | Ref ET MET ID   | Met Latitude   | Met Longitude   | Met Elevation (ft) | Area weighted average Permeability - in/hr |	Area weighted average WHC - in/ft |	Average soil depth - in	Hydrologic Group (A-C) (A='coarse'  B='medium') |	Hydrologic Group  (1-3) (1='coarse' 2='medium') |	Aridity Rating | Ref ET Data Path |
+--------------+--------------+-----------------+----------------+-----------------+--------------------+--------------------------------------------+------------------------------------+-------------------------------------------------------------------------+-------------------------------------------------+----------------+------------------+

MeanCuttings
""""""""""""
Sets the assumed number of alfalfa cuttings.  This is important since the CropET module will use different crop coefficient curves for the first and last cutting.

* Format: **.txt**

* Structure:
ET Cell ID, ET Cell Name, Lat (DD), Number Dairy, Number Beef
