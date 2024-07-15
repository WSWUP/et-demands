
gridMET Example Prep Workflow
===================
The following documentation outlines the prep workflow for running ETDemands for
the Upper Carson River Basin (HUC16050201) in northeastern Nevada for 2017-2018.
The Upper Carson watershed was separated into 4km grid cells based on the
gridMET meteroloical dataset. Out of the 184 gridMET cells overlapping the Upper
Carson Basin, 70 cells contained irrigated agriculture. Irrigated cells were
determined using the 2015 Crop Data Layer (CDL). Each grid cell represents an
ETZone of interest for this region. ETDemands will calculate crop area and
evaporation metrics for each ETZone separately. Historical climate timeseries
were aquired from 2017-20187 for each unique ETZone from the gridMET climate
dataset.

NOTE: The ETDemands repository contains many of the files created by the
following workflow for demonstration purposes. Re-running the prep workflow
will not cause any issues and will write over the existing files with identical
(updated) files.

Clone the repository
--------------------
If you already have a local copy of the et-demands repository, make sure to pull
the latest version from GitHub.  If you don't already have a local copy of the
repository, either clone the repository locally or download a zip file of the
scripts from Github.

NOTE: For this example, it is assumed that the repository was cloned directly to
the C: drive (i.e. C:\\et-demands).

Command prompt / Terminal
-------------------------
All of the CropET prep scripts should be run from the command prompt or terminal
window.  In Windows, to open the command prompt, press the "windows" key and "r"
or click the "Start" button, "Accessories", and then "Command Prompt".

Within the command prompt or terminal, change to the target drive if necessary::

    > C:

Then change directory to the et-demands folder::

    > cd C:\et-demands

You may need to build the common folder if it doesn't exist::

    > mkdir common

Building the Example
--------------------
NOTE: For this example, all scripts and tools will be executed from the "examples\gridmet" folder.

Build the example folder if it doesn't exist::

    > mkdir examples\gridmet

Change directory into the example folder::

    > cd examples\gridmet

Cell Shapefile
--------------
For this example, ET-Demands will be run for a 70 gridMET cells in the Upper
Carson River Basin. Each individual ETZone within the shapefile must be assigned
a unique 'CELL_ID' and 'CELL_NAME'. The unique identfiers are used throughout
the model for identification and output file naming. In addition to 'CELL_ID',
the cells .shp must also contain a 'STATION_ID' field to pair each ETZone with
its specific weather dataset (see Weather Stations section below).

Weather Stations
----------------
The example stations are single 4km cells selected from the
`University of Idaho Gridded Surface Meteorological Data
<http://metdata.northwestknowledge.net/>`_. Each ETZone was assigned a specific
historical weather dataset in the 'STATION_ID' field of the Cell Shapefile.
The historical weather data file naming format is specific using the
'name_format' variable in the model .INI file (gridmet_example.INI).

Historical gridMET time series was acquired using the download_gridmet_opendap.py tool
found in the `gridwxcomp repository <https://github.com/WSWUP/gridwxcomp/>`_. The climate folder contains a .txt list of gridMET stations. This can be used with download_gridmet_opendap.py to download station
data. Output format from the download_gridmet_opendap.py script is ready for the INI supplied with this example::

    > python download_gridmet_opendap.py -i D:\et-demands\examples\gridmet\climate\upperCarson_agpts.txt -o D:\et-demands\examples\gridmet\climate -y 2017-2018

For this example each weather station data file follows the format:
'gridmet_historical_XXXXXX.csv' where XXXXXX represents a 6-digit gridMET cell identifier.
In the REFET section of the model .INI file, name_format = gridmet_historical_%s.csv.
A similar ID/Filenaming structure should be used to link each 'STATION_ID' with its corresponding timeseries file.

NOTE: The user should take caution when applying large gridded climate datasets to irrigated agricultural settings.
Gridded climate datasets often represent larger scale natural landscape conditions and do not factor in the impacts of
localized irrigation (i.e. evaportive cooling, changes in surface roughness, increased vapor pressure, etc.). ETDemands
offers options to scale (or bias correct) the input ETo/ETr values based on comparisons with local weather
station data. Please refer to 'et_ratios' functionality in the XXXXX documentation for further details.


Crop Shapefile
--------------
For this example, the crop shapefile will be built from the 2015 Cropland Data
Layer (CDL) raster.

For this example the CDL raster will be downloaded directly to the project GIS
folder. ::

    > python ..\..\et-demands\prep\download_cdl_raster.py --ini gridmet_example_prep.ini -o

If the download script doesn't work, please try downloading the
`2015_30m_cdls.zip <ftp://ftp.nass.usda.gov/download/res/2010_30m_cdls.zip>`_
file directly from your browser or using a dedicated FTP program.

Clip the CDL raster to the cell shapefile extent::

    > python ..\..\et-demands\prep\clip_cdl_raster.py --ini gridmet_example_prep.ini -o

Convert the clipped CDL raster to a polygon, with the non-agricultural areas
removed::

    > python ..\..\et-demands\prep\build_ag_cdl_shapefile.py --ini gridmet_example_prep.ini -o

Soils
-----
Download the pre-computed STATSGO2 shapefiles::

    > python ..\..\et-demands\prep\download_statsgo_shapefiles.py --ini gridmet_example_prep.ini -o

Zonal Stats
-----------
Compute the soil properties and crop acreages for each ETZone polygon. ::

    > python ..\..\et-demands\prep\et_demands_zonal_stats.py --ini gridmet_example_prep.ini

Static Text Files
-----------------
Build the static text files from the templates in "et-demands\\static". ::

    > python ..\..\et-demands\prep\build_static_files.py --ini gridmet_example.ini
