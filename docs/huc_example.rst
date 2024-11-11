
HUC8 Example Prep Workflow
===================
The following documentation outlines the prep workflow for running ETDemands over 6 HUC8 watersheds in the Upper Colorado River Basin
from 2017-2018. The example utilizes 2015 CDL crop layer information along with historical weather data from gridMET. Each HUC8 watershed
was assigned weather data from a single gridMET cell. Two of the ETZones utilize the same weather station in order to demonstrate
the ETZone/Wx Station pairing options.

NOTE: The ETDemands repository contains many of the files created by the following workflow for demonstration purposes. Re-running
the prep workflow will not cause any issues and will write over the existing files with identical (updated) files.

Clone the repository
--------------------
If you already have a local copy of the et-demands repository, make sure to pull the latest version from GitHub.  If you don't already
have a local copy of the repository, either clone the repository locally or download a zip file of the scripts from Github.

NOTE: For this example, it is assumed that the repository was cloned directly to the C: drive (i.e. C:\\et-demands).

Command prompt / Terminal
-------------------------
All of the CropET prep scripts should be run from the command prompt or terminal window.  In Windows, to open the command prompt, press
the "windows" key and "r" or click the "Start" button, "Accessories", and then "Command Prompt".

Within the command prompt or terminal, change to the target drive if necessary::

    > C:

Then change directory to the et-demands folder::

    > cd C:\et-demands

You may need to build the common folder if it doesn't exist::

    > mkdir common

Building the Example
--------------------
NOTE: For this example, all scripts and tools will be executed from the "examples\huc8" folder.

Build the example folder if it doesn't exist::

    > mkdir examples\huc8

Change directory into the example folder::

    > cd examples\huc8

Cell Shapefile
--------------
For this example, ET-Demands will be run for a six HUC8 watersheds in the Upper Colordo River Basin
(14080001, 14080002, 14080003, 14080004, 14080005). These ETZones were extracted from the full
`USGS Watershed Boundary Dataset <http://nhd.usgs.gov/wbd.html>`_ (WBD) geodatabase. Each individual ETZone
within the .shp must be assigned a unique 'CELL_ID' and 'CELL_NAME'. These unique identfiers are used
throughout the model for identification and output file naming. In addition to 'CELL_ID', the Cell .shp must also
contain a 'STATION_ID' field to pair each ETZone with its specific weather dataset (see Weather Stations
section below).

Weather Stations
----------------
The example stations are single 4km cells selected from the `University of Idaho Gridded Surface Meteorological Data
<http://metdata.northwestknowledge.net/>`_. Each ETZone was assigned a specific historical weather dataset in the 
'STATION_ID' field of the Cell Shapefile. Note that it is possible to assign the same historical weather dataset to
multiple ETZones. The historical weather data file naming format is specific using the 'name_format' variable
in the model .INI file (huc_example.INI).

For this example each weather station data file follows the format:
'gridmet_historical_XXXXXX.csv' where XXXXXX represents a 6-digit wx station identifier.
In the REFET section of the model .INI file, name_format = gridmet_historical_%s.csv.
A similar ID/Filenaming structure should be used to link each 'STATION_ID' with its corresponding timeseries file.

Historical gridMET time series supplied with this repor wer acquired using the download_gridmet_opendap.py tool
found in the `gridwxcomp repository <https://github.com/WSWUP/gridwxcomp/>`_. The climate folder contains a .txt list of gridMET stations (gridmet_huc8_stations.csv). This can be used with download_gridmet_opendap.py to download station
data. Output format from the download_gridmet_opendap.py script is ready for the INI supplied with this example.

Crop Shapefile
--------------
For this example, the crop shapefile will be built from the 2015 Cropland Data Layer (CDL) raster.

For this example the CDL raster will be downloaded directly to the project GIS folder. ::

    > python ..\..\et-demands\prep\download_cdl_raster.py --ini huc_example_prep.ini -o

If the download script doesn't work, please try downloading the `2015_30m_cdls.zip <ftp://ftp.nass.usda.gov/download/res/2010_30m_cdls.zip>`_ file directly from your browser or using a dedicated FTP program.

Clip the CDL raster to the cell shapefile extent::

    > python ..\..\et-demands\prep\clip_cdl_raster.py --ini huc_example_prep.ini -o

Convert the clipped CDL raster to a polygon, with the non-agricultural areas removed::

    > python ..\..\et-demands\prep\build_ag_cdl_shapefile.py --ini huc_example_prep.ini -o

Soils
-----
Download the pre-computed STATSGO2 shapefiles::

    > python ..\..\et-demands\prep\download_statsgo_shapefiles.py --ini huc_example_prep.ini -o

Zonal Stats
-----------
Compute the soil properties and crop acreages for each ETZone polygon. ::

    > python ..\..\et-demands\prep\et_demands_zonal_stats.py --ini huc_example_prep.ini

Static Text Files
-----------------
Build the static text files from the templates in "et-demands\\static". ::

    > python ..\..\et-demands\prep\build_static_files.py --ini huc_example.ini
