CropET Input Data
=================

Cell Shapefile
--------------
The smallest spatial unit in the ET-Demands model is the "cell" (sometimes referred to as units, zones or ET cells throughout the code and documentation).  The user must provide a cell shapefile with at least one polygon feature.  Each cell can have a variety of crop types and acreages.

Typically each cell will be paired with data from separate weather station, although one to many relationships between the stations and cells are supported.  The cell shapefile must have a column/field specifying the weather station ID that should be used named "STATION_ID" (see Weather Station Points section below).

Cells are commonly defined as HUC8s, counties, or raster grids.  Examples with HUC8 and GRIDMET cells are provided in the examples folder.

HUC8 and HUC10 features can be extracted from the full `USGS Watershed Boundary Dataset <http://nhd.usgs.gov/wbd.html>`_ (WBD) geodatabase.  A subset of the WBD HUC polygons can downloaded using the `USDA Geospatial Data Gateway <https://gdg.sc.egov.usda.gov/>`_ or the full dataset can be downloaded using the `USGS FTP <ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/WBD/>`_.

County features can be downloaded from the `USDA Geospatial Data Gateway <https://gdg.sc.egov.usda.gov/>`_.  Other census features (such as the `US Census Cartographic Boundary Shapefiles <https://www.census.gov/geo/maps-data/data/tiger-cart-boundary.html>`_) could also be used.

If the user is running the model with gridded weather data (e.g. `GRIDMET <http://www.climatologylab.org/gridmet.html>`_), the cell features can be built using the ArcGIS fishnet tool or GDAL.

The cells shapefile must be in a projected coordinate system (i.e. not NAD83 or WGS84 geographic).

Weather Station Points
----------------------
To generate the ET-Demands static input files, the user must provide a weather station shapefile with at least one point feature.  The shapefile must have column/field of the weather station ID named "STATION_ID".  The shapefile must also have fields for the station latitude and longitude (in degrees), named "LAT" and "LON" respectively.  These field names are currently hard coded into the scripts.

The weather station elevation is not currently used in the ET-Demands CropET module, but should also be included in the stations shapefile.  The station elevation should be named ELEV_FT or ELEV_M depending on the whether the units are in feet or meters respectively.

Weather Data
------------
The user must provide weather and reference ET data for each weather station.

Crop Shapefile
--------------
The crop types and acreages for each cell are defined using a "crop" shapefile.

If the user does not have an existing crop shapefile, the prep tools can be used to generate one using the Cropland Data Layer raster (see below).

The crop shapefile can also used to mask out non-agricultural areas when computing the average soil conditions.

The user defined crop types need to be mapped to the ET-Demands crop types using a crosswalk CSV file.  This file should be specified in the prep INI file by the "crosswalk_path" parameter.  An example of the mapping of CDL crop codes to ET-Demands crop codes can be found in in the prep folder (et-demands\\prep\\cdl_crosswalk_default.csv).

Cropland Data Layer (CDL)
`````````````````````````
The crop shapefile can be generated from the CDL raster using the provided tools (see prep_tools section).  Because the ET-Demands crop type numbers differ If the user does not have an existing crop shapefile, the CDL raster can be used to generate one determine the acreage of crops in each ET cell/unit.

.. _data-soils:

Soils
-----
The average agricultural area available water capacity (AWC) and hydrologic soils group must be computed for each cell.  The hydrologic soils group can be estimated based on the percent sand and clay.  The AWC and percent clay/sand are typically extracted from the STATSGO2 or SSURGO datasets.

SSURGO has a higher resolution (1:12,000 to 1:63,360) than STATSGO2 (1:250,000), but the SSURGO datasets cover a smaller area and may have areas of missing data.  Since the ET estimates are generally not sensitive to the these parameters and the values are averaged to the "cell", the higher resolution and complexity of SSURGO may not be needed.

Precomputed CONUS STATSGO2 shapefiles can be downloaded using the provided tool (see the prep_tools section).  These shapefiles were generated using the soil data viewer with X options.

The `STATSGO2 <http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053629>`_ statewide databases can be downloaded from the `USDA Geospatial Data Gateway <https://gdg.sc.egov.usda.gov/>`_.  Shapefiles of the soil properties can be extracted using the `NRCS Soil Data Viewer <http://www.nrcs.usda.gov/wps/portal/nrcs/detailfull/soils/home/?cid=nrcs142p2_053620>`_.  The `SSURGO <http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053627>`_ databases can also be used by the soil data viewer.  It may also be possible to used the gridded SSRUGO data, but this has not been tested.

The file paths to the soils shapefiles are set in the prep INI file.  The names of the downloaded soil shapefiles are "{}_WTA_0to152cm_statsgo.shp", where {} can be "AWC", "Clay", or "Sand" (see :docs:`structure`).  For each shapefile, the value field name is hardcoded as the upper case of the property (i.e. "AWC", "CLAY", or "SAND").
