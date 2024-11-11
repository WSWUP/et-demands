Getting Started
===============

Installation
------------

Dependencies
^^^^^^^^^^^^
ET-Demands requires Python 3.x, the following Python packages:

- bokeh>=0.13.0
- gdal>=2.2.4
- geopandas>=0.4.0
- numpy>=1.15.0
- openpyxl=2.5.5
- pandas>=0.23.4
- pyshp=1.2.12
- pytest=3.8.0
- python=3.6.6
- rtree
- shapely
- xlrd>=1.1.0
- requests

and GDAL. Steps for installing all dependencies are described below.

Python
~~~~~~
ET-Demands requires Python 3.x. While any install of Python will work,
he easiest way to install Python and the required modules is to use
`Miniconda <https://conda.io/miniconda.html>`_ and create a separate
Python environment.

`conda-forge <https://conda-forge.github.io/>`_ should be added to the list of
conda channels

Miniconda can be installed to any location, but the install location should be
noted as it will be used later in the setup. This will be referred to in this
documentation as PYTHONPATH (e.g. C:\Programs\miniconda3\).

Once installed, several f GDAL ~~~~ The Geospatial Data Abstraction Library (`GDAL <https://www.gdal.org/>`_) is required
for ET-Demands. The easiest way to install GDAL for Windows is to use the
`OSGeo4W <https://trac.osgeo.org/osgeo4w/>`_ installer. 

Creating a Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We encourange the use of a project-specific Python environment when running
ET-Demands. An "etdemands" conda environment can be created using the
"environment.yml" file in the project main directory.

``> conda env create -f environment.yml``

This will create an etdemands Python environment as well as install all required
Python packages. An alternative if the etdemands Python environment is not desired
is to install the required Python packages manually using the following:

``> conda install bokeh=0.13.0 gdal=2.3.1  numpy=1.15.0 pandas=0.23.4 openpyxl=2.5.5``

|
The following packages are not included within the environment.yml file, but are required to use some of the post-processing tools:

- Matplotlib
- Fiona
- Descartes
|
The following command line call will download the packages above (make sure to download these after activating the etdemands environment):

``> conda install matplotlib fiona descartes``


Model Inputs
------------

|
**Cell Shapefile**

The ETDemands model prep workflow requires a user to provide a study area polygon shapefile with at least one feature. For the example workflows, HUC8 boundaries (WBDHU8_32612.shp) or 4-km gridMET cells (uppercarson_agcells.shp) are used as the ETZones. Each individual ETZone (i.e. feature/polygon) within the shapefile must be assigned a unique ‘CELL_ID’ and ‘CELL_NAME’. These attributes are used throughout the model for identification and output file naming. In addition to ‘CELL_ID’, the shapefile must also contain a ‘STATION_ID’ field so that the ETZone can be paired with its respective weather dataset (see Weather Stations section below). 

|
**Weather Stations**

The RefET module requires hourly or daily meteorological data. If the calculated reference ET will be used in the CropET module, care should be taken to ensure this meteorological data is representative of agricultural conditions. Meteorological data can be obtained from `agricultural weather networks <https://et-demands.readthedocs.io/en/master/data_sources.html#data-sources-ag-met>`_ , or `adjustments <https://et-demands.readthedocs.io/en/master/model_description_code.html#model-description-cropet-aridfctr>`_ to the reference ET can be made within the CropET module.

The example stations are single 4-km cells selected from the `University of Idaho Gridded Surface Meteorological Data <http://www.climatologylab.org/gridmet.html>`_. Each cell, or ETZone, was assigned a specific historical weather dataset in the ‘STATION_ID’ field of the Cell Shapefile. Note that it is possible to assign the same historical weather dataset to multiple ETZones. The historical weather data file naming format is specific using the ‘name_format’ variable in the model .INI file (huc_example.INI).

For the example workflow, each weather station data file follows the format: ‘gridmet_historical_XXXXXX.csv’ where XXXXXX represents a 6-digit wx station identifier. In the REFET section of the model .INI file, name_format = gridmet_historical_%s.csv. A similar ID/File naming structure should be used to link each ‘STATION_ID’ with its corresponding timeseries file.

Historical gridMET time series supplied with this repo were acquired using the download_gridmet_opendap.py tool found in the gridwxcomp repository. The climate folder contains a .txt list of gridMET stations (gridmet_huc8_stations.csv). This can be used with download_gridmet_opendap.py to download station data. Output format from the download_gridmet_opendap.py script is ready for the INI supplied with this example. Specific daily or hourly meteorological variables required by the RefET module to calculate reference ET are listed in the Model Inputs section below.

|
**Crop Shapefile**

A polygon shapefile that delineates specific crop types is required for the ETDemands model. In the example workflows, the 2015 CDL raster is downloaded `(2015_30m_cdls.zip) <ftp://ftp.nass.usda.gov/download/res/2015_30m_cdls.zip>`_ and converted into a polygon shapefile that excludes all non-irrigated areas. Each polygon/feature that makes up the shapefile is attributed with a CDL code that designates which type of crop the area is. The CDL codes are used throughout the ETDemands model to keep various crop types and fields separate during the analysis. Users can specify whether they would like to analyze cropland or non-cropland areas  in the analysis by commenting/uncommenting the ‘cdl_crops’ variable or ‘cdl_nonag’ variable in the [CROP_ET_PREP] section of the model prep .INI file. A list of each crop type (and NLCD-derived classes) and their respective CDL codes is in the Appendix.

|
**Soils Data**

The ETDemands model requires the average agricultural area available water capacity (AWC) and hydrologic soils group for each ET cell. The hydrologic soils group can be estimated based on the percent sand and clay for each ET cell. The data can be downloaded from the `[STATSGO] <http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053629>`_ database for the target state(s) using the `[USDA Geospatial Data Gateway] <https://gdg.sc.egov.usda.gov/>`_. Shapefiles of the soil properties can be extracted using the `[NRCS Soil Data Viewer] <http://www.nrcs.usda.gov/wps/portal/nrcs/detailfull/soils/home/?cid=nrcs142p2_053620>`_ The `[SSURGO] <http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053627>`_  databases can also be used, but these typically cover a smaller area and may have areas of missing data. To use the soil prep tools, the soils data must be provided as separate shapefiles for each product. The names of the soil shapefiles are hard coded in the rasterize_soil_polygons.py script as “{}_WTA_0to152cm_statsgo.shp”, where {} can be “AWC”, “Clay”, or “Sand” (see [Model Structure](structure.md)). For each shapefile, the value field name is hard coded as the upper case of the property (i.e. “AWC”, “CLAY”, or “SAND”).

|
Model Control Files
-------------------

ETDemands is controlled using two initialization (.INI) files. The model prep .INI file is used to run the CropET prep tools. Users must change the default paths to their system’s relative paths in order to prep/analyze input datasets correctly. The CropET .INI file is used to run the ETDemands model. Similar to the model prep .INI file, users must define their relative paths in the CropET .INI file in order to run the ETDemands model correctly. 

|
Model Structure
---------------

ET-Demands is comprised of several modules with *prep* and *refET* containing
pre-processing tools, *cropET* containing the crop irrigation water requirement
calculations, and *areaET* and *tools* containing post-processing tools. The
source code model structure is shown below::

  et-demands
   |--- prep
   |    |--- download_cdl_raster.py
   |    |--- clip_cdl_raster.py
   |    |--- build_ag_cdl_shapefile.py
   |    |--- download_statsgo_shapefiles.py
   |    |--- et_demands_zonal_stats.py
   |    |--- build_static_files.py
   |    |--- interpolate_spatial_crop_params.py
   |
   |--- refET
   |    |--- run_ret.py
   |    |    |--- mod_ref_et.py
   |    |         |--- ret_config.py
   |    |         |--- met_nodes.py
   |    |              |--- ref_et_data.py
   |    |              |--- ret_utils.py
   |    |--- ret_utils.py
   |
   |--- cropET
   |    |--- run_cet.py
   |    |    |--- mod_crop_et.py
   |    |         |--- crop_et_data.py
   |    |         |    |--- crop_parameters.py
   |    |         |    |--- crop_coefficients.py
   |    |         |--- et_cell.py
   |    |         |    |--- shapefile.py
   |    |         |--- crop_cycle.py
   |    |         |    |--- initialize_crop_cycle.py
   |    |         |    |--- compute_crop_gdd.py
   |    |         |    |    |--- open_water_evap.py
   |    |         |    |--- calculate_height.py
   |    |         |    |--- kcb_daily.py
   |    |         |    |    |--- runoff.py
   |    |         |    |--- compute_crop_et.py
   |    |         |    |    |--- grow_root.py
   |    |--- util.py
   |
   |--- areaET
   |    |--- run_aet.py
   |    |    |--- mod_area_et.py
   |    |         |--- aet_config.py
   |    |         |--- aet_cells.py
   |    |         |    |--- ref_et_.py
   |    |--- aet_utils.py
   |
   |--- tools
   |    |--- compute_growing_season.py
   |    |--- cropweighted_shapefiles_gpd.py
   |    |--- example_check.py
   |    |--- indicatormethod_restructure.py
   |    |--- plot_crop_daily_groupstats.py
   |    |--- plot_crop_daily_timeseries.py
   |    |--- plot_crop_summary_maps.py
   |    |--- summary_shapefiles_gpd.py
   |    |--- util.py

Running the Model
-----------------

The prep portion of ETDemands requires the model inputs from above to be downloaded/prepped. Zonal statistics must then be run to assign crop acreages and soil properties to the various ETZones in the input shapefile. The final preparations are made by building the static text files from the templates in “et-demands\static”.

Once input data is downloaded/prepped and the static input files are generated, the CropET portion of ETDemands can be run. The CropET portion of the model does not use study area shapefiles and relies completely on static .txt generated by the prep workflow. NOTE ON SPATIALLY VARYING CALIBRATION: If run in spatially varying calibration mode, then the user must utlizes calibration .shp generated by the build_spatial_crop_params.py script.

Post-processing options are available in the Tools portion of ETDemands. Timeseries plots, summary shapefiles, and growing season summaries can be generated to help analyze results from the model runs. See the Running the Model and Analysis Tools sections for more details about running the post-processing tools.

