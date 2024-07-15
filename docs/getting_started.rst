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

Once installed, several f
GDAL
~~~~
The Geospatial Data Abstraction Library (`GDAL <https://www.gdal.org/>`_) is required
for ET-Demands. The easiest way to install GDAL for Windows is to use the
`OSGeo4W <https://trac.osgeo.org/osgeo4w/>`_ installer. Once installed, GDAL

Creating a Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We encourange the use of a project-specific Python environment when running
ET-Demands. An "etdemands" conda environment can be created using the
"environment.yml" file in the project main directory.

``
conda env create -f environment.yml
``

This will create an etdemands Python environment as well as install all required
Python packages. An alternative if the etdemands Python environment is not desired
is to install the required Python packages manually using the following:

``
conda install bokeh=0.13.0 gdal=2.3.1  numpy=1.15.0 pandas=0.23.4 openpyxl=2.5.5
``

The following packages are not included within the environment.yml file, but are required to use some of the post-processing tools:
Matplotlib
Fiona
Descartes

The following command line call will download the packages above (make sure to download these after activating the etdemands environment):

''
conda install matplotlib fiona descartes
''

Model Inputs
------------
ET-Demands requires

Model Control Files
-------------------

ETDemands is controlled using two initialization (.INI) files. The model prep .INI file is used to run the *cropET* prep tools. Users must change the default paths to their system’s relative paths in order to prep/analyze input datasets correctly. The *cropET* .INI file is used to run the ETDemands model. Similar to the model prep .INI file, users must define their relative paths in the CropET .INI file in order to run the ETDemands model correctly. 

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

