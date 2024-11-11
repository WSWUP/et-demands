Quick Start
===========

Installing the Model
--------------------

Model
^^^^^

Clone/download the Crop ETDemands Model from the GitHub Repository here: `usbr/et-demands: Dual crop coefficient crop water demand model <https://github.com/usbr/et-demands>`_


Required Dependencies
^^^^^^^^^^^^^^^^^^^^^

It is recommended that Python 3.6+ be downloaded/used to run the ETDemands model.

The easiest way to install required external Python modules is to use `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and create separate conda environments for each project.

* NOTE: It is important to double check that you are calling Miniconda version, especially if you have two or more version of Python installed (e.g. Miniconda and ArcGIS). Use the Command Prompt (Windows) or Terminal (Linux/Mac) to check:
      * Windows: "where python"
      * Linux/Mac: "which python"


**Conda Forge**

After installing Miniconda, conda-forge should be added to the list of conda channels in order to have the most up to date python packages:

``> conda config --add channels conda-forge``


**Conda Environment**

To get started, it is recommended that the "etdemands" conda environment be created using the "environment.yml":

``> conda env create -f environment.yml``

After creating the environment, it then needs to be "activated":

``> conda activate etdemands``

**Installing Modules**

If you would prefer to not use the conda environment, see the README.md in the GitHub repo for details about installing necessary modules.

**GDAL_DATA**

After installing GDAL, you may need to manually set GDAL_DATA user environment variable (See the README.md in the GitHub repo for details about setting up gdal environment variable).

Downloading Examples
^^^^^^^^^^^^^^^^^^^^

There are two examples available in the GitHub repo to run the ETDemands model with similar workflows. The example workflow described in this section is for the HUC8 example, only. Command line calls to download/prep the input data are described below. Detailed walk-throughs of the example preparation workflow are available at the end of this document for both of the examples.  

First, navigate to the location of the HUC8 example on the target drive:

``> C:``

``> cd et-demands\examples\huc8``


**HUC8 Example Workflow** 

The HUC8 example workflow uses six HUC8 watersheds, or ETZones, extracted from the `USGS Watershed Boundary Dataset <https://www.usgs.gov/core-science-systems/ngp/national-hydrography/watershed-boundary-dataset?qt-science_support_page_related_con=4#qt-science_support_page_related_con>`_ (WBD) for the Upper Colorado River Basin .

The historical daily gridMET time series for the study area (2017-2019) is provided for this working example. Along with meteorological data, the time series includes daily grass and alfalfa reference evapotranspiration (ETo and ETr, respectively). The time series was downloaded from the `gridwxcomp repo <https://github.com/WSWUP/gridwxcomp/>`_ using a .csv list of gridMET stations in the example’s “climate” subfolder and the download_gridmet_opendap.py script from gridwxcomp. 

Before proceeding, make sure to change all of the relative paths that are defined in the two .ini files (huc_example_cet.ini and huc_example_prep.ini) to reflect your system’s setup.

A crop shapefile is required to run the example. A 2015 Cropland Data Layer (CDL) can be downloaded to the “common” folder using the following:

``> python ..\..\et-demands\prep\download_cdl_raster.py --ini huc_example_prep.ini -o``

Clip the CDL raster to the ETZones extent:

``> python ..\..\et-demands\prep\clip_cdl_raster.py --ini huc_example_prep.ini -o``

Convert the clipped CDL raster to a polygon, with any non-agricultural areas removed:

``> python ..\..\et-demands\prep\build_ag_cdl_shapefile.py --ini huc_example_prep.ini -o``

Download the pre-computed STATSGO2 Soils shapefile:

``> python ..\..\et-demands\prep\download_statsgo_shapefiles.py --ini huc_example_prep.ini -o``


Running Examples
^^^^^^^^^^^^^^^^

The example first requires soil properties and crop acreage for each of the ETZones polygons to be computed. Zonal statistics are run using the following command: 

``> python ..\..\et-demands\prep\et_demands_zonal_stats.py --ini huc_example_prep.ini``

The last step before running the model is to build the static text files from the templates that are provided in et-demands\static:

``> python ..\..\et-demands\prep\build_static_files.py --ini huc_example_cet.ini``

ETDemands is run for the huc8 example using the run_cet.py script using the following:

``> python ..\..\et-demands\cropET\bin\run_cet.py --ini huc_example_cet.ini -b ..\..\et-demands\cropET\bin -v -d``

To analyze/visualize the model outputs, there are some summary and plotting tools available. Running the following will create plots of the daily crop time series:

``> python ..\..\et-demands\tools\plot_crop_daily_timeseries.py --ini huc_example_cet.ini``

See the Analysis Tools section for a more detailed explanation of the available post-processing features.

