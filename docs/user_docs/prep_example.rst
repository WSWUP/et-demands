CropET Prep Example
===================

Clone the repository
--------------------
If you already have a local copy of the et-demands repository, make sure to pull the latest version from GitHub.  If you don't already have a local copy of the repository, either clone the repository locally or download a zip file of the scripts from Github.

.. note::
   For this example, it is assumed that the repository was cloned directly to the C: drive (i.e. C:\\et-demands).

Command prompt / Terminal
-------------------------
All of the CropET prep scripts should be run from the command prompt or terminal window.  In Windows, to open the command prompt, press the "windows" key and "r" or click the "Start" button, "Accessories", and then "Command Prompt".

Within the command prompt or terminal, change to the target drive if necessary::

    > C:

Then change directory to the et-demands folder::

    > cd C:\et-demands

You may need to build the common folder if it doesn't exist::

    > mkdir common

Building the Example
--------------------
.. note::
   For this example, all scripts and tools will be executed from the "example" folder.

Build the example folder if it doesn't exist::

    > mkdir example

Change directory into the example folder::

    > cd example

Cell Shapefile
--------------
DEADBEEF

For this example, ET-Demands will be run for a single HUC 8 watershed (12090105) in Texas that was extracted from the full `USGS Watershed Boundary Dataset <http://nhd.usgs.gov/wbd.html>`_ (WBD) geodatabase.

.. image:: https://cfpub.epa.gov/surf/images/hucs/12090105l.gif
   :target: https://cfpub.epa.gov/surf/huc.cfm?huc_code=12090105
.. image:: https://cfpub.epa.gov/surf/images/hucs/12090105.gif
   :target: https://cfpub.epa.gov/surf/huc.cfm?huc_code=12090105

Weather Stations
----------------
DEADBEEF

The example station is the centroid of a single 4km cell from the `University of Idaho Gridded Surface Meteorological Data <http://metdata.northwestknowledge.net/>`_ that is located in the study area.

Crop Shapefile
--------------
For this example, the crop shapefile will be built from the 2010 Cropland Data Layer (CDL) raster.

For this example the CDL raster will be downloaded directly to the project GIS folder. ::

    > python ..\et-demands\prep\download_cdl_raster.py --ini cet_prep.ini -o

If the download script doesn't work, please try downloading the `2010_30m_cdls.zip <ftp://ftp.nass.usda.gov/download/res/2010_30m_cdls.zip>`_ file directly from your browser or using a dedicated FTP program.

Clip the CDL raster to the cell shapefile extent::

    > python ..\et-demands\prep\clip_cdl_raster.py --ini cet_prep.ini -o

Convert the clipped CDL raster to a polygon, with the non-agricultural areas removed::

    > python ..\et-demands\prep\build_ag_cdl_shapefile.py --ini cet_prep.ini -o

Soils
-----
Download the pre-computed STATSGO2 shapefiles::

    > python ..\et-demands\prep\download_statsgo_shapefiles.py --ini cet_prep.ini -o

Zonal Stats
-----------
Compute the soil properties and crop acreages for each feature/polygon. ::

    > python ..\et-demands\prep\et_demands_zonal_stats.py --ini cet_prep.ini

Static Text Files
-----------------
Build the static text files from the templates in "et-demands\\static". ::

    > python ..\et-demands\prep\build_static_files.py --ini cet_prep.ini --area_threshold 10
