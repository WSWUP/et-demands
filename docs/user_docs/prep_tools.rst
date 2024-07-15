CropET Prep Tools
=================

The main purpose of the CropET Prep Tools is to generate the static input text files and/or the spatially varying shapefiles for running ET-Demands.  These files specify the crop types, acreages, and agricultural soil properties for each cell.

Prep Scripts
------------
For this example, all prep scripts will be be run from the example project folder.  For example, the following command will call the download CDL script from the example project folder.  The "..\" notation indicates that you will go up one folder to the et-demands root folder, and then go into the "et-demands\\prep" sub-folder.::

    > python ..\et-demands\prep\download_cdl_raster.py

Prep Input File
---------------
The prep scripts are controlled using an `INI <https://en.wikipedia.org/wiki/INI_file>`_ file.  The prep INI file shares many of the same elements as the CropET INI file, and a CropET INI file could be used for the prep scripts with only minimal additions.

An example/template prep INI file can be found in the "et-demands\\prep" folder.

The "--ini" parameter must be set when calling each of the prep scripts.  The INI file path can be absolute or relative to the script location.

Command Line Parameters
----------------------
To see what arguments are available for each script, pass the "-h" argument to the script.::

    > python ..\et-demands\prep\download_cdl_raster.py -h

The following arguments are available in all of the scripts

-i, --ini
  Prep INI file path
-d, --debug
  Enable debug level logging
-o, --overwrite
  Overwrite existing files

.. note::
   Note, many of the scripts will not do anything if the output file already exists and the "--overwrite" argument is not set.

CDL Crop Shapefile
------------------
If the user does not have an existing crop type shapefile, the following three scripts can be used to generate one from Cropland Data Layer (CDL) rasters.

download_cdl_raster.py
``````````````````````
This tool will attempt to download the CDL raster.  The download folder and target year must be set in the prep INI file using the "cdl_folder" and "cdl_year" parameters.

The CDL rasters can also be downloaded directly from the `USDA NASS FTP <ftp://ftp.nass.usda.gov/download/res/2010_30m_cdls.zip>`_ using your browser or dedicated FTP program.

clip_cdl_raster.py
``````````````````
This tool will clip the CONUS wide CDL raster to the cells shapefile extent.

It is not mandatory that the clip CDL raster tool be run.  Typically this would be run to reduce the file size of the CDL raster or make it load faster in a GIS.  Since the CDL rasters can take a long time to download and take up a lot of space, it might be more efficient to download the CDL raster one time to a "common" folder (by changing the "cdl_folder" parameter in the INI), then making clipped copies for each project.

The CDL clip utility will always try to write a clipped copy of the CDL raster into the project GIS folder.  If the "cdl_folder" parameter in the INI folder is already set to the project GIS folder this will overwrite the existing CDL raster.

build_ag_cdl_shapefile.py
`````````````````````````
This tool will construct the crop shapefile from the CDL raster.

Non-agricultural classes will be excluded from the final crop shapefile.  The agricultural CDL classes are set in the INI file using the "cdl_crops" parameter.  These values may need to be adjusted depending on the CDL year and the project.  The following non-irrigated CDL classes may need to be included.

-  Crop 61 is fallow/idle and is not included as an agricultural class
-  Crop 176 is Grassland/Pasture in the new national CDL rasters
-  Crop 181 was Pasture/Hay in the old state CDL rasters
-  Crop 182 was Cultivated Crop in the old state CDL rasters

Soils Shapefiles
----------------

download_statsgo_shapefiles.py
``````````````````````````````
This tool will attempt to download pre-computed CONUS wide STATSGO2 shapefiles of AWC, percent clay, and percent sand.

Zonal Stats
-----------

et_demands_zonal_stats.py
`````````````````````````
This tool is used to average the crop and soil data to the ET cells/units.  The output field names and sub-folder paths are all hardcoded in the script.  The data will be written directly to the cells shapefile specified in the prep INI file.

build_static_files.py
`````````````````````
This tool is used to populate the static text files based on the zonal stats data in the cells shapefile.  The static text files for each project are built from the templates in "et-demands\\static".

Along with the parameters listed above, there are other hardcoded parameters in the script that may eventually be read from the INI file.  These include the station elevation units (feet), soil_depth (60), aridity (50), permeability (-999), and the number of supported crops (86 - assuming the USBR mod crosswalk is used).

build_spatial_crop_params.py
````````````````````````````


Unused scripts
--------------
The _arcpy.py, _gdal_common.py, _ogr2ogr.py and _util.py python file are not run by the user.  These scripts are called by the other scripts.

Additional unused scripts can be found in the prep subfolders.  These will eventually be moved or removed.
