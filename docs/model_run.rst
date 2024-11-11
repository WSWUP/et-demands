Running the Model
=================

The ETDemands model should be run from the windows command prompt (or Linux terminal) so that configuration files can be passed as an argument directly to the script. Ensure that both of the example template .INI files are populated with the user’s relative file paths to folder locations for the current project, static text file specifications, and time series data specifications. The .INI files are currently set up to run the example workflows, however, the user can change these to meet their needs for their own analysis as long as all of the necessary changes to the .INI files and relative paths are made. The following command line call examples are set up as if the user has already navigated to the location of their local .INI file(s) being passed as an argument (e.g. using the command “cd” followed by the local path to the .INI file(s) location is a common way to navigate through the user’s file system). There are optional arguments provided at the bottom of each section that can be used if wanted.

RefET
-----
The RefET moduel is designed to prepare and estimate reference ET from station meterological data based the on the ASCE Standardized Reference ET approach. Additional estimatation methods are supported for datasets required input variables. The RefET module provides gap filling and data QA/QC and filtering. Users should refer to the RefET desciption section for more detail. Refer to the ret_template.ini included in the et-demands\refET\bin folder for additional information on formatting and control variables. 

Example command to run refet moduel:
  ``> python  run_ret.py -i my_ret.ini``

Arguments for run_ret.py script are:
  -h, --help  Show the help message then exit
  -i, --ini  Initialization file path
  -d, --debug  Save debug level comments to debug.txt file
  -m, --metid  Specify a particular single Met Node ID to run
  -v, --verbose  Display information level comments
  -mp, --multiprocessing  num

CropETPrep
----------

The CropET Prep workflow is designed to help users assemble ancillary data and create static files for input and running of the CropET module. Ancillary data includes crop type and coverage data as well as soil data including available water capacity (AWC) and percent clay and sand. By default, the preperation workflow utilizes the USDA Cropland Data Layer (CDL) and the NRCS STATSGO soils dataset. Use of other crop and soil datasets is possible, but should only be implemented by advanced users. Scripts run throughout the prep workflow utilized the prep .ini file to for directroy and path information as well as the crop field names and crosswalk information. See the template prep.ini file included in the et-demands\prep folder for specific formating and variable options. 

An example preperation workflow is shown below: 

Run Prep
  Scripts A1-A3 are only needed if user does not have a crop shapefile. The CDL clip script is only needed if storage space is an issue.

A1: Download CDL Raster

  ``> python ..\..\et-demands\prep\download_cdl_raster.py --ini example_prep.ini``

A2: Clip CDL Raster - Only needed to save storage space

  ``> python ..\..\et-demands\prep\clip_cdl_raster.py --ini example_prep.ini``

A3: Build CDL Shapefile

  ``> python ..\..\et-demands\prep\build_ag_cdl_shapefile.py --ini example_prep.ini``

1: Download Soils Files

  ``> python D:\et-demands\et-demands\prep\download_statsgo_shapefiles.py --ini example_prep.ini``

2: Calculate Zonal Statistics

  ``> python D:\et-demands\et-demands\prep\et_demands_zonal_stats.py --ini example_prep.ini``

NOTE: Crop Crosswalk File: prep/cdl_crosswalk_default.csv
By default, ETDemands includes a crop crosswalk file to convert CDL crop codes to their corresponding
ETDemands crop code. This file is specified using the 'crosswalk_path' variable in the prep .ini file.
When not using the USDA CDL layers, users should develop a custom crosswalk file that relates crop codes 
in their custom file to crop layer. The shapefile field name containing crop codes is also specified in the prep_ini. 

3: Build Static Files

  ``> python D:\et-demands\et-demands\prep\build_static_files.py --ini example_prep.ini --acres 0 -o``

Steps 4 and 4b are only needed if running spatially varying cablibration mode

4: Build Spatial Crop Parameter Files 
  
  ``> python C:\et-demands\et-demands\prep\build_spatial_crop_params.py --ini example_prep.ini --area 0``

4b:  Interpolate spatial crop params from preliminary calibration .shp

  ``> python ..\et-demands\et-demands\prep\interpolate_spatial_crop_params.py --ini example_prep.ini``
  
NOTE: Interpolate spatial crop parameters is utilized for large study areas that wish to interpolate 
crop parameter information from a few calibration cells to the entire study domain.

Static Files
------------

The text files in the "et-demands\\static" folder are templates of the static input files needed for running ET-Demands.  The files are tab delimited and the structure and naming mimic the Excel file used to control the VB version of the ET-Demands model.

The template files should not be modified directly.  Instead, the prep tools workflow should be used to automatically populate the files or the files can copied to a "project\\static" folder and the values set manually.

Crop Coefficients
^^^^^^^^^^^^^^^^^

The crop coefficient .txt files contain summary basal crop coefficient curves for all crops available within ET-Demands. Both alfalfa reference (ETr; CropCoefs_etr.txt) and grass reference (ETo; CropCoefs_eto.txt) versions are available. Users specify the crop coefficient file using the 'crop_coefs_name' parameter in the model .ini file.

Crop Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

The crop parameters contains default control paramaters for each crop within ET-Demands. This includes data to specify curve number, irrigation type, winter surface class, and many other soil and growing season control variables. The default values included with each crop are general starting parameters and should be adjusting according to specific study area and needs. 

During calibration, users will modify and adjust growth parameters such as T30, CGDD for EFC, CGDD for termination, and Killing Frost temperature to control growiing season time and curve shape. See the model calibration documentation section for more information.

ET Cell Crops
^^^^^^^^^^^^^^^^^^^^^^^^^

ETCellsCrops.txt is generatred during the perparation steps and includes a table of all unique ETCell/Crop combinations to be included in the ET-Demands analysis. ETCell/Crop combinations are denoted using a binary style classification system with 1=True and 0=False.

ET Cell Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

ETCellsPRoperties.txt contains ETCell information related to location, permeadbility, soil depth, hydrologic group, and ariditiy rating.

ET Monthly Ratios
^^^^^^^^^^^^^^^^^^^^^^^^^

When using gridded climate products, reference ET data is not always representative of irrigation conditions due to microclimate condition effects. ET-Demands allows users to apply monthly scaling factors to input RefET datasets in order to account for model bias related to scale and irrigtation practices. Users should modify either EToRatiosMon.txt (grass reference) or ETrRatiosMon.txt (alfalfa reference) according to their input dataset. Users specify the ratio file using teh crop_coefs_name parameter in the model .ini file. By default, the static file is orginially built with ratios equal to 1 (i.e. no scaling). Users can manualy adjust scaling factors or apply more advanced workflows to identify bias correction factors (https://github.com/WSWUP/gridwxcomp).

Mean Cuttings
^^^^^^^^^^^^^^^^^^^^^^^^^

For crops that experience cutting cycles (.e.g Alfalfa Hay), ET-Demands allows users to optimize the number of cuttings based on study area and local practices. The MeanCuttings.txt file is initially populated with temporary cutting estimates. After a calibration run. User can repopulate the cutting numbers based on output from ET-Demands. Iteration may be required to optimize cutting numbers and timing. 


CropET
------

The CropET module is the core of the ETDemands model. The CropET modules takes input weather, soil, and crop data to estimate consumptive use and net irrigation water requirements for each unique ETZone/crop combination. At this point, users should have run through the neccesary steps in the prep workflow and have generated project specific static files. Crop ET is controlled using a project specific CET .ini file. See the template cet.ini file included in the et-demands\cropET\bin folder for specific formating and variable options. 

The Crop ET module is run using the run_cet.py script. An example command is shown below. Users can include -h argument to see various input argumnet options. -i (-ini) and -b (-bin) are required arguments. 

Run CET
``> python C:\et-demands\et-demands\cropET\bin\run_cet.py -i example.ini -b C:\et-demands\et-demands\cropET\bin -h``

optional arguments:
  -h, --help            show this help message and exit
  -i PATH, --ini PATH   Input file (default: None)
  -b DIR, --bin DIR     Source code directory path (default: None)
  -d, --debug           Save debug level comments to debug.txt (default:
                        False)
  -c etcid_to_run, --etcid etcid_to_run
                        User specified et cell id to run (default: ALL)
  -v, --verbose         Print info level comments (default: False)
  -mp, --multiprocessing    Number of processers to use (default: 1)
  --cal                 Display mean annual start/end dates to screen
                        (default: False)


AreaET
------
The AreaET module is currently under development.

The AreaET module converts crop evapotranspiration and net irrigation water requirement rates estimates output by the CropET module to volume and flow estaimtes based on user supplied acreage information. Aggregations of daily, monthly, annual, and growing seaon volumes and flows are output along with crop percentages and ratios for each study cell. Users can define start and end dates to analyze specific time periods.

The AreaET module is run similar to the CropET module using command line calls with AreaET specific .ini files. Example commands are shown below: 

``> python run_aet.py -i my_aet.ini``

Arguments for run_aet.py script are:
    -h, --help  Show the help message then exit
    -i, --ini  Initialization file path
    -d, --debug  Save debug level comments to debug.txt file
    -c CellID, --metid  CellID Specify a particular single ET Cell to run
    -v, --verbose  Display information level comments
    -mp, --multiprocessing  Number of processers to use (default: 1)


PostProcessing
--------------

ETDemands offers some post-processing tools (Timeseries tools, Shapefile tools, etc.) that may be used to analyze the model outputs. More detailed descriptions of these tools and optional command line arguments are available in the Analysis Tools section of the Read the Docs. The following command line calls will use the output stats to produce timeseries plots, summary shapefiles, and other supplemental information.

To develop timeseries plots of Crop ET-Demands parameters ET, ETo, Kc, growing season, irrigation, precipitation, and NIWR, the following command line call can be used:

  ``> python ..\..\et-demands\tools\plot_crop_daily_timeseries.py --ini huc_example_cet.ini``


To develop timeseries plots of average Crop ET-Demands parameters ET, ETo, Kc, growing season, irrigation, precipitation, and NIWR, the following command line call can be used:

  ``> python ..\..\et-demands\tools\plot_crop_daily_groupstats.py --ini huc_example_cet.ini``


To convert the daily output files into crop specific summary shapefiles the following command line call can be used:
  
  ``> python ..\..\et-demands\tools\summary_shapefiles_gpd.py --ini huc_example_cet.ini``


To convert the daily output files into crop weighted summary shapefiles the following command line call can be used:
  
  ``> python ..\..\et-demands\tools\cropweighted_shapefiles_gpd.py --ini huc_example_cet.ini``
 

The final post-processing command line call can be used to summarize growing season length and cutting information for each ETZone/crop combination:

  ``> python ..\..\et-demands\tools\compute_growing_season.py --ini huc_example_cet.ini``

