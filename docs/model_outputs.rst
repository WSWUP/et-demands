.. _model-outputs:

Model Outputs
=============

.. _model-outputs-refet:

RefET
-----

Outputs from the RefET module include daily or hourly reference ET (grass and alfalfa) calculated following the approach in ASCE-EWRI  `(2005) <https://ascelibrary.org/doi/book/10.1061/9780784408056>`_. 


**Timeseries Data** 

Daily Variables

	:etr_mm: Daily alfalfa reference ET [mm]
	:eto_mm: Daily grass reference ET [mm]

- Format: **.csv**

Hourly Variables

	:etr_mm: Hourly alfalfa reference ET [mm]
	:eto_mm: Hourly grass reference ET [mm]

- Format : **.csv**

|
CropETPrep
----------

The CropET Prep workflow is designed to build input files for the Crop ET module based on user specified files and needs. Many of the prepration scripts are helper functions to download and calculate zonal statistic information for the study area of interest. The preperation workflow requires users to set-up an intial study domain shapefile. The Crop ET module of ETDemands utilizes project specific static files for input and does not rely on .shp generated throughout the preparation workflow. It is recommended to build study area shpaefile and utilize the prepration workflow. Only advanced users should manually modify static files.

The descriptions below give a brief overview of the output from each preperation workflow script:

:download_cdl_raster.py: Downloads CONUS wide Crop Data Layer (CDL) raster for study crop identifcation and acreage estimates
:clip_cdl_raster.py:  Generates study area clipped version of CDL raster
:build_ag_cdl_shapefile.py: Build CDL shapefile for agricultural crop pixels only
:download_statsgo_shapefiles.py: Download statsgo soil available water capacity (AWC), Clay and Sand shapefiles
:et_demands_zonal_stats.py: Calculates zonal statistics needed to run Crop ET module
:build_static_files.py: Builds static files for Crop ET analysis

Optional (Spatially Varying Calibration Scripts):

:build_spatial_crop_params.py: Builds spatial parameter shapefiles for ET-Demands
:interpolate_spatial_crop_params.py: Interpolate spatial parameter files for ET-Demands (modifies spatial parameter shapefiles)

**File Formats**

|
CDL Shapefile

- Format: **.shp**
  
	:CDL: Cropland data layer code

|
Soils Shapefiles (3)

AWC_WTA_0to152cm_statsgo

- Format: **.shp**

	:AREASYMBOL: US
	:SPATIALVER: Version
	:MUSYM: Symbol
	:MUKEY: Key
	:AWC: Available water capacity (in/in)


Clay_WTA_0to152cm_statsgo

- Format: **.shp**

	:AREASYMBOL: US
	:SPATIALVER: Version
	:MUSYM: Symbol
	:MUKEY: Key
	:Clay: Percent clay (decimal)


Sand_WTA_0to152cm_statsgo

- Format: **.shp**

	:AREASYMBOL: US
	:SPATIALVER: Version
	:MUSYM: Symbol
	:MUKEY: Key
	:Sand: Percent sand (decimal)


Static Input Files (descriptions in previous section)

|
CropCoefs 

- Format: **.txt**
- Structure: 

  + Curve no.: 1-60
  
  + Curve type: ‘1=NCGDD: 2=%PL-EC: 3=%PL-EC+daysafter: 4=%PL-Term
  
  + Percent PL-EC or PL-TM (type 1-2-4) and/or Percent PL-EC+ days after (type 3)

  + GDD Base C
  
  + GDD Type
  
  + CGDD Planting to FC
  
  + CGDD Planting to Terminate
  
  + CGDD Planting to Terminate-alt
  
  + Comment:
  
  + Comment 2:

|
CropParams 

- Format: **.txt**
- Structure: 
	
  + Crop number and flag for crop type: negative is annual; positive in perennial
  
  + Irrigation flag: 1-yes, 2-reg., 3-required
  
  + Days after planting/green up for earliest irrigation: days
  
  + Fw: assume sprinkler
  
  + Winter surface cover class: 1-bare, 2-mulch, 3-sod
  
  + Kc max: max of value or Kcb+0.05
  
  + MAD during initial and development stage: percent
  
  + MAD during midseason and late season: percent
  
  + Initial rooting depth, m: On alfalfa, 2nd cycle, start at max
  
  + Maximum rooting depth, m: mrd
  
  + End of root growth, as a fraction of time from pl to EFC (or term if type 4)
  
  + Starting crop height, m: sch
  
  + Maximum crop height, m: mch
  
  + Crop curve number: ccn
  
  + Crop curve name: ccn
  
  + Crop curve type: 1=NCGDD, 2=%PL-EC, 3=%PL-EC,daysafter, 4=%PL-Term
  
  + Flag for means to estimate pl or gu: 1=CGDD, 2=T30, 3=date, 4 is on all the time
  
  + T30 for pl or gu or CGDD for pl or gu
  
  + Date of pl or gu (can be blank): A negative value is an offset to the prior row, pos is months (fraction)
  
  + For nCGDD based curves: Tbase: Temp Min. C (neg. For spec.)
  	
	+ CGDD for EFC: cgdd efc
	
	+ CGDD for termination: cgdd term

  + For time based curves:
	  
	+ Time for EFC: days after pl or gu
	
	+ Time for harvest (neg to extend until frost): Use as max length for CGDD crops
  
  + Killing frost temperature: C
  
  + Invoke Stress: 1-yes, 0-no, 2-yes and will wake up after severe stress (Ks<0.05)
  
  + Curve number:
  
	+ Coarse soil
	
	+ Medium soil
	
	+ Fine soil

|
ETCellsCrops 

- Format: **.txt** 
- Structure: 
	
	+ Number of Crops: XX,	Crop Number (CDL): XX…
	
	+ ET Cell ID/ET Index,	ET Cell Name,	Ref ET ID/Met Node Id,	ET Cell Irrigation (0 is off; 1 is on)


|
EToRatiosMon 

- Format: **.txt**
- Structure: 
	
	+ Met Node ID, Met Node, Month….


|
ETCellsProperties 

- Format: **.txt**
- Structure: 
	
	+ ET Cell ID, ET Cell Name, RefET MET ID, Met Latitude (DD), Met Longitude (DD), Met Elevation (feet), Area weighted average Permeability - in/hr, Area weighted average WHC - in/ft, Average soil depth - in, Hydrologic Group (A-C (A=’coarse’ B=’medium’,  Hydrologic Group  (1-3)   (1='coarse' 2='medium'), Aridity Rating (fromHuntington plus google), Ref ET Data Path
	
|
MeanCuttings 

- Format: **.txt**
- Structure: 
	
	+ ET Cell ID, ET Cell Name, Lat (DD), Number Dairy, Number Beef

|
CropET
------

ETDemands is capable of outputting both raw daily files as well as monthly, annual, and growing season summary files. All model output files are saved in .csv format. Model output files are controlled by the stat flags and folder names in the [CROP_ET] section of the model .ini file. Output files are enabled by setting the specific stat_flag = True.

- Stats flags
  
	+ daily_stats_flag = True
    
    	+ monthly_stats_flag = False
    
    	+ annual_stats_flag = False
    
    	+ growing_season_stats_flag = False


Statistic subfolders are created for each of the enabled stat flags above and will be located in the Project Folder

- ET sub-folder names

	+ daily_output_folder = daily_stats
    
    	+ monthly_output_folder = monthly_stats
    
    	+ annual_output_folder = annual_stats
    
    	+ gs_output_folder = growing_season_stats


In addition to date information, each stat file contains the following results:

:PMeto/PMetr: Input reference evapotranspiration (ET)
:ETact: Actual Crop ET including stress adjustments
:ETpot: Crop Specific Potential ET 
:ETbas: Basal evaporation component of ET
:Kc:  Crop Coefficient  
:Kcb: Basal crop coefficient
:PPT: Precipitation
:Irrigation:  Irrigation
:Runoff:  Runoff
:DPerc: Deep Percolation from the root zone
:P_rz:  Precipitation residing in the root zone
:P_eft: Effective Precipitation (precipitation residing in the root zone available for transpiration)
:NIWR:  Net Irrigation Water Requirement
:Season:  Growing Season Flag (1 = True, 0 = False)
:Cutting: Cutting Flag (applies to crops that harvested via cutting cycles (e.g. alfalfa))
  
Monthly, annual, and growing season statistics are aggregated from the daily output files according to the statistics in the list below:

:PMeto/PMetr:    sum
:ETact:          sum
:ETpot:          sum
:ETbas:          sum
:Kc:             mean
:Kcb:            mean
:PPT:           sum
:Irrigation:     sum
:Runoff:         sum
:DPerc:          sum
:P_rz:          sum
:P_eft:          sum
:NIWR:           sum
:Season:         sum
:Cutting:        sum

|
AreaET
------

|
PostProcessing
--------------

|
Timeseries Plots

Daily Timeseries

- Format: **.html** (e.g. 457500_crop_03_2018-2019.html)
- Structure:

	+ ET\ :sub:`act`\ - Actual daily ET [mm]
	
	+ ET\ :sub:`pot`\ - Potential daily ET [mm]
	
	+ ET\ :sub:`bas`\ - Basal daily ET [mm]
	
	+ PMetr_mm - Penman Monteith alfalfa reference daily ET [mm]
	
	+ K\ :sub:`c`\ - Crop coefficient [mm/mm]
	
	+ K\ :sub:`cb`\ - Basal crop coefficient [mm/mm]
	
	+ PPT - Daily precipitation [mm]
	
	+ Irrigation - Irrigation application amount [mm]

|
Daily Groupstats

- Format: **.html** (e.g. 457500_crop_03_avg.html)
- Structure:

	+ ET\ :sub:`act`\ Median - Median actual daily ET [mm]
	
	+ ET\ :sub:`act`\ 75th percentile - 75th percentile of the median actual daily ET [mm]
	
	+ ET\ :sub:`act`\ 25th percentile - 25th percentile of the median actual daily ET [mm]
	
	+ PMetr_mm Median - Median Penman Monteith alfalfa reference daily ET [mm]
	
	+ K\ :sub:`c`\ Median - median crop coefficient [mm/mm]
	
	+ K\ :sub:`c`\ 75th percentile - 75th percentile of the median crop coefficient [mm/mm]
	
	+ K\ :sub:`cb'\ Median - Median basal crop coefficient [mm/mm]
	
	+ K\ :sub:`cb`\ 75th percentile - 75th percentile of the median basal crop coefficient [mm/mm]
	
	+ K\ :sub:`cb`\ 25th percentile - 25th percentile of the median basal crop coefficient [mm/mm]
	
|
Summary Shapefiles

- Format: **.shp** (e.g. annual_crop_03.shp)
- Attribute table structure (ID may vary depending on user’s input ETZone shapefile):

	:CELL_ID: GridMET cell ID (example problem identifier)
	:LAT: Latitude [dd]
	:LON: Longitude [dd]
	:AG_ACRES: Agriculture area [acres]
	:CROP_03: Specific crop acreage (example is shown for CDL crop type 03)
	:ET_mn: Mean annual ET [mm]
	:ETact_mn: Mean annual actual ET [mm]
	:ETpot_mn: Mean annual potential ET [mm]
	:ETbas_mn: Mean annual basal ET [mm]
	:Kc_mn: Mean annual crop coefficient [mm/mm]
	:Kcb_mn: Mean annual basal crop coefficient [mm/mm]
	:PPT_mn: Mean annual precipitation [mm]
	:Irr_mn: Mean annual irrigation application amount [mm]
	:Runoff_mn: Mean annual runoff [mm]
	:DPerc_mn: Mean annual deep percolation past root zone [mm]
	:NIWR_mn: Mean annual net irrigation water requirement [mm]
	:Season_mn: Mean annual count of days within the growing season 
	:Start_mn: Mean annual growing season start day
	:End_mn: Mean annual growing season end day
	:ET_mdn: Median annual ET [mm]
	:ETact_mdn: Median annual actual ET [mm]
	:ETpot_mdn: Median annual potential ET [mm]
	:ETbas_mdn: Median annual basal ET [mm]
	:Kc_mdn: Median annual crop coefficient [mm/mm]
	:Kcb_mdn: Median annual basal crop coefficient [mm/mm]
	:PPT_mdn: Median annual precipitation [mm]
	:Irr_mdn: Median annual irrigation application amount [mm]
	:Runoff_mdn: Median annual runoff [mm]
	:DPerc_mdn: Median annual deep percolation past root zone [mm]
	:NIWR_mdn: Median annual net irrigation water requirement [mm]
	:Season_mdn: Median annual count of days within the growing season 
	:Start_mdn: Median annual growing season start day
	:End_mdn: Median annual growing season end day
	
|
Cropweighted Summary Shapefiles

- Format: **.shp** (e.g. annual_cropweighted.shp)
- Attribute table structure (ID may vary depending on user’s input ETZone shapefile):
	
	:GRIDMET_ID: gridMET ID code (6 digit code)
	:LAT: Latitude [dd]
	:LON: Longitude [dd]
	:ELEV_M: Elevation [m]
	:ELEV_FT: Elevation [ft]
	:FIPS_C: County level federal information processing system code (5 digit code)
	:STPO: State abbreviation
	:COUNTYNAME: County name
	:CNTYCATEGO: County/city category
	:STATENAME: State name
	:HUC8: Hydrologic unit code 8
	:AWC: Mean annual available water capacity(in/in)
	:CLAY: Mean annual percent clay [decimal]
	:SAND: Mean annual percent sand [decimal]
	:AWC_IN_FT: Mean annual available water capacity [in/ft]
	:HYDGRP_NUM: Hydrologic group number
	:HYDGRP: Hydrologic group
	:AG_ACRES: Agriculture area [acres]
	:CROP_XX: Specific crop type area [acres]
	:CELL_ID: Unique ID code (6 digits; matches gridMET code in example)
	:STATION_ID: Unique ID code (6 digits; matches gridMET code in example)
	:CELL_NAME: Unique ID name (6 digits; matches gridMET code in example)
	:CWETact_mn: Mean annual cropweighted actual ET [mm]
	:CWNIWR_mn: Mean annual cropweighted net irrigation water requirement [mm]
	:CWETact_md: Median annual cropweighted actual ET [mm]
	:CWNIWR_md: Median annual cropweighted net irrigation water requirement [mm]


|
Growing Season Full Summary	
	
- Format: **.csv** (e.g. growing_season_full_summary.csv)
- Structure:

	:CROP_NAME: Crop type name
	:YEAR: Year [YYYY]
	:START_DOY: Growing season start day of year
	:END_DOY: Growing season end day of year
	:START_DATE: Growing season start date
	:END_DATE: Growing season end date
	:GS_LENGTH: Growing season length [count of days]
	:CUTTING_X: Dates of 1st, 2nd, etc. cuttings

|
Growing Season Mean Annual Summary
	
- Format: **.csv** (e.g. growing season_mean_annual.csv)
- Structure:

	:STATION: Station ID code (6 digits; matches gridMET code in example)
	:CROP_NUM: Crop identifier corresponding to CDL code
	:CROP_NAME: Crop type name
	:MEAN_START_DOY: Growing season mean annual start day of year
	:MEAN_END_DOY: Growing season mean annual end day of year
	:MEAN_START_DATE: Growing season mean annual start date
	:MEAN_END_DATE: Growing season mean annual end date
	:MEAN_GS_LENGTH: Growing season mean annual length
	:MEAN_CUTTING_X: Mean annual cutting day of year


