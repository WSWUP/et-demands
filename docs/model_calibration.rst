.. _model-calibration:

Model Calibration
=================

.. _model-calibration-refet:

RefET
------

.. _model-calibration-refet-tr:

Thornton and Running Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
THIS SECTION IS CURRENTLY UNDER DEVELOPMET. 

Thorton and Running coefficients are set one of three ways within the RefET Module.
 1) Station Metadata File: tr_b0, tr_b1, tr_b2 columns
 2) ret .ini file [INMET] section: TR_b0, TR_B1, and TR_b2 varibles
 3) Defualt values coded within RefET python scripts
 
These are current hard-coded defaults if not found by other options. Should they be set to original 'universal' TR values?
# These match the WWCRA Klamath Values. 
# Thorton and Running Solar Radiation Estimation Coefficients
try: self.input_met['TR_b0'] = config.getfloat(input_met_sec, 'TR_b0')
  except: self.input_met['TR_b0'] = 0.0307075712855
try: self.input_met['TR_b1'] = config.getfloat(input_met_sec, 'TR_b1')
  except: self.input_met['TR_b1'] = 0.1960418743712
try: self.input_met['TR_b2'] = config.getfloat(input_met_sec, 'TR_b2')
  except: self.input_met['TR_b2'] = -0.2454592897026

Default Thornton and Running calibration coefficients are utlized if not specified by the user. 

Thornton and Running coefficients can be optimized to specific station data using Monte Carlo uniform random search techniques.
Allen (2009; http://www.kimberly.uidaho.edu/ETIdaho/ETIdaho_Report_April_2007_with_supplement.pdf) and Huntington (2010; http://water.nv.gov/mapping/et/Docs/Evapotranspiration_and_Net_Irrigation_Requirements_for_Nevada_Compiled.pdf) applied these technique to optimize coefficients during consumptive use studies in Idaho and Nevada.
More recently, https://www.usbr.gov/watersmart/baseline/docs/irrigationdemand/irrigationdemands.pdf (see report pages; 36-37).

Additional infromation on Thornton and Running calibration will be added during future releases. 


.. _model-calibration-cropet:

CropET
------
ETDemands utlizes crop specific parameters to determine crop growth and timing including growing season start and end dates, crop development rates, effective full cover, harvest timings, and killing frosts. Initial values for each of the control parameters are provided, however, users should customize these values for their input meterological data source and study area. Inherent bias in temperature data can lead to incorrect timing of the season start, effective full cover, and termination dates. A detailed decsiption of the various crop parameters can be found here: http://data.kimberly.uidaho.edu/ETIdaho/ETIdaho_Report_April_2007_with_supplement.pdf

The initial control parameters are located in the CropParams.txt file contained in the static folder. Note: Users should NOT modify the statics files found in the etemands model folder. Running the initial prep steps will build project specific static files that can be changed accordingly. ETDemands can be run with either spatially consistent or spatially varying calibration parameters. This mode is set in the model INI file via the spatial_cal_flag and spatial_cal_folder inputs.

  ## Spatially varying calibration
    spatial_cal_flag = True
    
    spatial_cal_folder= C:\\example\\calibration

Spatially Consistent Calibration Mode (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spatially consistent calibration mode applies the same crop parameter information to every ETZone in your model. The parameters are read directly from your project specific CropParams.txt file. This mode should be utilized for when the study zones are located in relatively simliar environments and your supplied weather data is consistent from zone to zone.

Spatially Varying Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spatially varying calibration mode applies different crop parameter information to every ETZone in your model. These varying parameters are orgnaized in crop specific .shp. To create the crop specific calibration .shp, run the build_spatial_crop_params.py script:
python ..\et-demands\et-demands\prep\build_spatial_crop_params.py --ini example.ini 

Running this script creates and populates the calibration folder with crop specific calibration .shp. Each .shp contains an attribute table with rows for each ETZone that contains that crop. The columns or fields of the attribute table are initially populated with the default crop parameters from the CropParams.txt. Users can now assign specific crop paramters from each ETZone and crop. 

Calibration Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calibrating the ETDemands model is an iterative process that requires users to perform complete runs followed by adjustments to the CropParam.txt or crop paramater .shp. The user re-runs the model after adjusting the crop parameter until the they are satistified with the timing and shape of the crop development curves. Literature review should be performed to determine typical planting, effective full cover, and harvest dates. Throughout the calibration process, users can limit the cells and/or crops being processed by the model by setting the 'crop_test_list' and 'cell_test_list' variblies in the [CROP ET] section of the model .ini file. For example, setting the parameters below in the model .ini will run 'crop 07' in 'cell 529813' and skip all other cells/crop combinations. 

  crop_test_list = 07
  cell_test_list = 529813
  
  
An understanding of the growing practices of the area is required for proper calibration, specifically typical planting (or greenup) dates and harvest (senescence) dates are needed. Data specific to the area is best, but general start and stop dates are also available from the U.S. Department of Agriculture (USDA). The most recent nation-wide
publication is entitled Field Crops: Usual Planting and harvesting Dates, October 2010 and is (as of this
writing) available at: (https://www.nass.usda.gov/Publications/Todays_Reports/reports/fcdate10.pdf).
Once start and stop dates are secured for the area of interest, follow this iterative process:

1) Calibrate one crop at a time. A single crop can be specified by setting a crop_test_list parameter in the
model .ini file. 

2) For a single crop, execute the crop ET model for a single ET Cell using the cell_test_list .ini paramter.
Choose a representative single ET Cell that will act as the calibration cell.

3) Execute the model run and review daily output. Growing season postprocessing script can be run to generate
summary data showing the start and stop dates for each year of execution, the second showing the average start
and stop date for the entire run. Open the annual average file, titled “growing_season_mean_annual.csv” and
look at the “MEAN_START_DATE” and “MEAN_END_DATE” fields. 


T4) Adjust the parameters for start and stop in the "CropParam.txt" or crop paramater .shp as needed.

  a) Row 21 specifies the method used to estimate planting/greenup: 1=Growing Degree Days
  (GDD), 2= 30-day average temperature (T30), 3=User specified date, 4=Always on

  b) Row 22 specifies the number of GDD or the T30 temperature. If the type displayed in row
  21 is 3, enter the start month in row 23. If the MEAN_START_DATE (from step 3 above) is
  too early, increase the value in row 22.

  c) Row 26 specifies the GDD required for termination. Note that this is the number of GDD
  after planting/greenup. If the MEAN_END_DATE (from step 3 above) is too early, increase
  the value in row 26.

1) Additionally, if the user specifies, an absolute maximum growing season length can be
entered in row 29.

2) For crops that grow until a killing frost, row 30 is used. Enter the temperature required
to end the crop growing season.

Once the start and stop dates are achieved, it is time to adjust the crop kc curve to match example
curves. This calibrates the planting/greenup to Effective Full Cover (EFC)/maturity portion of the curve.
Once again, this is an iterative process that relies on the judgement of the user to accomplish. Example
curves are available online or can be secured from the Reclamation report titled West-Wide Climate
Risk Assessment: Irrigation Demand and Reservoir Evaporation Projections. Example Kc curves can be found here:
http://data.kimberly.uidaho.edu/ETIdaho/ETIdaho_Report_April_2007_with_supplement.pdf
Comparing the resulting kc curves to the display curves allows the user to change maturity times for the
crops, thus dialing in the calibration. Steps for doing this are below:

5) Compare the resulting kc curve to a known curve for the crop of interest. Look specifically at the
time from planting/greenup to full maturity (the top of the curve) relative to the overall length
of the life cycle.

6) Adjust the values in row 25 (for GDD and T30 based crops or row 28 (for date based crops) of
the “CropParams” tab of the Meta data workbook to adjust the curve peak to better match the
example crop curves.

Once a curve for the crop looks good (on average, no two years growing conditions will be exactly the
same), choose a different crop on the “ETCellsCrops” tab and repeat for all crops to be simulated.
After each adjustment of the “CropParams” data, re-run crop ET model to get updated simulation
results. Unless meteorology is changed, the Reference ET model does not need to be re-run between
calibration steps. Typically, start and stop dates are calibrated first (steps 2-4 above), re-executing the
ET model between each adjustment of the “CropParams” data. Once the start and stop times are
calibrated, move on to the crop curve calibration, executing steps 2, 5 and 6, again re-running the ET
model between each parameter adjustment.
  


Interpolation of Spatial Crop Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For model runs with a large number of ETZones, users may prefer to spatially interpolate crop parameters from a set of preliminary calibration zones. To set-up and interpolate crop parameters from a set of preliminary zones, users should first run the ETDemands model with a subset of cells that are representative of spatial extent and crops selection throughout the larger study area. Once you've finalized the crop parameter .shp for the subset run, you will copy the crop specific .shp to the "preliminary_calibration" folder in the calibration folder of your complete model. Spatial interpolation will only occur for crops with crop parameter .shp located in teh preliminary calibration folder. Note that if the "preliminary_calibration" folder will need to be created if it doesn't already exist.

Users should run the build_spatial_crop_params.py script to create the calibration folder and crop specific .shp before building the preliminary calibration folder.  

python ..\et-demands\et-demands\prep\interpolate_spatial_crop_params.py --ini UC_2018.ini

Notes on Model Calibration:
Calibration of the ETDemands Models requires both time and experience. Users are encouraged to experiement with smaller models (limited cell/crop combinations) to build familiarity with each of the paramters before attempting calibartion over large areas with multiple crops. Each crop utilizes specific information related to its curve type and growth cycle. Curve type assignments for each crop are found within the CropParams.txt along with inital parameter values.   

Crops are assigned one of four differenct curve types: 1=NCGDD, 2=%PL-EC, 3=%PL-EC,daysafter, 4=%PL-Term
  
  - 1 = normalized cumulative growing degree days (NCGDD)
  - 2 = percent of time from planting (or greenup) to effective full cover, applied all season
  - 3 = percent of time from planting (or greenup) to effective full cover, then days after effective full cover
  - 4 = percent of time from planting (or greenup) until termination 

In addition to curve type, each crop also recieves a flag for estimating planting or greenup: 1=CGDD, 2=T30, 3=date, 4 is on all the time

  - 1 = Indicates that cumulative growing degree days from January is used
  - 2 = Indicates that 30 day mean air temperature is used
  - 3 = Indicates a specific date
  - 4 = Crop growth is always on

Depending on the assignments above, the crop will utilize different values to determine the start, greenup, effective full cover, harvest, and termination dates. 

Tips:
In general, it is easier to make small changes to one crop/parameter combination at a time. Large changes to multiple paramters can be difficult to track. Utilize the crop and cell test list varibles to limit your model run and speed up output results. Examining both the daily time series plots and summary .shp created with the postprocessing "tools" scripts will help identify problematic crops/cells. 



