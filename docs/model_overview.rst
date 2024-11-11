Model Overview
==============

The ETDemands model consists of three modules that help develop daily estimates of crop irrigation water requirements. The RefET module calculates hourly or daily reference ET from meteorological data following the procedure in ASCE-EWRI `(2005) <https://ascelibrary.org/doi/book/10.1061/9780784408056>`_. At the core of the ETDemands model is the FAO-56 dual crop coefficient model `(Allen et al., 1998) <http://www.fao.org/3/X0490E/X0490E00.htm>`_, or CropET module, which is used to calculate crop evapotranspiration from meteorological data and crop-specific coefficient curves. The actual ET for various crop types is calculated in the ETDemands model as a function of the two primary crop coefficients Kcb and Ke, and a crop stress coefficient Ks for all crop types as follows:


  ET\ :sub:`c` = (K\ :sub:`s`\  x K\ :sub:`cb`\  + K\ :sub:`e`)ET\ :sub:`o`\

where

  ET\ :sub:`c`\  = crop evapotranspiration 

  K\ :sub:`c`\  = crop stress coefficient 

  K\ :sub:`cb`\  = basal crop coefficient 

  K\ :sub:`e`\  = coefficient representing bare soil evaporation 

  ET\ :sub:`o`\  = reference crop evapotranspiration from a grass reference surface 
|
Daily Kcb values over season, commonly referred to as the crop coefficient curve, represent impacts on crop ET from changes in vegetation cover, leaf area, and vegetation height due to changes in crop phenology. These properties vary during the growing season and can vary from year to year depending on the start, duration, and termination of the growing season, all of which are dependent on temperature. Ks ranges from 0 to 1, where 1 represents a condition of no water stress. Ks is also dimensionless. A daily soil water balance of the simulated effective root zone is a key component of the ETDemands model to calculate Ks. In most cases, the ETDemands model was operated assuming full water supplies when computing ETc and the net irrigation water requirement (NIWR), so that Ks was generally 1.0. However, Ks can be less than 1 during the dormant season (winter) if precipitation is low.

Values for Kcb for a given crop generally vary seasonally and annually according to simulated plant phenology. Plant phenology is, in turn, impacted by solar radiation, temperature, precipitation, and agricultural practices. The ability to simulate year-to-year variation in the timing of green-up or planting, and timing of effective full cover, harvest, and termination, is necessary for integrating the effects of temperature on growing-season length and crop growth and development, especially under changing climate scenarios. Seasonal and year-to-year changes in vegetation cover and maturation are simulated in ETDemands for each crop Kcb as a crop-specific function of air temperature, based on cumulative growing degree days (GDD). After planting of annuals or the emergence of perennials, the value of Kcb tends to gradually increase with increasing
temperature until the crop reaches full cover. Once this happens and throughout the middle stage of the growing season, the Kcb value is generally constant, or is reduced to simulate cuttings and harvest. From the middle stage to the end of the growing season the Kcb value reduces to simulate senescence. A full description
on how GDD is calculated in ETDemands is provided in chapter 4.

The NIWR or depth is calculated in ETDemands as ETc minus effective precipitation residing in the root zone (Prz), and represents the amount of additional water that the crop would evapotranspire in excess of precipitation residing in the root zone. The NIWR is synonymous with the terms irrigation demand, net consumptive use, and precipitation deficit. Prz is calculated as a function of daily precipitation (from the climate data set), antecedent soil moisture prior to a precipitation event, deep percolation of precipitation, and precipitation runoff. 
