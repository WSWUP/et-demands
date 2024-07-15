.. _data-sources:

Data Sources
============

RefET
-----
Meteorological data can be obtained from numerous sources.

.. _data-sources-ag-met:

Agricultural Weather Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| Network                                                                                           | Supporting Organization                                   | Coverage                                                                      |
+===================================================================================================+===========================================================+===============================================================================+
| `CoAgMet  <http://coagmet.colostate.edu/>`_                                                       | Colorado State University                                 | Colorado                                                                      |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `AZMET  <https://cals.arizona.edu/azmet/>`_                                                       | University of Arizona                                     | Arizona                                                                       |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `AgWeatherNet <https://weather.wsu.edu/>`_                                                        | Washington State University                               | Washington                                                                    |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `ZiaMet <https://weather.nmsu.edu/ziamet/>`_                                                      | New Mexico State University                               | New Mexico                                                                    |

+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `AgriMet-GP <https://www.usbr.gov/gp/agrimet/index.html>`_                                        | Bureau of Reclamation Great Plains Regional Office        | Montana                                                                       |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `AgriMet-PN <https://www.usbr.gov/pn/agrimet/>`_                                                  | Bureau of Reclamation Pacific Northwest Regional Office   | Washington, Oregon, California (northern), Nevada, Utah, Idaho, Montana       |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `CIMIS <https://cimis.water.ca.gov/>`_                                                            | California Department of Water Resources                  | California                                                                    |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `NICE Net <https://nicenet.dri.edu/>`_                                                            | Desert Research Institute                                 | Nevada                                                                        |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+
| `West Texas Mesonet <http://www.depts.ttu.edu/nwi/research/facilities/wtm/index.php>`_            | Texas Tech University                                     | Texas (western)                                                               |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------------------------------------+

Other Weather Networks
~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| Network                                                                                           | Supporting Organization                                   | Coverage                                        |
+===================================================================================================+===========================================================+=================================================+
| `Montana Mesonet <http://climate.umt.edu/mesonet/>`_                                              | University of Montana                                     | Montana                                         |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `TexMesonet <https://www.texmesonet.org/>`_                                                       | Texas Water Development Board                             | Texas                                           |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `Lower Colorado River Authority Hydromet <https://hydromet.lcra.org/>`_                           | Lower Colorado River Authority (TX)                       | Texas (Colorado River Basin)                    |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `Kansas Mesonet <http://mesonet.k-state.edu/>`_                                                   | Kansas State University                                   | Kansas                                          |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `Nebraska Mesonet <https://mesonet.unl.edu/>`_                                                    | University of Nebraska, Lincoln                           | Nebraska                                        |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `Mesonet <http://mesonet.org/>`_                                                                  | University of Oklahoma, Oklahoma State University         | Oklahoma                                        |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+

Multi-Network Sources
~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| Source                                                                                            | Supporting Organization                                   | Coverage                                        |
+===================================================================================================+===========================================================+=================================================+
| `MesoWest and SynopticLabs <https://synopticlabs.org/>`_                                          | University of Utah                                        | United States                                   |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+
| `Integrated Surface Database <https://www.ncdc.noaa.gov/isd/>`_                                   | NOAA National Center for Environmental Information (NCEI) | Global                                          |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------+-------------------------------------------------+

CropET
------

Study Area
~~~~~~~~~~

HUC8 and HUC10 features can be extracted from the full - [USGS Watershed Boundary Dataset](http://nhd.usgs.gov/wbd.html) (WBD) geodatabase. A subset of the WBD HUC polygons can downloaded using the [USDA Geospatial Data Gateway](https://gdg.sc.egov.usda.gov/) or the full dataset can be downloaded using the [USGS FTP](ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/WBD/).

County features can be downloaded from the [USDA Geospatial Data Gateway](https://gdg.sc.egov.usda.gov/). For the zonal stats prep tool to work, the shapefile must have a field called "COUNTYNAME".  Other county features (such as the [US Census Cartographic Boundary Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-cart-boundary.html) could eventually be supported (or the name field could be manually changed to COUNTYNAME).

The GRIDMET grid cells can be constructed how?


Crop Type Data
~~~~~~~~~~~~~~

Cropland Data Layer (CDL)
^^^^^^^^^^^^^^^^^^^^^^^^^

The Cropland Data Layer (CDL) is a product of the USDA National Agricultural Statistics Service (NASS) with the mission
“to provide timely, accurate and useful statistics in service to U.S. agriculture”
`(Boryan et al 2011) <https://doi.org/10.1080/10106049.2011.562309>`_. The CDL is a crop-specific
land cover classification product of more than :ref:`100 crop categories <cdl-crop-types>` grown in
the United States developed using remote sensing. The CDL can be downloaded using NASS’s
`CropScape tool <https://nassgeodata.gmu.edu/CropScape/>`_. Updates about and references for the
CDL can be found at `NASS <https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php>`_.
A version of the CDL has been released annually from 1994-Present.

## Soils Data
~~~~~~~~~~~~~


The average agricultural area available water capacity (AWC) and hydrologic soils group are needed for each ET cell/unit.  The hydrologic soils group can be estimated based on the percent sand and clay for each ET cell/unit.

The AWC, percent clay, and percent sand data cannot (currently) be directly downloaded.
The easiest way to obtain these soils data is to download the
[STATSGO] (http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053629)
database for the target state(s) using the [USDA Geospatial Data Gateway](https://gdg.sc.egov.usda.gov/).
Shapefiles of the soil properties can be extracted using the [NRCS Soil Data Viewer](http://www.nrcs.usda.gov/wps/portal/nrcs/detailfull/soils/home/?cid=nrcs142p2_053620)
The [SURGO](http://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/geo/?cid=nrcs142p2_053627)
databases can also be used, but these typically cover a smaller area and may have areas of missing data.
 It may also be possible to used the gridded SSRUGO data, but this has not been tested.

*Add additional details about which options were used in the Soil Data Viewer*

To use the soil prep tools, the soils data must be provided as separate shapefiles for each product.  The names of the soil shapefiles are hard coded in the rasterize_soil_polygons.py script as "{}_WTA_0to152cm_statsgo.shp", where {} can be "AWC", "Clay", or "Sand" (see [Model Structure](structure.md)).  For each shapefile, the value field name is hardcoded as the upper case of the property (i.e. "AWC", "CLAY", or "SAND").
