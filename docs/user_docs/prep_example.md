# ET-Demands Model Example

## Prep Tools


### Input Data
A set of input data to run this example is packaged with the model. This input data can also be used as a template in developing input data for other applications. Details about the set of input data including the file format and structure are found below. More details about the required input data can be found on the [CropET Input Data](data.md) page and details about the standard datasets used can be found on the [CropET Standard Data](standard_data.md) page.
#### Weather Data


###### Timeseries Data
The example station is the centroid of a single 4km cell from the [University of Idaho Gridded Surface Meteorological Data](http://metdata.northwestknowledge.net/) located in the study area.


* File: **214033E2.dat**

* Format: .dat

* Structure:

| Date   | TMax   | TMin   | Precip   | Snow     | SDep | EstRs   | EsWind | EsTDew | Penm48   | PreTay   | ASCEr    | ASCEg    | 85Harg   |
| -------| ------ | ------ | -------- | -------- | ---- | ------- | ------ | ------ | -------- | -------- | -------- | -------- | -------- |
| Units  | [C]    | [C]    | [In*100] | [In*100] | [In] | [MJ/m2] | [m/s]  | [C]    | [mm/day] | [mm/day] | [mm/day] | [mm/day] | [mm/day] |

###### Location Shapefile

* File: **nldas_4km_dd_pts.shp**

* Format: .shp

* Attribute Table Structure:

| NLDAS_ID | LAT   | LON     | HUC8     | ELEV_M | ELEV_FT |
|----------|-------|---------|----------|--------|---------|
| 214033   | 31.48 | -100.23 | 12090105 | 553.30 | 1815.28 |

#### Study Area
For this example, the study area is a single HUC 8 watershed (12090105) in Texas that was extracted from the full [USGS Watershed Boundary Dataset](http://nhd.usgs.gov/wbd.html).

* File: **wdbhu8_albers.shp**

* Format: .shp

* Attribute Table Structure:

| OBJECTID | AreaAcres | AreaSqKm | States | HUC8     | Name   |
| -------- | --------- | ---------| -------| -------- | ------ |
| 861      | 789046    | 3193.16  | TX     | 12090105 | Concho |

#### Crop Type Data
Crop type data is obtained from the Cropland Data Layer. A script below

#### Soils Data
The soils data used were
### Example Steps

#### Step 1: Getting Started
Follow the [getting started] instructions to download or clone a copy of the latest version of the model.

#### Step 2: Open a Terminal Window


#### Step 3:

#### Step XX: Download and Prep the CDL
Download the CONUS CDL raster. The CONUS CDL rasters should be downloaded to the "common" folder so that they can be used for other projects. For this example we will be using the 2010 CDL raster.

```
    python ..\et-demands\prep\download_cdl_raster.py --cdl ..\common\cdl --years 2010
```

If the download script doesn't work, please try downloading the [2010_30m_cdls.zip](ftp://ftp.nass.usda.gov/download/res/2010_30m_cdls.zip) file directly from your browser or using a dedicated FTP program.


Clip the CDL raster to the study area:

```
    python ..\et-demands\prep\clip_cdl_raster.py --cdl ..\common\cdl --years 2010 --stats -o
```

Mask the non-agricultural CDL pixels:

```
    python ..\et-demands\prep\build_ag_cdl_rasters.py --years 2010 --mask -o --stats
```
