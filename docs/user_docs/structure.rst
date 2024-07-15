ET-Demands Folder Structure
===========================

The ET-Demands scripts and tools are assuming that the user will use a folder structure similar to the one below.  The exact folder paths can generally be adjusted by either changing the INI file or explicitly setting the folder using the script command line arguments.  Most of the GIS sub-folders can be built and populated using the :doc:`CropET prep tools<prep_tools>`::

    et-demands
    |
    +---common
    |   +---cdl
    |   |       2010_30m_cdls.img
    |   |       2010_30m_cdls.zip
    |   +---huc8
    |   |       wbdhu8_albers.shp
    |   +---gridmet_4km
    |   |       gridmet_4km_albers_full.shp
    |   \---soils
    |           AWC_WTA_0to152cm_statsgo.shp
    |           Clay_WTA_0to152cm_statsgo.shp
    |           Sand_WTA_0to152cm_statsgo.shp
    |
    +---et-demands
    |   +---cropET
    |   |   \---bin
    |   +---prep
    |   +---refET
    |   +---static
    |   |       CropCoefs.txt
    |   |       CropParams.txt
    |   |       ETCellsCrops.txt
    |   |       ETCellsProperties.txt
    |   |       EToRatiosMon.txt
    |   |       MeanCuttings.txt
    |   |       TemplateMetAndDepletionNodes.xlsx
    |   \---tools
    |
    \---example
        |       example.ini
        +---annual_stats
        +---daily_baseline
        +---daily_plots
        +---daily_stats
        +---gis
        |   +---cdl
        |   +---huc8
        |   |       wbdhu8_albers.shp
        |   +---soils
        |   |       awc_30m_albers.img
        |   |       clay_30m_albers.img
        |   |       sand_30m_albers.img
        |   \---stations
        |           gridmet_4km_dd_pts.shp
        +---monthly_stats
        \---static