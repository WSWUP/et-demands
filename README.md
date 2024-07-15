[![Docs Status](https://readthedocs.org/projects/et-demands/badge/?version=latest)]( https://readthedocs.org/projects/et-demands/?badge=latest)
[![Build Status](https://travis-ci.org/usbr/et-demands.svg?branch=master)](https://travis-ci.org/usbr/et-demands)
[![Coverage Status](https://coveralls.io/repos/github/usbr/et-demands/badge.svg?branch=master)](https://coveralls.io/github/usbr/et-demands?branch=master)

# CropET
Crop ET Demands Model

## Documentation
ET-Demands [Manual and Documentation](https://et-demands.readthedocs.io/en/bminor_development/)

## Running tools/scripts
Currently, scripts should be run from windows command prompt (or Linux terminal) so that configuration file can be passed as an argument directly to script.  It is possible to execute some scripts by double clicking, but this is still in development.

#### Help
To see what arguments are available for a script and their default values, pass "-h" argument to script.
```
> python run_ret.py -h
usage: run_ret.py [-h] [-i PATH] [-d] [-v] [-m [M]] [-mp [N]]

Reference ET

optional arguments:
  -h, --help            show this help message and exit
  -i PATH, --ini PATH   Configuration (initialization) file (default: None)
  -d, --debug           Save debug level comments to debug.txt (default:
                        False)
  -v, --verbose         Print info level comments (default: False)
  -m, --metid		Met node id to run (default 'ALL')
  -mp [N], --multiprocessing [N]
                        Number of processers to use (default: 1)

> python run_cet.py -h
usage: run_cet.py [-h] [-i PATH] [-d] [-v] [-c [C]] [-mp [N]]

Crop ET-Demands

optional arguments:
  -h, --help            show this help message and exit
  -i PATH, --ini PATH   Configuration (initialization) file (default: None)
                        False)
  -d, --debug           Save debug level comments to debug.txt (default:
                        False)
  -v, --verbose         Print info level comments (default: False)
  -c, --etcid		ET cell to run (default 'ALL')
  -mp [N], --multiprocessing [N]
                        Number of processers to use (default: 1)

> python run_aet.py -h
usage: run_aet.py [-h] [-i PATH] [-d] [-v] [-c [C]] [-mp [N]]

Area ET

optional arguments:
  -h, --help            show this help message and exit
  -i PATH, --ini PATH   Configuration (initialization) file (default: None)
  -d, --debug           Save debug level comments to debug.txt (default:
                        False)
  -v, --verbose         Print info level comments (default: False)
  -c, --etcid		ET cell to run (default 'ALL')
  -mp [N], --multiprocessing [N]
                        Number of processors to use (default: 1)
```


#### Configuration files
Key parameters in configuration files are folder location of current project, static (meta) data file specifications and time series data specifications.  To set configuration file, use "-i" or "--ini" argument.
```
> python run_cet.py -i cet_example.ini
> python run_ret.py -i csv_ret_dri.ini
> python run_cet.py -i csv_cet_dri.ini
> python run_aet.py -i dri_aet_csv.ini
```

#### Multiprocessing
The ET scripts support basic multiprocessing that can be enabled using "-mp N" argument, where N is number of cores to use.  If N is not set, script will attempt to use all cores.  For each ET cell, N crops will be run in parallel.  Using multiprocessing will typically be must faster, but speed improvement may not scale linearly with number of cores because processes are all trying to write to disk at same time.
Multiprocessing is not available for single met node reference et runs or for single et cell area et runs.
Multiprocessing is not available for some output formats.
```
> python run_cet.py -i example.ini -mp
```

#### Plots
Plots of Crop ET-Demands paramters ET, ETo, Kc, growing season, irrigation, precipitation, and NIWR can be generated using plotting tool.  plots are generated using [Bokeh](http://bokeh.pydata.org/en/latest/) and saved as HTML files.  output folder for plots is set in configuration file, typically "daily_plots".
```
> python ..\et-demands\tools\plot_py_crop_daily_timeseries.py -i example.ini
```

## Dependencies
The ET-Demands tools are being developed primarily for Python 3.6+ but should still be mostly backwards compatible with Python 2.7.

Please see environment.yml file for details on the versioning requirements.  Older versions of the modules may work but have not been extensively tested.

#### RefET
+ [NumPy](http://www.numpy.org)
+ [Pandas](http://pandas.pydata.org)
+ [openpyxl](https://pypi.python.org/pypi/openpyxl/2.4.7)

#### CropET
+ [NumPy](http://www.numpy.org)
+ [Pandas](http://pandas.pydata.org)
+ [openpyxl](https://pypi.python.org/pypi/openpyxl/2.4.7)

#### AreaET
+ [NumPy](http://www.numpy.org)
+ [Pandas](http://pandas.pydata.org)
+ [openpyxl](https://pypi.python.org/pypi/openpyxl/2.4.7)

#### Lib
+ [NumPy](http://www.numpy.org)
+ [Pandas](http://pandas.pydata.org)
+ [openpyxl](https://pypi.python.org/pypi/openpyxl/2.4.7)

#### Prep tools
+ [GDAL](http://gdal.org/)
+ [NumPy](http://www.numpy.org)

#### Spatial crop parameters
+ [PyShp](https://github.com/GeospatialPython/pyshp)

#### Time series figures
+ [Bokeh](http://bokeh.pydata.org/en/latest/) is only needed if generating daily time series figures (tools/plot_crop_daily_timeseries.py).  Must be version 0.12.0 to support new responsive plot features.

#### Summary maps
Following modules are only needed if making summary maps (tools/plot_crop_summary_maps.py)

+ [Matplotlib](http://matplotlib.org)
+ [Fiona](https://github.com/Toblerity/Fiona)
+ [Descartes](https://bitbucket.org/sgillies/descartes)
+ [Shapely](https://github.com/Toblerity/Shapely)

## Miniconda

The easiest way to install required external Python modules is to use [Miniconda](https://conda.io/miniconda.html) and create separate conda environments for each project.

It is important to double check that you are calling Miniconda version, especially if you have two or more version of Python installed (e.g. Miniconda and ArcGIS).

+ Windows: "where python"
+ Linux/Mac: "which python"

#### Conda Forge

After installing Miniconda, [conda-forge](https://conda-forge.github.io/) should be added to the list of conda channels in order to have the most up to date python packages:
```
> conda config --add channels conda-forge
```

#### Conda Environment

The "etdemands" conda environment can be created using the "environment.yml":
```
> conda env create -f environment.yml
```

After creating the environment, it then needs to be "activated":
```
conda activate etdemands
```

#### Installing Modules

If you would prefer to not use the conda environment, external modules can installed/updated with conda all at once (this is preferred approach):
```
> conda install numpy pandas gdal bokeh openpyxl>=2.4.7
```

or one at a time:
```
> conda install numpy
> conda install pandas
> conda install gdal
> conda install bokeh
> conda install openpyxl>=2.4.7
```

#### GDAL_DATA

After installing GDAL, you may need to manually set GDAL_DATA user environmental variable.

###### Windows

You can check current value of variable by typing following in command prompt:
```
echo %GDAL_DATA%
```
If GDAL_DATA is set, this will return a folder path (something similar to C:\Miniconda3\Library\share\gdal)

If GDAL_DATA is not set, type following in command prompt (note, your path may vary):
```
> setx GDAL_DATA "C:\Miniconda3\Library\share\gdal"
```

The GDAL_DATA environment variable can also be set through the Windows Control Panel (System -> Advanced system settings -> Environment Variables).

###### Linux/Mac

You can check current value of variable by typing following in command prompt:
```
echo $GDAL_DATA
```
If GDAL_DATA is set, this will return a folder path (something similar to /User/<USER>/Miniconda3/Library/share/gdal)

If GDAL_DATA is not set, type the following in command prompt (note, your path may vary):
```
> export GDAL_DATA=/User/<USER>/Miniconda3/Library/share/gdal"
```
