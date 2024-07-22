# -*- coding: utf-8 -*-
"""
Download gridMET climatic time series for multiple variables, e.g. ETr, temp,
wind speed 2m, swrad, etc. Uses `OpeNDAP <https://www.opendap.org>`_.

"""
import argparse
import datetime as dt
import logging
import os
import sys
import timeit
from time import sleep

import pandas as pd
pd.options.display.float_format = '{:,.10f}'.format
import refet
import xarray

def main(input_csv, out_folder, year_filter='', 
        update_data=False, optional_vars=None):

    """
    Download gridMET time series data for multiple climate variables for 
    select gridMET cells as listed in ``input_csv``. Uses 
    `OpeNDAP <https://www.opendap.org>`_ and the Thredds server.
    
    There are three columns that are required to be in the first line of 
    the input CSV file, which can contain any number of stations or locations
    to download, those are: "LAT", "LON", and "GRIDMET_ID". Lat and lon
    are in decimal degrees, and "GRIDMET_ID" is the grid cell ID number
    that corresponds with the cell that is to have gridMET data downloaded. 
    
    The gridMET cell IDs are used only to name the output files at this
    point, and this script could be easily modified to not require knowledge
    of the gridMET cell ID's. 
    
    Here is an example of the contents of a valid input CSV file:
    
		+------------+------------+---------------+----------------+
		| Station_ID | GRIDMET_ID | LAT           | LON            |
		+============+============+===============+================+
		| blbu       | 509011     | 40.3583333333 | -110.224999967 |
		| loau       | 443835     | 38.4          | -111.641666633 |
		| BRK01      | 441130     | 38.3166666667 | -108.849999967 |
		| csvu       | 452205     | 38.65         | -109.391666633 |
		+------------+------------+---------------+----------------+
		
	Any additional columns in the file are also allowed, after running
	this script a new column with the paths to the downloaded gridMET
	files for each row will be added. 
    
    By default, this function will download the following variables: etr,
    pet, pr, sph, srad, vs, tmmx, and tmmn. Note these are the short names
    as hosted on OpeNDAP. You can browse the data here: 
    http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_MET_catalog.html 
    The variables th and vpd, are also available using
    the ``optional_vars`` argument. 
    
    This function will download elevation
    from the gridMET dataset at the locations given and uses this elevation
    to compute air pressure and then vapor pressure. In addition, the 10 m wind
    speed ('vs') from gridMET is scaled to 2 m using the vertical logarithmic
    velocity profile as given in equation 33 in [Allen2005]. These calculations
    are provided in the RefET package (https://github.com/WSWUP/RefET). The 
    actual downloading of gridMET timeseries data uses the Xarray Python package
    (https://docs.xarray.dev/en/stable/index.html). Other unit conversions and
    reformatting of variable names will also be performed, and the following
    will be final header of the downloaded gridMET data timeseries:
    
		['date',
		 'year',
		 'month',
		 'day',
		 'centroid_lat',
		 'centroid_lon',
		 'elev_m',
		 'u2_ms',
		 'tmin_c',
		 'tmax_c',
		 'srad_wm2',
		 'ea_kpa',
		 'pair_kpa',
		 'prcp_mm',
		 'etr_mm',
		 'eto_mm']
		 
	And the GRIDMET_ID will be used to name the output files e.g., 
	"gridmet_historical_509011.csv".

    Arguments:
        input_csv (str): file path of input CSV with LAT, LON, and GRIDMET_ID
        out_folder (str): directory path to save gridMET timeseries CSV files

    Keyword Arguments:
        year_filter (list): default ''. Single year or range to download
        update_data (bool): default False. Re-download existing data
        optional_vars (None or list): default None. List of additional gridMET 
            vars to download using gridMET short names, currently wind 
            direction named "th" and vapor pressure deficit "vpd" are 
            available.

    Returns:
        None

    Examples:
        Say we wanted to download data for 2016 through 2018, from the command 
        line,

        .. code-block:: sh

            $ download_gridmet_opendap.py -i input.csv -o gridmet_data -y 2016-2018

        note, "input.csv" should have been created by first. If the 
        ``[-y, --years]`` option is note given the default behaviour is to 
        download gridMET data from 1979 up through yesterday.

        If the data for 2018 has changed since the last run or for debugging
        purposes you can re-download data for all or select years with the
        ``[-u, --update-data]`` option

        .. code-block:: sh

            $ download_gridmet_opendap.py -i merged_input.csv -o gridmet_data -y 2018 -u

        To download the same gridMET data within Python

        >>> import download_gridmet_opendap
        >>> download_gridmet_opendap.main('input.csv',
                'gridmet_data',
                '2016-2018'
            )

        Again, running download_gridmet_opendap.py updates the CSV input file
        to include file paths to gridMET time series files that were downloaded. 

    References:
		[Allen2005] R.G. Allen, I.A. Walter, R.L. Elliott, T.A. Howell, D. Itenfisu, 
			M.E. Jensen, and R.L. Snyder, The ASCE Standardized Reference 
			Evapotranspiration Equation, American Society of Civil Engineers, 
			2005. https://doi.org/10.1061/9780784408056.

    """
    if not os.path.exists(out_folder):
        logging.info('\nCreating output folder: {}'.format(out_folder))
        os.makedirs(out_folder)

    # Input .csv containing GRIDMET_ID, LAT, LON
    input_df = pd.read_csv(input_csv)

    # Specify column order for output .csv Variables:
    output_order = ['date', 'year', 'month', 'day', 'centroid_lat',
                    'centroid_lon', 'elev_m', 'u2_ms', 'tmin_c', 'tmax_c',
                    'srad_wm2', 'ea_kpa', 'pair_kpa', 'prcp_mm', 'etr_mm', 
                    'eto_mm']
    opendap_url = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC'
    elev_nc = '{}/{}'.format(
        opendap_url, '/MET/elev/metdata_elevationdata.nc#fillmismatch')
    params = {
        'etr': {
            'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
            'var': 'daily_mean_reference_evapotranspiration_alfalfa',
            'col': 'etr_mm'},
        'pet': {
            'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
            'var': 'daily_mean_reference_evapotranspiration_grass',
            'col': 'eto_mm'},
        'pr': {
            'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
            'var': 'precipitation_amount',
            'col': 'prcp_mm'},
        'sph': {
            'nc': 'agg_met_sph_1979_CurrentYear_CONUS',
            'var': 'daily_mean_specific_humidity',
            'col': 'q_kgkg'},
        'srad': {
            'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
            'var': 'daily_mean_shortwave_radiation_at_surface',
            'col': 'srad_wm2'},
        'vs': {
            'nc': 'agg_met_vs_1979_CurrentYear_CONUS',
            'var': 'daily_mean_wind_speed',
            'col': 'u10_ms'},
        'tmmx': {
            'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
            'var': 'daily_maximum_temperature',
            'col': 'tmax_k'},
        'tmmn': {
            'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
            'var': 'daily_minimum_temperature',
            'col': 'tmin_k'},
        'th': {
            'nc': 'agg_met_th_1979_CurrentYear_CONUS',
            'var': 'daily_mean_wind_direction',
            'col': 'wdir_deg'},
        'vpd': {
            'nc': 'agg_met_vpd_1979_CurrentYear_CONUS',
            'var': 'daily_mean_vapor_pressure_deficit',
            'col': 'vpd_kpa'}
    }

    extra_vars_available = ['th','vpd']
    default_vars = list(set(params.keys()).difference(extra_vars_available))

    if optional_vars is not None:
        if isinstance(optional_vars, list):
            pass
        elif isinstance(optional_vars, str): # if using CLI
            optional_vars = optional_vars.split(',')

        if not set(optional_vars).issubset(extra_vars_available):
            raise ValueError(
                'ERROR: one or more optional gridMET variables is not'
                'avaliable, these are all currently available: '
                f'{",".join(params.keys())}'
            )
        # add the written names to list 
        else:
            for el in optional_vars:
                output_order.append(params[el].get('col'))
    else:
        optional_vars = []

    
    # Year Filter
    if year_filter:
            year_list = sorted(list(_parse_int_set(year_filter)))
            logging.info('\nDownloading Years: {0}-{1}'.format(min(year_list),
                                                               max(year_list)))
            date_list = pd.date_range(
                dt.datetime.strptime('{}-01-01'.format(min(year_list)),
                                     '%Y-%m-%d'),
                dt.datetime.strptime('{}-12-31'.format(max(year_list)),
                                     '%Y-%m-%d'))
    else:
        logging.info('\nDownloading full historical record (1979-present).')
        # Create List of all dates
        # determine end date of data collection
        current_date = dt.datetime.today()
        end_date = dt.date(current_date.year, current_date.month,
                           current_date.day - 1)
        date_list = pd.date_range(dt.datetime.strptime('1979-01-01',
                                                       '%Y-%m-%d'), end_date)

    # Loop through dataframe row by row and grab desired met data from
    # GRID collections based on Lat/Lon and Start/End dates
    for index, row in input_df.iterrows():
        start_time = timeit.default_timer()
        # Reset original_df
        original_df = None
        export_df = None

        GRIDMET_ID_str = str(row.GRIDMET_ID)
        logging.info('Processing GRIDMET ID: {}'.format(GRIDMET_ID_str))

        output_name = 'gridmet_historical_' + GRIDMET_ID_str + '.csv'
        output_file = os.path.join(out_folder, output_name)

        if os.path.isfile(output_file):
            logging.info('{} exists. Checking for missing data.'.format(
                output_name))
            original_df = pd.read_csv(output_file, parse_dates=True)
            missing_dates = list(set(date_list) - set(pd.to_datetime(
                original_df['date'])))
            original_df.date = pd.to_datetime(original_df.date.astype(str),
                                              format='%Y-%m-%d')
            original_df['date'] = original_df.date.apply(lambda x: x.strftime(
                '%Y-%m-%d'))
        else:
            logging.info('{} does not exists. Creating file.'.format(
                output_name))
            missing_dates = list(set(date_list))
        if not missing_dates and not update_data:
            logging.info('No missing data found. Skipping')
            # Add gridMET file path to input table if not already there
            input_df.loc[input_df.GRIDMET_ID == row.GRIDMET_ID,\
                'GRID_FILE_PATH'] = os.path.abspath(output_file)
            input_df.to_csv(input_csv, index=False)
            continue
        # for option to redownload all data in given year range
        elif not missing_dates and update_data:
            if min(date_list.year) == max(date_list.year):
                yr_rng = min(date_list.year)
            else:
                yr_rng = '{}-{}'.format(
                    min(date_list.year), max(date_list.year))
            logging.info('Updating data for years: {}'.format(yr_rng))
            start_date = min(date_list)
            end_date = max(date_list)
            # missing years are all years for updating
            missing_years = sorted(list(set(date_list.year)))

        # only download years with missing data
        else:
            # Min and Max of Missing Dates (Start: Inclusive; End: Exclusive)
            start_date = min(missing_dates)
            end_date = max(missing_dates)
    
            missing_years = []
            for date in missing_dates:
                missing_years = missing_years + [date.year]
            missing_years = sorted(list(set(missing_years)))
        logging.debug('  Missing Years: {}'.format(
            ', '.join(map(str, missing_years))))

        # Loop through ee pull by year (max 5000 records for getInfo())
        # Append each new year on end of dataframe
        # Process dataframe units and output after
        start_date_yr = start_date.year    
            
        # Calculate out grid cell centroid
        # gridMET elevation asset lower left corner coordinates
        gridmet_lon = -124.78749996666667
        gridmet_lat = 25.04583333333334
        gridmet_cs = 0.041666666666666664
        gridcell_lat = (int(abs(row.LAT - gridmet_lat) / gridmet_cs) * gridmet_cs + 
                        gridmet_lat + 0.5 * gridmet_cs)
        gridcell_lon = (int(abs(row.LON - gridmet_lon) / gridmet_cs) * gridmet_cs + 
                        gridmet_lon + 0.5 * gridmet_cs)
        gridmet_crs = 'EPSG:4326'
        gridmet_geo = [gridmet_cs, 0, gridmet_lon, 
                       0, -gridmet_cs, gridmet_lat + gridmet_cs * 585]
        logging.debug('  Latitude:  {}'.format(gridcell_lat))
        logging.debug('  Longitude: {}'.format(gridcell_lon))

        # OpenDAP call for elevation
        elev_ds = xarray.open_dataset(elev_nc)\
            .sel(lon=gridcell_lon, lat=gridcell_lat, method='nearest')
        elev = elev_ds['elevation'].values[0]
        logging.debug('  Elevation: {}'.format(elev))
        
        # OpenDAP call for each variable
        met_df_list = []
        vars_to_download = default_vars + optional_vars
        for met_name in vars_to_download:
            logging.debug('  Variable: {}'.format(met_name))
            # connect with the full time series then filtering later is faster              # # day=pd.date_range(start=start_date, end=end_date), 

            # had to add #fillmismatch to fill any data entries that are of
            # a different type (presumably incorrectly filled null values
            # by the OpenDap Thredds server) with null values. 
            # issue here: https://github.com/Unidata/netcdf-c/issues/1299
            n_attempts = 3
            for i in range(n_attempts):
                try:# try to download up to n times before raising error
                    met_nc = '{}/{}.nc#fillmismatch'.format(
                        opendap_url, params[met_name]['nc']
                    )
                    met_ds = xarray.open_dataset(met_nc).sel(
                            lon=gridcell_lon, lat=gridcell_lat, method='nearest'
		            ).drop_vars(
                            ['crs', 'lat', 'lon']
                        ).rename({
			            params[met_name]['var']:params[met_name]['col'],
                        'day':'date'
                    })
                except:
                    sleep(5)
                    logging.info(
                        f'Failed to download {met_name}, attempt {i+1}'
                    )
                    if i == n_attempts-1:
                        raise RuntimeError(
			                f'ERROR: Failed to download {met_name} after '
                            f'{n_attempts} attempts'
                        )
                    continue
                break

            met_df = met_ds.to_dataframe()
            # logging.debug(met_df.head())
            met_df_list.append(met_df)

        # This might need to be a merge call if the indices don't match
        export_df = pd.concat(met_df_list, axis=1)
        logging.debug(export_df.head())
        # End OpenDAP stuff

        # If export_df is None (skip to next ID)
        if export_df is None:
            continue
        # Reset Index
        export_df = export_df.reset_index(drop=False)

        # Convert dateNum to datetime and create Year, Month, Day, DOY variables
        export_df.date = pd.to_datetime(export_df.date.astype(str),
                                        format='%Y-%m-%d')
        export_df['year'] = export_df['date'].dt.year
        export_df['month'] = export_df['date'].dt.month
        export_df['day'] = export_df['date'].dt.day
        # export_df['DOY'] = export_df['Date'].dt.dayofyear
        # Format Date for export
        export_df['date'] = export_df.date.apply(lambda x: x.strftime(
            '%Y-%m-%d'))

        # CGM - This is needed here since filtering by date on the OpenDAP call was really slow
        # Should the dataframe be filtered to missing years or based on start/end date?
        # export_df = export_df.loc[(export_df['date'] >= start_date.strftime('%Y-%m-%d')) & 
        #                           (export_df['date'] <= end_date.strftime('%Y-%m-%d'))]
        export_df = export_df.loc[export_df['year'].isin(missing_years)]
        # export_df = export_df.reset_index(drop=False)
        
        # Remove all negative Prcp values (GRIDMET Bug)
        export_df.prcp_mm = export_df.prcp_mm.clip(lower=0)

        # Convert 10m windspeed to 2m (ASCE Eqn. 33)
        zw = 10
        export_df['u2_ms'] = refet.calcs._wind_height_adjust(
            export_df.u10_ms, zw)
        # elevation from gridMET elevation layer
        export_df['elev_m'] = elev
        export_df['centroid_lat'] = gridcell_lat
        export_df['centroid_lon'] = gridcell_lon

        # air pressure from gridmet elevation using refet module
        export_df['pair_kpa'] = refet.calcs._air_pressure(
            export_df.elev_m, method='asce')

        # actual vapor pressure (kg/kg) using refet module
        export_df['ea_kpa'] = refet.calcs._actual_vapor_pressure(
            export_df.q_kgkg, export_df.pair_kpa)

        # Unit Conversions
        export_df['tmax_c'] = export_df.tmax_k - 273.15  # K to C
        export_df['tmin_c'] = export_df.tmin_k - 273.15  # K to C
        # export_df['Tavg_C'] = (export_df.Tmax_C + export_df.Tmin_C)/2

        # Relative Humidity from gridMET min and max
        # export_df['RH_avg'] = (export_df.RH_max + export_df.RH_min)/2

        # Add new data to original dataframe, remove duplicates
        export_df = pd.concat([original_df, export_df], ignore_index=True,
                              sort=True)
        export_df = export_df[output_order].drop_duplicates('date')
        export_df = export_df.sort_values(by=['year', 'month', 'day'])
        export_df = export_df.dropna(how='all')

        # Add gridMET file path to input table
        input_df.loc[input_df.GRIDMET_ID == row.GRIDMET_ID,\
                'GRID_FILE_PATH'] = os.path.abspath(output_file)
        input_df.to_csv(input_csv, index=False)

        # Write csv files to working directory
        export_df.to_csv(output_file, columns=output_order, index=False, 
                         float_format='%.10f')
        elapsed = timeit.default_timer() - start_time
        logging.info('\nDownload Time: {}'.format(elapsed))


def _parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    http://thoughtsbyclayg.blogspot.com/2008/10/
    parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()
    # tokens are comma separated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def arg_parse():
    """
    Command line usage of download_gridmet_opendap.py for downloading
    gridMET time series of several climatic variables using OpenDAP .
    """
    parser = argparse.ArgumentParser(
        description=arg_parse.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    optional = parser._action_groups.pop() # optionals listed second
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-i', '--input', metavar='', required=True,
        help='Input file containing station LAT, LON, and gridMET ID')
    required.add_argument(
        '-o', '--out-dir', metavar='', required=True,
        help='Output directory to save time series CSVs of gridMET data')
    optional.add_argument(
        '-y', '--years', metavar='', default=None, type=str,
        help='Year(s) to download, single year (YYYY) or range (YYYY-YYYY)')
    optional.add_argument(
        '-u','--update-data', required=False, default=False, 
        action='store_true', help='Flag to re-download existing data')
    optional.add_argument(
        '-ov','--optional-vars', required=False, default=None, 
        type=str, help='Additional gridMET vars as comma separated list')
    optional.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser._action_groups.append(optional)# to avoid optionals listed first
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(input_csv=args.input, out_folder=args.out_dir,
         year_filter=args.years, update_data=args.update_data, 
         optional_vars=args.optional_vars)
