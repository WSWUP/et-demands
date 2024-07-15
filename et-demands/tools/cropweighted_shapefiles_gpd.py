import argparse
import datetime as dt
import geopandas as gpd
import logging
import numpy as np
import os
import pandas as pd
import re
import shapefile
import sys
# Eventually rename util.py to _util.py
import util as util


def main(ini_path, time_filter, start_doy, end_doy, year_filter=''):

    """Create Median NIWR Shapefiles from annual_stat files

    Args:
        ini_path (str): file path of the project INI file
        year_filter (list): only include certain years for summary
            (single YYYY or range YYYY:YYYY)
        time_filter (str): 'annual', 'growing_season', 'doy'
        start_doy (int): starting julian doy (inclusive)
        end_doy (int): ending julian doy (inclusive)
    Returns:
        None
    """

    # INI path
    config = util.read_ini(ini_path, section='CROP_ET')
    try:
        project_ws = config.get('CROP_ET', 'project_folder')
    except:
        logging.error(
            'project_folder parameter must be set in the INI file, exiting')
        return False
    try:
        gis_ws = config.get('CROP_ET', 'gis_folder')
    except:
        logging.error(
            'gis_folder parameter must be set in the INI file, exiting')
        return False
    try:
        et_cells_path = config.get('CROP_ET', 'cells_path')
    except:
        logging.error(
            'et_cells_path parameter must be set in the INI file, exiting')
        return False

    try:
        daily_output_path = config.get('CROP_ET', 'daily_output_folder')
    except:
        logging.error(
            'ERROR: daily_output_folder ' +
            'parameter is not set inINI file')
        sys.exit()

    # Year Filter
    year_list = None
    if year_filter:
        try:
            year_list = sorted(list(util.parse_int_set(year_filter)))
        except:
            pass

    # Sub folder names
    daily_ws = os.path.join(project_ws, daily_output_path)
    output_ws = os.path.join(project_ws, 'cropweighted_shapefiles')
    if not os.path.exists(output_ws):
        os.makedirs(output_ws)

    # Check input folders
    if not os.path.exists(daily_ws):
        logging.critical('ERROR: The daily_stat folder does not exist.'
                         ' Check .ini settings')
        sys.exit()

    # Check input folders
    if not os.path.isdir(project_ws):
        logging.critical(('ERROR: The project folder ' +
                          'does not exist\n  {}').format(project_ws))
        sys.exit()
    elif not os.path.isdir(gis_ws):
        logging.critical(('ERROR: The GIS folder ' +
                          'does not exist\n  {}').format(gis_ws))
        sys.exit()
    logging.info('\nGIS Workspace:      {0}'.format(gis_ws))

    # Create Output folder if it doesn't exist
    output_folder_path = os.path.join(project_ws, 'cropweighted_shapefiles')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Regular expressions
    data_re = re.compile('(?P<CELLID>\w+)_crop_(?P<CROP>\d+).csv$', re.I)
    # data_re = re.compile('(?P<CELLID>\w+)_daily_crop_(?P<CROP>\d+).csv$',
    #  re.I)

    # Build list of all data files
    data_file_list = sorted(
        [os.path.join(daily_ws, f_name) for f_name in os.listdir(daily_ws)
         if data_re.match(f_name)])
    if not data_file_list:
        logging.error(
            '  ERROR: No daily ET files were found\n' +
            '  ERROR: Check the folder_name parameters\n')
        sys.exit()

    cells = read_shapefile(et_cells_path)

    # Start with empty lists
    stations = []
    crop_nums = []

    # Process each file
    for file_path in data_file_list:
        file_name = os.path.basename(file_path)
        logging.debug('')
        logging.info('  {0}'.format(file_name))

        # station, crop_num = os.path.splitext(file_name)[0].split(
        # '_daily_crop_')
        station, crop_num = os.path.splitext(file_name)[0].split('_crop_')
        stations.append(station)
        crop_num = int(crop_num)
        crop_nums.append(crop_num)

    # Find unique crops and station ids
    unique_crop_nums = list(set(crop_nums))
    unique_stations = list(set(stations))

    # Testing (should this be an input option?)
    # unique_crop_nums = [86]
    # unique_stations = [608807]

    # Variables to calculate output statistics
    var_list = ['ETact', 'NIWR', 'P_rz', 'P_eft', 'PrzF', 'PeftF']

    logging.info('\nCreating Crop Area Weighted Shapefiles')
    # Apply Time Filter (annual, etd growing season, doy (start/end))
    if time_filter == 'annual':
        logging.info('\nSummarizing data based on calendar year.')
    if time_filter == 'growing_season':
        logging.info(
            '\nFiltering data using ETDemands defined growing season.')
    if time_filter == 'doy':
        logging.info('\nFiltering data using doy inputs. Start doy: {:03d} '
                     'End doy: {:03d}'.format(start_doy, end_doy))
    if time_filter == 'water_year':
        logging.info(
            '\nSummarizing data based on water year.')

    for crop in unique_crop_nums:
        logging.info('\n Processing Crop: {:02d}'.format(crop))

        # Initialize df variable to check if pandas df needs to be created
        df = None
        for station in unique_stations:
            # Build File Path
            file_path = os.path.join(daily_ws,
                                     '{}_crop_{:02d}.csv'.format(station,
                                                                 crop))
            # Only process files that exists (crop/cell combinations)
            if not os.path.exists(file_path):
                continue

            # Read file into df
            daily_df = pd.read_csv(file_path, skiprows=1)

            # Apply Time Filter (annual, etd growing season, doy (start/end))
            if time_filter == 'growing_season':
                daily_df = daily_df[(daily_df['Season'] == 1)]
            if time_filter == 'doy':
                daily_df = daily_df[
                    (daily_df['DOY'] >= start_doy) &
                    (daily_df['DOY'] <= end_doy)]
            if time_filter == 'water_year':
                daily_df['WY'] = daily_df.Year.where(daily_df.Month < 10, daily_df.Year + 1)

            # Apply Year Filter (apply after adding water year)
            if year_list:
                if time_filter == 'water_year':
                    daily_df = daily_df[
                        (daily_df['WY'] >= min(year_list)) &
                        (daily_df['WY'] <= max(year_list))]
                else:
                    daily_df = daily_df[
                        (daily_df['Year'] >= min(year_list)) &
                        (daily_df['Year'] <= max(year_list))]
                    # logging.info('Including Years: {}'.format(year_list))


            if daily_df.empty:
                logging.info(' Growing Season never started. Skipping cell {}'
                             ' for crop {}.'.format(station, crop))
                continue

            # Dictionary to control agg of each variable
            a = {
                'ETact': 'sum',
                'NIWR': 'sum',
                'P_rz': 'sum',
                'P_eft': 'sum',
                'PPT': 'sum'}

            # GroupStats by Year of each column follow agg assignment above
            if time_filter == 'water_year':
                yearlygroup_df = daily_df.groupby('WY', as_index=True).agg(a)
            else:
                yearlygroup_df = daily_df.groupby('Year', as_index=True).agg(a)

            yearlygroup_df['PrzF'] = yearlygroup_df.P_rz / yearlygroup_df.PPT
            yearlygroup_df['PeftF'] = yearlygroup_df.P_eft / yearlygroup_df.PPT

            # Take Mean of Yearly GroupStats
            mean_df = yearlygroup_df.mean(axis=0)
            mean_fieldnames = [v + '_mn_{:02d}'.format(crop) for v in var_list]

            # Take Median of Yearly GroupStats
            median_df = yearlygroup_df.median(axis=0)
            median_fieldnames = [v + '_md_{:02d}'.format(crop) for v in
                                 var_list]

            # Create Dataframe if it doesn't exist
            if df is None:
                df = pd.DataFrame(index=unique_stations,
                                  columns=mean_fieldnames + median_fieldnames)

            # Write data to each station row
            df.loc[station] = list(mean_df[var_list]) + list(median_df[
                                                                 var_list])

            # Cast summary objects to floats
            df = df.astype(float)

            # Grab min/max year for output folder naming
            # assumes all daily files cover same time period
            # year represents CY or WY based on time_filter
            min_year = min(year_list)
            max_year = max(year_list)

        # Convert index to integers
        df.index = df.index.map(int)

        # Remove rows with Na (Is this the best option???)
        df = df.dropna()

        # Merge Crop ETact and NIWR to cells dataframe
        cells = pd.merge(cells, df, how='left', left_on=['CELL_ID'],
                         right_index=True)

    # Change Ag_Acres cells with zero area to nan (Avoid ZeroDivisionError)
    cells[cells['AG_ACRES'] == 0] = np.nan

    # Calculate CropArea Weighted ETact and NIWR for each cell
    # List Comprehension (all combinations of var_list and stat)
    # https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch01s15.html
    for var, stat in [(var, stat) for var in var_list for stat in ['mn', 'md']]:
        # initialize empty columns (zeros)
        cells['CW{0}_{1}'.format(var, stat)] = 0
        for crop in unique_crop_nums:
            print(crop)
            # reset temp
            temp = []
            # calculate crop fraction of weighted rate
            temp = cells['CROP_{0:02d}'.format(crop)].multiply(
                cells['{0}_{1}_{2:02d}'.format(var, stat, crop)]).divide(
                cells['AG_ACRES'])
            # replace nan with zero
            temp = temp.fillna(0)
            # add crop fraction to total calculate weighted rate
            cells['CW{0}_{1}'.format(var, stat)] = cells['CW{0}_{1}'.format(
                var, stat)].add(temp)

    # Subset to "Final" dataframe for merge to output .shp
    # final_df = cells[['GRIDMET_ID', 'CWETact_mn', 'CWNIWR_mn', 'CWETact_md',
    #                   'CWNIWR_md']]
    final_df = cells[['CELL_ID', 'CWETact_mn', 'CWNIWR_mn', 'CWETact_md',
                      'CWNIWR_md', 'CWPrzF_mn', 'CWPrzF_md', 'CWPeftF_mn', 'CWPeftF_md']]


    # Copy ETCELLS.shp and join cropweighted data to it
    data = gpd.read_file(et_cells_path)

    # UPDATE TO NEWER ETCELLS STATION_ID FORMAT !!!!!
    merged_data = data.merge(final_df, on='CELL_ID')

    # Output file name
    out_name = "{}_cropweighted.shp".format(time_filter)
    if time_filter == 'doy':
        out_name = "{}_{:03d}_{:03d}_cropweighted.shp".format(
            time_filter, start_doy, end_doy)

    # output folder
    output_folder_path = os.path.join(output_ws, 'cropweighted_{}to{}'.format(
        min_year, max_year))
    if min_year == max_year:
        output_folder_path = os.path.join(output_ws, 'cropweighted_{}'.format(
            min_year))

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    # Write output .shp
    merged_data.to_file(os.path.join(output_folder_path, out_name))


def read_shapefile(shp_path):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' column holding
    the geometry information. This uses the pyshp package
    """
    # read file, parse out the records and shapes
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    # shps = [s.points for s in sf.shapes()]
    # write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)
    # df = df.assign(coords=shps)
    return df


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET-Demands Cropweighted Shapefiles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-t', '--time_filter', default='annual', choices=['annual',
                                                          'growing_season',
                                                          'doy', 'water_year'], type=str,
        help='Data coverage options. If "doy", -start_doy and'
             ' -end_doy required.')
    parser.add_argument(
        '-s', '--start_doy', type=int,
        help='Starting julian doy (inclusive)')
    parser.add_argument(
        '-e', '--end_doy', type=int,
        help='Ending julian doy (inclusive)')
    parser.add_argument(
        '-y', '--year', default='', type=str,
        help='Years to include, single year (YYYY) or range (YYYY-YYYY)')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    if args.time_filter == 'doy' and \
            (args.start_doy is None or args.end_doy is None):
        logging.error(
            '\nstart/end doy must be input when using "doy" time filter.'
            '\nExiting.')
        sys.exit()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, time_filter=args.time_filter,
         start_doy=args.start_doy, end_doy=args.end_doy, year_filter=args.year)
