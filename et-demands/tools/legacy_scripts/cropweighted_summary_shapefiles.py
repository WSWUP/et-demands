import argparse
import pandas as pd
import os
import re
import logging
import sys
# Eventually rename util.py to _util.py
import util as util
import datetime as dt
import numpy as np
import geopandas as gpd
import shapefile


def main(ini_path, overwrite_flag=True, cleanup_flag=True,
         growing_season = False, year_filter=[]):
    """Create Crop Area Weighted ETact and NIWR shapefiles
     from monthly_stat files

    Args:
        ini_path (str): file path of the project INI file
        overwrite_flag (bool): If True (default), overwrite existing files
        cleanup_flag (bool): If True, remove temporary files
        growing_season (bool): If True, filters data to April-October
        year_filter (int): Only includes data for one year in statistics
    Returns:
        None
    """
    # print('SCRIPT STILL IN DEVELOPMENT (SEE CODE). EXITING')
    # sys.exit()

    logging.info('\nCreating Crop Area Weighted Shapefiles')
    #  INI path
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

    # Year Filter
    if year_filter:
        logging.info('\nEstimating Data for {0}'.format(year_filter))

    # Sub folder names
    daily_ws = os.path.join(project_ws, 'daily_stats')
    gs_ws = os.path.join(project_ws, 'growing_season_stats')

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
    output_folder_path = os.path.join(project_ws, 'cropweighted_shapefile')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Regular expressions
    data_re = re.compile('(?P<CELLID>\w+)_crop_(?P<CROP>\d+).csv$', re.I)
    # data_re = re.compile('(?P<CELLID>\w+)_daily_crop_(?P<CROP>\d+).csv$', re.I)

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

        # station, crop_num = os.path.splitext(file_name)[0].split('_daily_crop_')
        station, crop_num = os.path.splitext(file_name)[0].split('_crop_')
        stations.append(station)
        crop_num = int(crop_num)
        crop_nums.append(crop_num)

    # Find unique crops and station ids
    unique_crop_nums = list(set(crop_nums))
    unique_stations = list(set(stations))

    # Variables to calculate output statistics
    var_list = ['ETact', 'NIWR']

    logging.info('\n Creating Crop Area Weighted Shapefiles')
    if year_filter:
        logging.info('\nFiltering by Year: {}'.format(year_filter))
    if growing_season:
        logging.info('\nFiltering stats to Growing Season, Apr-Oct')

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
            # Filter data based on year_filter
            if year_filter:
                daily_df = daily_df[daily_df['Year']== year_filter]
                logging.info('\nFiltering by Year: {}'.format(year_filter))

            # Remove all non-growing season data if growing season flag = True
            # UPDATE TO USE SEASON FLAG IN DAILY CSV FILES (0 or 1)
            if growing_season:
                daily_df = daily_df[
                    (daily_df['Month'] >= 4) & (daily_df['Month'] <= 10)]
                logging.info('\nFiltering stats to Growing Season, Apr-Oct')
            # if growing_season:
            #     daily_df = daily_df[(daily_df['Season'] == 1)]

            # Dictionary to control agg of each variable
            a = {
                'ETact': 'sum',
                'NIWR': 'sum'}

            # GroupStats by Year of each column follow agg assignment above
            yearlygroup_df = daily_df.groupby('Year', as_index=True).agg(a)

            #Take Mean of Yearly GroupStats
            mean_df = yearlygroup_df.mean(axis=0)
            mean_fieldnames = [v + '_mn_{:02d}'.format(crop) for v in var_list]

            #Take Median of Yearly GroupStats
            median_df = yearlygroup_df.median(axis=0)
            median_fieldnames =[v + '_md_{:02d}'.format(crop) for v in var_list]

            #Create Dataframe if it doesn't exist
            if df is None:
               df = pd.DataFrame(index=unique_stations,
                                 columns=mean_fieldnames + median_fieldnames)

            #Write data to each station row
            df.loc[station] = list(mean_df[var_list]) + \
                              list(median_df[var_list])

        # Convert index to integers)
        df.index = df.index.map(int)

        # Remove rows with Na (Is this the best option???)
        df = df.dropna()

        # Merge Crop ETact and NIWR to cells dataframe
        cells = pd.merge(cells, df, how='left', left_on=['GRIDMET_ID'],
                         right_index=True)

    # Change Ag_Acres cells with zero area to nan (Avoid ZeroDivisionError)
    cells[cells['AG_ACRES'] == 0] = np.nan

    # Calculate CropArea Weighted ETact and NIWR for each cell
    # List Comprehension (All combinations of var_list and stat)
    # https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch01s15.html
    for var, stat in [(var, stat) for var in var_list for stat in ['mn', 'md']]:
        # nitialize empty columns (zeros)
        cells['CW{0}_{1}'.format(var, stat)] = 0
        for crop in unique_crop_nums:
            # reset temp
            temp = []
            # calculate crop fraction of weighted rate
            temp = cells['CROP_{0:02d}'.format(crop)].multiply(
                cells['{0}_{1}_{2:02d}'.format(var, stat, crop)]).divide(cells['AG_ACRES'])
            # replace nan with zero
            temp = temp.fillna(0)
            # add crop fraction to total calculate weighted rate
            cells['CW{0}_{1}'.format(var, stat)] = cells['CW{0}_{1}'.format(var, stat)].add(temp)

    # Subset to "Final" dataframe for merge to output .shp
    Final = cells[['GRIDMET_ID', 'CWETact_mn', 'CWNIWR_mn','CWETact_md', 'CWNIWR_md']]

    # Copy ETCELLS.shp and join cropweighted data to it
    data = gpd.read_file(et_cells_path)

    # UPDATE TO NEWER ETCELLS STATION_ID FORMAT !!!!!
    merged_data = data.merge(Final, on='GRIDMET_ID')
    if not year_filter:
        year_filter = 'AllYears'
    if growing_season:
        out_filepath = os.path.join(output_folder_path,
                                    '{}_GS_CropWeighted.shp'.format(year_filter))
    else:
        out_filepath = os.path.join(output_folder_path,
                                    '{}_Ann_CropWeighted.shp'.format(year_filter))
    #Write output .shp
    merged_data.to_file(out_filepath)


def read_shapefile(shp_path):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' column holding
    the geometry information. This uses the pyshp package
    """
    # read file, parse out the records and shapes
    sf = shapefile.Reader(shp_path)
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    #	shps = [s.points for s in sf.shapes()]

    # write into a dataframe
    df = pd.DataFrame(columns=fields, data=records)
    #	df = df.assign(coords=shps)
    return df

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET-Demands Annual Stat Shapefiles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-o', '--overwrite', default=True, action='store_true',
        help='Overwrite existing file')
    parser.add_argument(
        '--clean', default=False, action='store_true',
        help='Remove temporary datasets')
    parser.add_argument(
        '-y', '--year', type=int,
        help='Year of interest (single year)')
    parser.add_argument(
        '-gs','--growing_season', default=False, action='store_true',
        help='Growing Season Flag, Include only April-October Data')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
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

    main(ini_path=args.ini, overwrite_flag=args.overwrite,
         cleanup_flag=args.clean, growing_season=args.growing_season,
         year_filter=args.year)


















