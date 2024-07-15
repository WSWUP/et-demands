import argparse
import pandas as pd
import os
import re
import logging
import sys
import geopandas as gpd
#Eventually rename util.py to _util.py
import util as util
import datetime as dt

def main(ini_path, overwrite_flag=True, cleanup_flag=True , year_filter=''):
    """Create Median NIWR Shapefiles from annual_stat files

    Args:
        ini_path (str): file path of the project INI file
        overwrite_flag (bool): If True (default), overwrite existing files
        cleanup_flag (bool): If True, remove temporary files

    Returns:
        None
    """
    logging.info('\nCreating Annual Stat Shapefiles')
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
#    try:
#        calibration_ws = config.get(crop_et_sec, 'spatial_cal_folder')
#    except:
#        calibration_ws = os.path.join(project_ws, 'calibration')
    try:
        etref_field = config.get('REFET', 'etref_field')
    except:
        logging.error(
            'etref_field parameter must be set in the INI file, exiting')
        return False

    # Sub folder names
    annual_ws = os.path.join(project_ws, 'annual_stats')

    # Check input folders
    if not os.path.exists(annual_ws):
        logging.critical('ERROR: The annual_stat folder does not exist.'
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

    #Year Filter
    year_list = None
    if year_filter:
        try:
            year_list= sorted(list(util.parse_int_set(year_filter)))
            logging.info('\nyear_list = {0}'.format(year_list))
        except:
            pass

    #output folder
    output_folder_path = os.path.join(annual_ws, 'summary_shapefiles')
    if year_list:
        output_folder_path = os.path.join(annual_ws,
                            'summary_shapefiles_{}to{}'.format(min(year_list),
                                                              max(year_list)))
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Regular expressions
    data_re = re.compile('(?P<CELLID>\w+)_crop_(?P<CROP>\d+).csv$', re.I)

    # Build list of all data files
    data_file_list = sorted(
        [os.path.join(annual_ws, f_name) for f_name in os.listdir(annual_ws)
         if data_re.match(f_name)])
    if not data_file_list:
        logging.error(
            '  ERROR: No annual ET files were found\n' +
            '  ERROR: Check the folder_name parameters\n')
        sys.exit()

    #make sure lists are empty
    stations = []
    crop_nums = []

    # Process each file
    for file_path in data_file_list:
        file_name = os.path.basename(file_path)
        logging.debug('')
        logging.info('  {0}'.format(file_name))
        station, crop_num = os.path.splitext(file_name)[0].split('_crop_')
        stations.append(station)
        crop_num = int(crop_num)
        crop_nums.append(crop_num)

    # Find unique crops and station ids
    unique_crop_nums = list(set(crop_nums))
    unique_stations = list(set(stations))

    # Loop through each crop and station list to build summary dataframes for
    # variables to include in output (if not in .csv skip)
    # Should PMETo/ETr come from the .ini?
    var_list = ['ETact', 'ETpot', 'ETbas', 'Kc', 'Kcb',
                'PPT', 'Irrigation', 'Runoff', 'DPerc', 'NIWR', 'Season']
    PMET_field =  'PM{}'.format(etref_field)
    var_list.insert(0, PMET_field)
    
    # Arc fieldnames can only be 10 characters. Shorten names to include _stat
    if 'ETr' in etref_field:
        var_fieldname_list = ['ETr', 'ETact', 'ETpot', 'ETbas', 'Kc',
                    'Kcb', 'PPT', 'Irr', 'Runoff', 'DPerc', 'NIWR', 'Season']
    elif 'ETo' in etref_field:
        var_fieldname_list = ['ETo', 'ETact', 'ETpot', 'ETbas', 'Kc',
                    'Kcb', 'PPT', 'Irr', 'Runoff', 'DPerc', 'NIWR', 'Season']
    else:
        var_fieldname_list = ['ET', 'ETact', 'ETpot', 'ETbas', 'Kc',
                    'Kcb', 'PPT', 'Irr', 'Runoff', 'DPerc', 'NIWR', 'Season']   

    # Testing (should this be an input option?)
    # unique_crop_nums = [3]

    print('\n Creating Summary Shapefiles')
    for crop in unique_crop_nums:
        print('\n Processing Crop: {:02d}'.format(crop))
        #create output dataframe
        output_df = pd.DataFrame(index=unique_stations)

        for var ,var_name in zip(var_list, var_fieldname_list):
            #Initialize df variable to check if pandas df needs to be created
            df = None
            for station in unique_stations:
                #Build File Path
                file_path = os.path.join(annual_ws,
                                         '{}_crop_{:02d}.csv').format(station,
                                                                      crop)
                print(file_path)
                #Only process files that exists (crop/cell combinations)
                if not os.path.exists(file_path):
                    continue
                #Read file into df
                annual_df = pd.read_csv(file_path, skiprows=1)

                #Filter to only include years specified by user
                if year_list:
                    annual_df = annual_df[annual_df['Year'].isin(year_list)]

                #Check to see if variable is in .csv (ETr vs ETo)
                #SHOULD THIS Come FROM THE .ini?)
                if var not in annual_df.columns:
                    continue
                #Create Dataframe if it doesn't exist
                if df is None:
                   years = list(map(str, annual_df['Year']))
                   year_fieldnames =  ['Year_' + y for y in years]
                   df = pd.DataFrame(index=unique_stations,
                                     columns=year_fieldnames)
                #Write data to each station row
                df.loc[station] = list(annual_df[var])

            #Add Column of Mean and Median of All Years
            #Check to see if variablie in .csv  ETr vs ETo
            # SHOULD THIS Come FROM THE .ini?
            if var not in annual_df.columns:
                    continue

            # Median Fields
            median_fieldname = '{}_mdn'.format(var_name)
            output_df[median_fieldname] = df.median(axis=1)
            # Mean Fields
            mean_fieldname = '{}_mn'.format(var_name)
            output_df[mean_fieldname] = df.mean(axis=1)

        #Create station ID column from index (ETCells GRIDMET ID is int)
        output_df['Station'] = output_df.index.map(int)

        #Remove rows with Na (Is this the best option???)
        #Write all stations to index and then remove empty
        output_df = output_df.dropna()

        # Output file name
        out_name = "Crop_{:02d}_annual_stats.shp".format(crop)
        temp_name = "temp_annual.shp"

        # Copy ETCELLS.shp and join cropweighted data to it
        data = gpd.read_file(et_cells_path)

        # Data keep list (geometry is needed to write out as geodataframe)
        keep_list = ['geometry','GRIDMET_ID', 'LAT', 'LON', 'ELEV_M', 'ELEV_FT',
                     'COUNTYNAME', 'STATENAME', 'STPO', 'HUC8',
                     'AG_ACRES', 'CROP_{:02d}'.format(crop)]

        # Filter ETCells using keep list
        data = data[keep_list]

        # UPDATE TO NEWER ETCELLS STATION_ID FORMAT !!!!!
        merged_data = data.merge(output_df, left_on='GRIDMET_ID',
                                 right_on='Station')
        # Remove redundant Station column
        merged_data = merged_data.drop(columns='Station')
        # Write output .shp
        merged_data.to_file(os.path.join(output_folder_path, out_name),
                            driver='ESRI Shapefile')

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
        '-y', '--year', default='', type=str,
        help='Years, comma separate list and/or range')
    # parser.add_argument(
    #     '--growing_season', default=False, action='store_true',
    #     help='Run statistics on April through October dataset')
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
         cleanup_flag=args.clean, year_filter=args.year)














