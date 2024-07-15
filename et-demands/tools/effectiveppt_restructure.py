import argparse
import logging
import pandas as pd
import os
import re
import sys

import util


def main(ini_path, time_agg, year_filter=''):

    """Read monthly summary files and create monthly, calendar year, and water year (oct-sep)
    summary files for each crop cell combination


    Args:
        ini_path (str): file path of the project INI file
        year_filter (list): only include certain years for summary
            (single YYYY or range YYYY:YYYY)
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
        et_cells_path = config.get('CROP_ET', 'cells_path')
    except:
        logging.error(
            'et_cells_path parameter must be set in the INI file, exiting')
        return False

    if time_agg == 'annual':
        print('\n Summarizing Annual Effective Precipitation Stats')
        ws = os.path.join(project_ws, r'annual_stats')
        date_var = 'Year'
    elif time_agg == 'wateryear':
        print('\n Summarizing Water Year Effective Precipitation Stats')
        ws = os.path.join(project_ws, r'monthly_stats')
        date_var = 'WY'
    else:
        print('\n Summarizing Monthly Effective Precipitation Stats')
        ws = os.path.join(project_ws, r'monthly_stats')
        date_var = 'Date'

    # Identify unique crops and station_ids in monthly_stats folder
    # Regular expressions
    data_re = re.compile('(?P<CELLID>\w+)_crop_(?P<CROP>\d+).csv$', re.I)
    # data_re = re.compile('(?P<CELLID>\w+)_daily_crop_(?P<CROP>\d+).csv$',
    #  re.I)

    # testing
    # monthly_ws = r"D:\upper_co_full\monthly_stats"
    # et_cells_path = os.path.join('D:\upper_co_full\gis','ETCells.shp')
    # etref_field = 'ETr_ASCE'

    # Build list of all data files
    data_file_list = sorted(
        [os.path.join(ws, f_name) for f_name in os.listdir(ws)
         if data_re.match(f_name)])
    if not data_file_list:
        logging.error(
            '  ERROR: No annual ET files were found\n' +
            '  ERROR: Check the folder_name parameters\n')
        sys.exit()

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
    unique_crop_nums = sorted(list(set(crop_nums)))
    unique_stations = sorted(list(set(stations)))

    # Year Filter
    year_list = None
    if year_filter:
        try:
            year_list = sorted(list(util.parse_int_set(year_filter)))
        except:
            pass
    # Min/Max for file naming
    year_min = min(year_list)
    year_max = max(year_list)

    # Build full variable list for output order
    et_list = []
    for crop in unique_crop_nums:
        et_list.append('P_rz_{:02d}'.format(crop))
        et_list.append('P_rz_fraction_{:02d}'.format(crop))
        et_list.append('P_eft_{:02d}'.format(crop))
        et_list.append('P_eft_fraction_{:02d}'.format(crop))
    full_var_list = ['Station_ID', date_var] + ['PPT'] + et_list

    # Testing (cell with multiple crops)
    # unique_stations = [377392]
    # Loop through each station and crop list to build summary dataframes for
    print('\n Reading Data and Creating Effective PPT Files')

    out_df = pd.DataFrame(columns=full_var_list)
    for station in unique_stations:
        logging.info('\n Processing Station: {}'.format(station))
        loop_df = pd.DataFrame()
        for crop in unique_crop_nums:
            logging.info('\n Processing Crop: {:02d}'.format(crop))
            crop_vars_list = ['ETp_{:02d}'.format(crop), 'PPT', 'Season']
            # Initialize df variable to check if pandas df needs to be created
            # Build File Path
            file_path = os.path.join(ws,
                                     '{}_crop_{:02d}.csv'.format(station,
                                                                 crop))
            # Only process files that exists (crop/cell combinations)
            if not os.path.exists(file_path):
                logging.info('Crop not present in cell. Skipping')
                continue

            # Read file into df (skip header)
            df = pd.read_csv(file_path, skiprows=1)

            # Filter based on Year List
            if year_list:
                df = df[df['Year'].isin(year_list)]

            if time_agg == 'wateryear':
                # add water year column
                df['WY'] = df.Year.where(df.Month < 10, df.Year + 1)
                # groupby WY (sum); select PPT variables
                df = df[['PPT', 'P_rz', 'P_eft']].groupby(df.WY).sum()

                # calculate WY fractions
                df['P_rz_fraction'] = df.P_rz / df.PPT
                df['P_eft_fraction'] = df.P_eft / df.PPT
                df = df.reset_index()
                if year_list:
                    df = df[df['WY'].isin(year_list)]

            # Rename Columns to Match USBR Naming
            df = df.rename({'P_rz': 'P_rz_{:02d}'.format(crop),
                                            'P_eft': 'P_eft_{:02d}'.format(crop),
                                            'P_rz_fraction': 'P_rz_fraction_{:02d}'.format(crop),
                                            'P_eft_fraction': 'P_eft_fraction_{:02d}'.format(crop)}, axis='columns')
            # Add Station_ID column
            df['Station_ID'] = station

            # First pass create loop DF with PPT and Season
            if loop_df.empty:
                loop_df = df[['Station_ID', date_var, 'PPT',
                                      'P_rz_{:02d}'.format(crop),
                                      'P_rz_fraction_{:02d}'.format(crop),
                                      'P_eft_{:02d}'.format(crop),
                                      'P_eft_fraction_{:02d}'.format(crop)]]
            else:
                # After df is built merge new ET data to existing df
                # Merge on both Station_ID and Date
                loop_df = loop_df.merge(df[['Station_ID', date_var,
                                      'P_rz_{:02d}'.format(crop),
                                      'P_rz_fraction_{:02d}'.format(crop),
                                      'P_eft_{:02d}'.format(crop),
                                      'P_eft_fraction_{:02d}'.format(crop)]],
                    left_on=['Station_ID', date_var],
                    right_on=['Station_ID', date_var], how='outer')

        # Concat station_df to output df
        out_df = pd.concat([out_df, loop_df], axis=0, ignore_index=True, sort=True)
    #        df = pd.concat([df,loop_df])
        out_df = out_df.fillna(-9999)

    output_ws = os.path.join(project_ws, 'effective_ppt_stats')
    if not os.path.exists(output_ws):
        os.makedirs(output_ws)

    output_path = os.path.join(output_ws, 'effective_ppt_{}_{}_{}.csv'.format(time_agg,
        year_min, year_max))

    # Write Output File
    out_df.to_csv(output_path, sep=',', columns=full_var_list, index=False)
        

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET Demands Effective PPT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-t', '--time_agg', default='monthly', choices=['annual', 'monthly', 'wateryear'], type=str,
        help='Data output options. monthly, annual, or wateryear.')
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

    main(ini_path=args.ini, time_agg=args.time_agg, year_filter=args.year)
