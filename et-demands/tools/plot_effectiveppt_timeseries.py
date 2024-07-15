import argparse
import logging
import matplotlib.pyplot as plt
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
        print('\nSummarizing Annual Effective Precipitation')
        ws = os.path.join(project_ws, r'annual_stats')
        date_var = 'Year'
    elif time_agg == 'wateryear':
        print('\nSummarizing Water Year Effective Precipitation')
        ws = os.path.join(project_ws, r'monthly_stats')
        date_var = 'WY'
    # else:
    #     print('\nSummarizing Monthly Effective Precipitation')
    #     ws = os.path.join(project_ws, r'monthly_stats')
    #     date_var = 'Date'

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



    print('\n Reading Data and Creating Effective PPT Plots')
    for station in unique_stations:
        logging.info('\n Processing Station: {}'.format(station))
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
            # print(df.head())
            df.set_index('Year', inplace=True)

            # Filter based on Year List
            if year_list:
                df = df[df.index.isin(year_list)]

            # Min/Max for file naming
            year_min = min(df.index)
            year_max = max(df.index)

            if time_agg == 'wateryear':
                # add water year column
                df['WY'] = df.index.where(df.Month < 10, df.index + 1)
                # groupby WY (sum); select PPT variables
                df = df[['PPT', 'DPerc', 'P_rz', 'P_eft']].groupby(df.WY).sum()


                # calculate WY fractions
                df['P_rz_fraction'] = df.P_rz / df.PPT
                df['P_eft_fraction'] = df.P_eft / df.PPT
                df = df.reset_index()
                df.set_index('WY', inplace=True)
                if year_list:
                    df = df[df['WY'].isin(year_list)]
                else:
                    print('Removing first and last year of dataset to avoid partial water year totals.')
                    df = df.iloc[:-1]  # drop last year
                    df = df.iloc[1:]  # drop first year

            # print(df.head())
            # sys.exit()

            fig, ax1 = plt.subplots(1, 1, figsize=(16, 5))
            ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.


            ax1.plot(df.index, df.P_rz_fraction, color='k', linestyle='-', lw=1.5, label='P_rz')
            ax1.plot(df.index, df.P_eft_fraction, color='mediumblue', lw=1.5, linestyle='-',
                     label='P_eft')
            ax1.set_ylabel('Fraction of Effectiveness', size=12)

            ax2.bar(df.index, df.PPT, color='lightgrey', label='PPT')
            ax2.bar(df.index, df.DPerc, color='darkgrey', label='DPerc')
            ax2.set_ylabel('Annual Precipitation/Deep Percolation', size=12)

            ax1.axes.set_title('Cell: {}, Crop: {:02d}'.format(station, crop, time_agg), fontsize=12, color="black", alpha=1)

            fig.legend()
            ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
            ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2

            output_ws = os.path.join(project_ws, 'effective_ppt_plots', '{}'.format(time_agg))
            if not os.path.exists(output_ws):
                os.makedirs(output_ws)

            output_path = os.path.join(output_ws, '{}_crop_{:02d}_{}_{}_{}.png'.format(station, crop, time_agg,
                                                                                     year_min, year_max))
            # print(output_path)
            fig.savefig(output_path, dpi=300)
            plt.close()


        

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET Demands Effective PPT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-t', '--time_agg', default='annual', choices=['annual', 'wateryear'], type=str,
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
