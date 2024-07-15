import argparse
import datetime as dt
import logging
import os
import pandas as pd
import sys

import util


def main(ini_path, start_yr=None, end_yr=None):
    """Update MeanCutting.txt file with cutting information from annual
    stat file. Updating the MeanCutting.txt occurs after an initial "test"
    run to determine the total number of cuttings in each cell for either crop
    02 of crop 03. The model should be re-run after updating the MeanCutting.txt
    to apply the new cutting numbers.

    Args:
        ini_path (str): file path of project INI file
        start_yr (int): YYYY
        end_yr(int): YYYY

    Returns:
        None
    """

    logging.info('\nUpdating Mean Cutting File')
    logging.info('  INI: {}'.format(ini_path))

    # Check that INI file can be read
    crop_et_sec = 'CROP_ET'
    config = util.read_ini(ini_path, crop_et_sec)

    # Get project workspace and daily ET folder from INI file
    # project workspace can use old or new ini file
    try:
        project_ws = config.get('PROJECT', 'project_folder')
    except:
        try:
            project_ws = config.get(crop_et_sec, 'project_folder')
        except:
            logging.error(
                'ERROR: project_folder ' +
                'parameter is not set in INI file')
            sys.exit()

    def get_config_param(config, param_name, section):
        """"""
        try:
            param_value = config.get(section, param_name)
        except:
            logging.error(('ERROR: {} parameter is not set' +
                           ' in INI file').format(param_name))
            sys.exit()
        return param_value

    ann_stats_ws = os.path.join(project_ws, get_config_param(
            config, 'annual_output_folder', crop_et_sec))

    static_ws = os.path.join(project_ws, 'static')

    try:
        mean_cutting_name = config.get(crop_et_sec, 'cell_cuttings_name')
    except:
        logging.error('cell_cuttings_name  must be set in the INI file, '
                      'exiting')
        sys.exit()

    mean_cuttings_path = os.path.join(static_ws, mean_cutting_name)

    # Check workspaces
    if not os.path.isdir(ann_stats_ws):
        logging.error(('\nERROR: Annual ET stats folder {0} ' +
                       'could be found\n').format(ann_stats_ws))
        sys.exit()

    # Range of data to use
    try:
        year_start = start_yr
        logging.info('  Start Year:  {0}'.format(year_start))
    except:
        year_start = None
    try:
        year_end = end_yr
        logging.info('  End Year:    {0}'.format(year_end))
    except:
        year_end = None
    if year_start and year_end and year_end < year_start:
        logging.error('\n  ERROR: End Year cannot be less than start year.\n')
        sys.exit()

    if year_start and year_end:
        logging.info('\nFiltering Cutting Statistic to include data from'
                     ' {}-{}.'.format(start_yr, end_yr))

    # Loop through annual result files and update cutting numbers for
    # both crop02 and crop03 in MeanCuttings.txt file (static folder)
    cutting_crop_list = ['02', '03']
    cutting_fieldname_list = ['Number Dairy', 'Number Beef']

    # initialize mean_cutting_df
    mean_cutting_df = []

    for crop, cuttingname in zip(cutting_crop_list, cutting_fieldname_list):
        logging.info('  Processiong Crop: {}, Cutting Field: {}'.format(
            crop, cuttingname))
        mean_cutting_df = pd.read_csv(mean_cuttings_path, skiprows=[0],
                                      sep='\t')
        mean_cutting_df.set_index(['ET Cell ID'], inplace=True)
        # convert index to str to match handle all cell ID data types
        mean_cutting_df.index = mean_cutting_df.index.map(str)
        # print(mean_cutting_df.head())
        for index, row in mean_cutting_df.iterrows():
            # cell_id = row['ET Cell ID']
            # Handle both str and float/int inputs and remove .0 decimal
            # https://docs.python.org/release/2.7.3/library/stdtypes.html#boolean-operations-and-or-not
            cell_id = (str(index)[-2:] == '.0' and
                       str(index)[:-2] or str(index))
            # print(cell_id)

            stats_path = os.path.join(ann_stats_ws, '{}_crop_{}.csv'.format(
                cell_id, crop))
            # print(stats_path)
            if os.path.exists(stats_path):
                stat_df = pd.read_csv(stats_path, usecols=['Cutting', 'Year'],
                                      skiprows=[0])
            else:
                logging.debug('\nCrop {} not present in cell {}. Not updating '
                              'cuttings information.'.format(crop, cell_id))
                continue

            # Filter df based on start and end year (if given)
            if year_start and year_end:
                stat_df = stat_df.loc[(stat_df.Year >= year_start) &
                                      (stat_df.Year <= year_end)]

            # take average of all years (round down to nearest int)
            avg_cutting = int(stat_df.Cutting.mean())
            # round up to 1 if avg is < 1
            if avg_cutting < 1:
                avg_cutting = 1
            # set cuttings value in output df
            mean_cutting_df.at[cell_id, cuttingname] = avg_cutting
            # print(mean_cutting_df.head())
    logging.info('\nUpdating MeanCuttings File: {}'.format(mean_cuttings_path))
    mean_cutting_df.to_csv(mean_cuttings_path, sep='\t')
    header_line = 'This file contains first (temporary) numbers of cutting ' \
                  'cycles for dairy and beef hay, based on latitude. ' \
                  'R.Allen 4/1/08\n'
    with open(mean_cuttings_path, 'r') as original:
        data = original.read()
    with open(mean_cuttings_path, 'w') as modified:
        modified.write(header_line + data)


def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description='Update MeanCuttings.txt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '--start_yr', default=None, type=int,
        help='Start Year (format YYYY)', metavar='YEAR')
    parser.add_argument(
        '--end_yr', default=None, type=int,
        help='End Year (YYYY)', metavar='YEAR')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = parse_args()

    # Try using command line argument if it was set
    
    if args.ini:
        ini_path = args.ini
        
    # If script was double clicked, set project folder with GUI
    
    elif 'PROMPT' not in os.environ:
        ini_path = util.get_path(os.getcwd(), 'Select target INI file')

    # Try using current working directory if there is only one INI
    # Could look for daily_stats folder, run_basin.py, and/or ini file

    elif len([x for x in os.listdir(os.getcwd()) if
              x.lower().endswith('.ini')]) == 1:
        ini_path = [
            os.path.join(os.getcwd(), x) for x in os.listdir(os.getcwd())
            if x.lower().endswith('.ini')][0]
    else:
        ini_path = util.get_path(os.getcwd(), 'Select target INI file')
    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{0}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path, start_yr=args.start_yr, end_yr=args.end_yr)
