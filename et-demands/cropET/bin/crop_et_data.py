"""crop_et_data.py
Defines CropETData class
Functions in class to read INI file, refet data, met data, ...
Called by mod_crop_et.py

"""

import configparser
import logging
import os
import pandas as pd
import sys

import crop_coefficients
import crop_parameters
import util


class CropETData:
    """Crop et data container

    Attributes
    ----------

    """

    def __init__(self):
        """ """

    def __str__(self):
        """ """
        return '<Cropet_data>'

    def read_cet_ini(self, ini_path, debug_flag=False):
        """Read and parse INI file
        Parameters
        ---------
        ini_path : str
            absolute file path to INI file
        debug_flag : boolean
            True : write debug level comments to debug.txt
            False

        Returns
        -------

        Notes
        -----

        """
        logging.info('  INI: {}'.format(os.path.basename(ini_path)))

        # Check that INI file can be read
        config = configparser.RawConfigParser()
        try:
            config.read_file(open(ini_path))
        except:
            logging.error('\nERROR: Config file could not be read, ' +
                          'is not an input file, or does not exist\n')
            sys.exit()

        # Check that all sections are present
        crop_et_sec = 'CROP_ET'  # required
        weather_sec = 'WEATHER'  # required
        refet_sec = 'REFET'  # required
        # Constant phenology section
        hist_temps_sec = 'HIST'  # optional

        cfg_secs = config.sections()

        # Original INI Format (Add phenology (hist_temps_sec) to original INI)
        original = True

        # verify existence of common required sections
        if crop_et_sec not in cfg_secs or refet_sec not in cfg_secs or \
                weather_sec not in cfg_secs:
            logging.error(
                '\nERROR:input file must have following sections:\n' +
                '  [{}], [{}], and [{}]'.format(crop_et_sec, weather_sec,
                                                refet_sec))
        # Check if phenology section is present ('HIST'); Else NONE
        if hist_temps_sec not in cfg_secs:
            hist_temps_sec = None

        # project specifications
        try:
            self.project_ws = config.get(crop_et_sec, 'project_folder')
        except:
            logging.error(
                'ERROR: project_folder ' +
                'parameter is not set in INI file')
            sys.exit()

        # project folder needs to be full/absolute path
        # check existence
        if not os.path.isdir(self.project_ws):
            logging.critical(
                'ERROR:project folder does not exist\n  %s' % self.project_ws)
            sys.exit()

        # Basin
        try:
            self.basin_id = config.get(crop_et_sec, 'basin_id')
            if self.basin_id is None or self.basin_id == 'None':
                self.basin_id = 'Default Basin'
        except:
            self.basin_id = 'Default Basin'
        logging.info('  Basin: {}'.format(self.basin_id))

        # Timestep -
        self.time_step = 'day'

        # Timestep quantity - specify an integer
        self.ts_quantity = int(1)

        # user starting date
        try:
            sdt = config.get(crop_et_sec, 'start_date')
            if sdt == 'None':
                sdt = None
        except:
            sdt = None
        if sdt is None:
            self.start_dt = None
        else:
            self.start_dt = pd.to_datetime(sdt)

        # ending date
        try:
            edt = config.get(crop_et_sec, 'end_date')
            if edt == 'None': edt = None
        except:
            edt = None
        if edt is None:
            self.end_dt = None
        else:
            self.end_dt = pd.to_datetime(edt)

        """
        INI [HIST] Section

        """
        if hist_temps_sec:
            try:
                self.phenology_option = config.getint(
                    hist_temps_sec, 'phenology_option')
                if self.phenology_option is None or \
                        self.phenology_option == 'None':
                    self.phenology_option = 0
            except:
                self.phenology_option = 0
        else:
            self.phenology_option = 0

        # static (aka) meta data specifications
        try:
            self.static_folder = config.get(crop_et_sec, 'static_folder')
            if self.static_folder is None or self.static_folder == 'None':
                logging.warning("Static workspace set to default 'static'")
                self.static_folder = 'static'
        except:
            logging.warning("Static workspace set to default 'static'")
            self.static_folder = 'static'
        if not os.path.isdir(self.static_folder):
            self.static_folder = os.path.join(self.project_ws,
                                              self.static_folder)

        # elevation units
        try:
            self.elev_units = config.get(crop_et_sec, 'elev_units')
            if self.elev_units is None or self.elev_units == 'None':
                self.elev_units = 'feet'
        except:
            self.elev_units = 'feet'

        # et cells properties

        try:
            cell_properties_name = config.get(crop_et_sec,
                                              'cell_properties_name')
            if cell_properties_name is None or cell_properties_name == 'None':
                logging.error('ERROR: ET Cells properties data file must be'
                              ' specified')
                sys.exit()
        except:
            logging.error('ERROR:  ET Cells properties data file must be'
                          ' specified')
            sys.exit()

        # test joined path
        self.cell_properties_path = os.path.join(self.static_folder,
                                                 cell_properties_name)
        if not os.path.isfile(self.cell_properties_path):
            self.cell_properties_path = cell_properties_name

            # test if fully specified path
            if not os.path.isfile(self.cell_properties_path):
                logging.error('ERROR:  ET Cells properties file {} does not'
                              ' exist'.format(self.cell_properties_path))
                sys.exit()
        logging.info('  ET Cell Properties file: {}'
                     .format(self.cell_properties_path))

        self.cell_properties_delimiter = '\t'
        self.cell_properties_ws = ''
        self.cell_properties_names_line = 1
        self.cell_properties_header_lines = 1

        # et cells crops
        try:
            cell_crops_name = config.get(crop_et_sec, 'cell_crops_name')
            if cell_crops_name is None or cell_crops_name == 'None':
                logging.error('ERROR:  ET Cells crops data file must be'
                              ' specified')
                sys.exit()
        except:
            logging.error('ERROR:  ET Cells crops data file must be specified')
            sys.exit()
        self.cell_crops_path = os.path.join(self.static_folder, cell_crops_name)
        if not os.path.isfile(self.cell_crops_path):
            self.cell_crops_path = cell_crops_name
            if not os.path.isfile(self.cell_crops_path):
                logging.error('ERROR:  ET Cells crops file {} does not exist'
                              .format(self.self.cell_crops_path))
                sys.exit()
        logging.info('  ET Cell crops file: {}'.format(self.cell_crops_path))

        self.cell_crops_delimiter = '\t'
        self.cell_crops_ws = ''
        self.cell_crops_header_lines = 3
        self.cell_crops_names_line = 2

        # et cells cuttings
        try:
            cell_cuttings_name = config.get(crop_et_sec, 'cell_cuttings_name')
            if cell_cuttings_name is None or cell_cuttings_name == 'None':
                logging.error('ERROR:  ET Cells cuttings data file must be'
                              ' specified')
                sys.exit()
        except:
            logging.error('ERROR:  ET Cells cuttings data file must be'
                          ' specified')
            sys.exit()
        self.cell_cuttings_path =\
            os.path.join(self.static_folder, cell_cuttings_name)
        if not os.path.isfile(self.cell_cuttings_path):
            self.cell_cuttings_path = cell_cuttings_name
            if not os.path.isfile(self.cell_cuttings_path):
                logging.error('ERROR:  ET Cells cuttings file {} does not'
                              ' exist'.format(self.cell_cuttings_path))
                sys.exit()
        logging.info('  ET Cell cuttings file: {}'
                     .format(self.cell_cuttings_path))

        self.cell_cuttings_delimiter = '\t'
        self.cell_cuttings_ws = ''
        self.cell_cuttings_header_lines = 2
        self.cell_cuttings_names_line = 2

        # set crop parameter specs
        try:
            crop_params_name = \
                config.get(crop_et_sec, 'crop_params_name')
            if crop_params_name is None or crop_params_name == 'None':
                logging.error('ERROR:  Crop parameters data file must be'
                              ' specified')
                sys.exit()
        except:
            logging.error('ERROR:  Crop parameters data file must be specified')
            sys.exit()
        self.crop_params_path = \
            os.path.join(self.static_folder, crop_params_name)
        if not os.path.isfile(self.crop_params_path):
            self.crop_params_path = crop_params_name
            if not os.path.isfile(self.crop_params_path):
                logging.error('ERROR:  crop parameters file {} does not exist'
                              .format(self.crop_params_path))
                sys.exit()
        logging.info('  Crop parameters file: {}'.format(self.crop_params_path))
        if original:
            self.crop_params_delimiter = '\t'
            self.crop_params_ws = ''
            self.crop_params_header_lines = 4
            self.crop_params_names_line = 3

        # set crop coefficient specs
        try:
            crop_coefs_name = config.get(crop_et_sec, 'crop_coefs_name')
            if crop_coefs_name is None or crop_coefs_name == 'None':
                logging.error('ERROR:  Crop coefficients data file must be'
                              ' specified')
                sys.exit()
        except:
            logging.error('ERROR:  Crop coefficients data file must be'
                          ' specified')
            sys.exit()
        self.crop_coefs_path = os.path.join(self.static_folder, crop_coefs_name)
        if not os.path.isfile(self.crop_coefs_path):
            self.crop_coefs_path = crop_coefs_name
            if not os.path.isfile(self.crop_coefs_path):
                logging.error('ERROR:  crop coefficients file {} does not'
                              ' exist'.format(self.self.crop_coefs_path))
                sys.exit()
        logging.info('  Crop coefficients file: {}'
                     .format(self.crop_coefs_path))

        self.crop_coefs_delimiter = '\t'
        self.crop_coefs_ws = ''
        self.crop_coefs_names_line = 1
        self.crop_coefs_header_lines = 1

        # reference ET adjustment ratios
        try:
            et_ratios_name = config.get(crop_et_sec,
                                         'et_ratios_name')
            self.refet_ratios_path = os.path.join(self.static_folder,
                                                  et_ratios_name)
        except configparser.NoOptionError:
            logging.info('\net_ratios_name not found in INI. Setting to None.')
            self.refet_ratios_path = None

        if self.refet_ratios_path and not os.path.isfile(
                self.refet_ratios_path):
            logging.error('Warning:  ET Ratios file not found. ET scaling will not be applied.')
            self.refet_ratios_path = None

        # Default et_ratios file format
        self.et_ratios_delimiter = '\t'
        self.et_ratios_ws = ''
        self.et_ratios_header_lines = 1
        self.et_ratios_names_line = 1
        self.et_ratios_id_field = 'Met Node ID'
        self.et_ratios_name_field = 'Met Node Name'
        self.et_ratios_month_field = 'month'
        self.et_ratios_ratio_field = 'ratio'

        """
        INI [CROP_ET] Section

        """
        # cet output flags
        self.cet_out = {}
        try:
            self.cet_out['daily_output_flag'] = \
                config.getboolean(crop_et_sec, 'daily_stats_flag')
        except:
            logging.debug('    daily_stats_flag = False')
            self.cet_out['daily_output_flag'] = False
        try:
            self.cet_out['monthly_output_flag'] = \
                config.getboolean(crop_et_sec, 'monthly_stats_flag')
        except:
            logging.debug('    monthly_stats_flag = False')
            self.cet_out['monthly_output_flag'] = False
        try:
            self.cet_out['annual_output_flag'] = \
                config.getboolean(crop_et_sec, 'annual_stats_flag')
        except:
            logging.debug('    annual_stats_flag = False')
            self.cet_out['annual_output_flag'] = False
        try:
            self.gs_output_flag = \
                config.getboolean(crop_et_sec, 'growing_season_stats_flag')
        except:
            logging.debug('    growing_season_stats_flag = False')
            self.gs_output_flag = False

        #  Allow user to only run annual or perennial crops
        try:
            self.annual_skip_flag = \
                config.getboolean(crop_et_sec, 'annual_skip_flag')
        except:
            logging.info('    annual_skip_flag = False')
            self.annual_skip_flag = False
        try:
            self.perennial_skip_flag = \
                config.getboolean(crop_et_sec, 'perennial_skip_flag')
        except:
            logging.info('    perennial_skip_flag = False')
            self.perennial_skip_flag = False

        # For testing, allow user to process a subset of crops
        try:
            self.crop_skip_list = list(util.parse_int_set(
                config.get(crop_et_sec, 'crop_skip_list')))
        except:
            logging.debug('    crop_skip_list = []')
            self.crop_skip_list = []
        try:
            self.crop_test_list = list(util.parse_int_set(
                config.get(crop_et_sec, 'crop_test_list')))
        except:
            logging.debug('    crop_test_list = False')
            self.crop_test_list = []

        # Bare soils must be in crop list for computing winter cover
        if self.crop_test_list:
            self.crop_test_list = sorted(list(set(
                self.crop_test_list + [44, 45, 46])))

        # For testing, allow the user to process a subset of cells
        try:
            self.cell_skip_list = config.get(
                crop_et_sec, 'cell_skip_list').split(',')
            self.cell_skip_list = [c.strip() for c in self.cell_skip_list]
        except:
            logging.debug('    cell_skip_list = []')
            self.cell_skip_list = []
        try:
            self.cell_test_list = config.get(
                crop_et_sec, 'cell_test_list').split(',')
            self.cell_test_list = [c.strip() for c in self.cell_test_list]
        except:
            logging.debug('    cell_test_list = False')
            self.cell_test_list = []

        # Input/output folders
        if self.cet_out['daily_output_flag']:
            try:
                self.cet_out['daily_output_ws'] = os.path.join(
                    self.project_ws,
                    config.get(crop_et_sec, 'daily_output_folder'))
                if not os.path.isdir(self.cet_out['daily_output_ws']):
                   os.makedirs(self.cet_out['daily_output_ws'])
            except:
                logging.debug('    daily_output_folder = daily_stats')
                self.cet_out['daily_output_ws'] = 'daily_stats'
        if self.cet_out['monthly_output_flag']:
            try:
                self.cet_out['monthly_output_ws'] = os.path.join(
                    self.project_ws,
                    config.get(crop_et_sec, 'monthly_output_folder'))
                if not os.path.isdir(self.cet_out['monthly_output_ws']):
                   os.makedirs(self.cet_out['monthly_output_ws'])
            except:
                logging.debug('    monthly_output_folder = monthly_stats')
                self.cet_out['monthly_output_ws'] = 'monthly_stats'
        if self.cet_out['annual_output_flag']:
            try:
                self.cet_out['annual_output_ws'] = os.path.join(
                    self.project_ws,
                    config.get(crop_et_sec, 'annual_output_folder'))
                if not os.path.isdir(self.cet_out['annual_output_ws']):
                   os.makedirs(self.cet_out['annual_output_ws'])
            except:
                logging.debug('    annual_output_folder = annual_stats')
                self.cet_out['annual_output_ws'] = 'annual_stats'
        if self.gs_output_flag:
            try:
                self.gs_output_ws = os.path.join(
                    self.project_ws,
                    config.get(crop_et_sec, 'gs_output_folder'))
                if not os.path.isdir(self.gs_output_ws):
                   os.makedirs(self.gs_output_ws)
            except:
                logging.debug('    gs_output_folder = growing_season_stats')
                self.gs_output_ws = 'growing_season_stats'

        # cet file type specifications
        self.cet_out['file_type'] = 'csv'
        # self.cet_out['data_structure_type'] = "DRI"
        self.cet_out['name_format'] = '%s_crop_%c.csv'
        self.cet_out['header_lines'] = 1
        self.cet_out['names_line'] = 1
        self.cet_out['delimiter'] = ','

        # pick up user growing season file specifications
        if self.gs_output_flag:
            self.gs_name_format = self.cet_out['name_format'].replace(
                '%s', '%s_gs')

        """
        Computational switches
            False : sets crop 1 to alfalfa peak with no cuttings
            True : sets crop 1 to nonpristine alfalfa w/cuttings

        """
        try:
            self.crop_one_flag = config.getboolean(
                crop_et_sec, 'crop_one_flag')
        except:
            self.crop_one_flag = True

        # crop one (alfalfa) reduction factor
        try:
            self.crop_one_reducer = config.getfloat(crop_et_sec,
                                                    'crop_one_reducer')
        except:
            self.crop_one_reducer = 0.9

        # Compute additional variables
        try:
            self.cutting_flag = config.getboolean(crop_et_sec,
                                                  'cutting_flag')
        except:
            self.cutting_flag = True
        try:
            self.niwr_flag = config.getboolean(crop_et_sec, 'niwr_flag')
        except:
            self.niwr_flag = True
        try:
            self.kc_flag = config.getboolean(crop_et_sec, 'kc_flag')
        except:
            self.kc_flag = True
        try:
            self.co2_flag = config.getboolean(crop_et_sec, 'co2_flag')
        except:
            self.co2_flag = False

        # added 8/2020 to allow for growing season start doy to progress unchecked (no limit)
        try:
            self.gs_limit_flag = config.getboolean(crop_et_sec, 'gs_limit_flag')
        except:
            self.gs_limit_flag = True


        # Spatially varying calibration
        try: self.spatial_cal_flag = config.getboolean(crop_et_sec,
                                                       'spatial_cal_flag')
        except: self.spatial_cal_flag = False
        try:
            self.spatial_cal_ws = config.get(crop_et_sec, 'spatial_cal_folder')
        except:
            self.spatial_cal_ws = None
        if (self.spatial_cal_flag and self.spatial_cal_ws is not None and
            not os.path.isdir(self.spatial_cal_ws)):
            logging.error(('ERROR:spatial calibration folder {} ' +
                           'does not exist').format(self.spatial_cal_ws))
            sys.exit()

        # output date formats and values formats
        try:
            self.cet_out['daily_date_format'] = config.get(
                crop_et_sec, 'daily_date_format')
            if self.cet_out['daily_date_format'] is None or \
                    self.cet_out['daily_date_format'] == 'None':
                self.cet_out['daily_date_format'] = '%Y-%m-%d'
        except:
            self.cet_out['daily_date_format'] = '%Y-%m-%d'
        try:
            self.cet_out['daily_float_format'] = config.get(
                crop_et_sec, 'daily_float_format')
            if self.cet_out['daily_float_format'] == 'None':
                self.cet_out['daily_float_format'] = None
        except:
            self.cet_out['daily_float_format'] = None
        try:
            self.cet_out['monthly_date_format'] = config.get(
                crop_et_sec, 'monthly_date_format')
            if self.cet_out['monthly_date_format'] is None or \
                    self.cet_out['monthly_date_format'] == 'None':
                self.cet_out['monthly_date_format'] = '%Y-%m'
        except:
            self.cet_out['monthly_date_format'] = '%Y-%m'
        try:
            self.cet_out['monthly_float_format'] = config.get(
                crop_et_sec, 'monthly_float_format')
            if self.cet_out['monthly_float_format'] == 'None':
                self.cet_out['monthly_float_format'] = None
        except:
            self.cet_out['monthly_float_format'] = None
        try:
            self.cet_out['annual_date_format'] = config.get(
                crop_et_sec, 'annual_date_format')
            if self.cet_out['monthly_date_format'] is None or \
                    self.cet_out['monthly_date_format'] == 'None':
                self.cet_out['annual_date_format'] = '%Y'
        except:
            self.cet_out['annual_date_format'] = '%Y'
        try:
            self.cet_out['annual_float_format'] = \
                config.get(crop_et_sec, 'annual_float_format')
            if self.cet_out['annual_float_format'] == 'None':
                self.cet_out['annual_float_format'] = None
        except:
            self.cet_out['annual_float_format'] = None

        """
        INI [REFET] Section

        """
        self.refet = {}
        self.refet['fields'] = {}
        self.refet['units'] = {}
        self.refet['ws'] = config.get(refet_sec, 'refet_folder')
        # refet folder could be full or relative path
        # Assume relative paths or from project folder

        if os.path.isdir(self.refet['ws']):
            pass
        elif (not os.path.isdir(self.refet['ws']) and
              os.path.isdir(os.path.join(self.project_ws, self.refet['ws']))):
            self.refet['ws'] = os.path.join(self.project_ws, self.refet['ws'])
        else:
            logging.error('ERROR:refet folder {} does not exist'
                          .format(self.refet['ws']))
            sys.exit()

        # DEADBEEF
        # self.refet['ws'] = os.path.join(
        #     .project_ws, config.get(refet_sec, 'refet_folder'))
        self.refet['type'] = config.get(refet_sec, 'refet_type').lower()
        if self.refet['type'] not in ['eto', 'etr']:
            logging.error('  ERROR: RefET type must be ETo or ETr')
            sys.exit()

        self.refet['file_type'] = 'csv'
        # self.refet['data_structure_type'] = 'SF P'
        self.refet['name_format'] = config.get(refet_sec, 'name_format')
        self.refet['header_lines'] = config.getint(refet_sec, 'header_lines')
        self.refet['names_line'] = config.getint(refet_sec, 'names_line')
        try:
            self.refet['delimiter'] = config.get(refet_sec, 'delimiter')
            if self.refet['delimiter'] is None or \
                    self.refet['delimiter'] == 'None':
                self.refet['delimiter'] = ','
            else:
                if self.refet['delimiter'] not in [' ', ',', '\\t']:
                    self.refet['delimiter'] = ','
                if "\\" in self.refet['delimiter'] and \
                        "t" in self.refet['delimiter']:
                    self.refet['delimiter'] = \
                        self.refet['delimiter'].replace('\\t', '\t')
        except:
                self.refet['delimiter'] = ','

        # Field names and units
        # Date can be read directly or computed from year, month, and day
        try:
            self.refet['fields']['date'] = config.get(refet_sec, 'date_field')
        except:
            self.refet['fields']['date'] = None
        try:
            self.refet['fields']['year'] = config.get(refet_sec, 'year_field')
            self.refet['fields']['month'] = config.get(refet_sec, 'month_field')
            self.refet['fields']['day'] = config.get(refet_sec, 'day_field')
        except:
            self.refet['fields']['year'] = None
            self.refet['fields']['month'] = None
            self.refet['fields']['day'] = None
        if self.refet['fields']['date'] is not None:
            logging.debug('  REFET: Reading date from date column')
        elif (self.refet['fields']['year'] is not None and
              self.refet['fields']['month'] is not None and
              self.refet['fields']['day'] is not None):
            logging.debug('\nREFET: Reading date from year, month,'
                          ' and day columns')
        else:
            logging.error('  ERROR: REFET date_field (or year, month, and ' +
                          'day fields) must be set in INI')
            sys.exit()
        try:
            self.refet['fields']['etref'] = config.get(refet_sec, 'etref_field')
        except:
            logging.error('  ERROR: REFET etref_field must set in INI')
            sys.exit()
        try:
            self.refet['fnspec'] = config.get(refet_sec, 'etref_name')
            if self.refet['fnspec'] is None or self.refet['fnspec'] == 'None':
                self.refet['fnspec'] = self.refet['fields']['etref']
        except:
            self.refet['fnspec'] = self.refet['fields']['etref']
        try:
           self.refet['units']['etref'] = config.get(refet_sec, 'etref_units')
        except:
            logging.error('  ERROR: REFET etref_units must set in INI')
            sys.exit()

        # Check RefET parameters
        if not os.path.isdir(self.refet['ws']):
            logging.error(
                ('  ERROR:RefET data folder does not ' +
                 'exist\n  %s') % self.refet['ws'])
            sys.exit()

        # Check fields and units
        elif self.refet['units']['etref'].lower() not in ['mm/day', 'mm']:
            logging.error(
                '  ERROR:  Ref ET units {0} are not currently '
                'supported'.format(self.refet['units']['etref']))
            sys.exit()

        """
        INI [WEATHER] Section

        """
        # Weather parameters
        self.weather = {}
        self.weather['fields'] = {}
        self.weather['units'] = {}

        # fnspec - parameter extension to file name specification
        self.weather['fnspec'] = {}
        self.weather['wsspec'] = {}
        self.weather['ws'] = config.get(weather_sec, 'weather_folder')

        # weather folder could befull or relative path
        # Assume relative paths or fromproject folder
        if os.path.isdir(self.weather['ws']):
            pass
        elif (not os.path.isdir(self.weather['ws']) and
              os.path.isdir(os.path.join(self.project_ws,
                                         self.weather['ws']))):
            self.weather['ws'] = os.path.join(self.project_ws,
                                              self.weather['ws'])
        else:
            logging.error('ERROR:refet folder {} does not exist'.format(
                self.weather['ws']))
            sys.exit()

        self.weather['file_type'] = 'csv'
         # self.weather['data_structure_type'] = 'SF P'
        self.weather['name_format'] = config.get(weather_sec, 'name_format')
        self.weather['header_lines'] = config.getint(weather_sec,
                                                     'header_lines')
        self.weather['names_line'] = config.getint(weather_sec, 'names_line')
        try:
            self.weather['delimiter'] = config.get(weather_sec, 'delimiter')
            if self.weather['delimiter'] is None or \
                    self.weather['delimiter'] == 'None':
                self.weather['delimiter'] = ','
            else:
                if self.weather['delimiter'] not in [' ', ',', '\\t']:
                    self.weather['delimiter'] = ','
                if "\\" in self.weather['delimiter'] and \
                        "t" in self.weather['delimiter']:
                    self.weather['delimiter'] = \
                        self.weather['delimiter'].replace('\\t', '\t')
        except:
                self.weather['delimiter'] = ','

        # Field names and units
        # Date can be read directly or computed from year, month, and day
        try:
            self.weather['fields']['date'] = \
                config.get(weather_sec, 'date_field')
        except:
            self.weather['fields']['date'] = None
        try:
            self.weather['fields']['year'] = config.get(weather_sec,
                                                        'year_field')
            self.weather['fields']['month'] = config.get(weather_sec,
                                                         'month_field')
            self.weather['fields']['day'] = config.get(weather_sec,
                                                       'day_field')
        except:
            self.weather['fields']['year'] = None
            self.weather['fields']['month'] = None
            self.weather['fields']['day'] = None
        if self.weather['fields']['date'] is not None:
            logging.debug('  WEATHER: Reading date from date column')
        elif (self.weather['fields']['year'] is not None and
              self.weather['fields']['month'] is not None and
              self.weather['fields']['day'] is not None):
            logging.debug('  WEATHER: Reading date from year, month,'
                          ' and day columns')
        else:
            logging.error('  ERROR: WEATHER date_field (or year, month, and ' +
                          'day fields) must be set in INI')
            sys.exit()

        # Field names
        # Following fields are mandatory
        # DEADBEEF - Are snow and snow depth required?
        # Add check for (tdew or ea or q)
        field_list = ['tmin', 'tmax', 'ppt', 'wind']
        for f_name in field_list:
            try:
                self.weather['fields'][f_name] = config.get(
                    weather_sec, f_name + '_field')
            except:
                logging.error(('  ERROR: WEATHER {}_field must be set ' +
                     'in INI').format(f_name))
                sys.exit()

        # Units
        for f_name in field_list:
            if f_name == 'date':
                continue
            elif self.weather['fields'][f_name] is not None:
                try:
                    self.weather['units'][f_name] = config.get(
                        weather_sec, f_name + '_units')
                except:
                    logging.error(
                        ('  ERROR: WEATHER {}_units must be set ' +
                         'in INI').format(f_name))
                    sys.exit()

        # fnspec
        for f_name in field_list:
            if f_name == 'date':
                continue
            elif self.weather['fields'][f_name] is not None:
                try:
                    self.weather['fnspec'][f_name] = config.get(
                        weather_sec, f_name + '_name')
                except:
                    self.weather['fnspec'][f_name] = f_name
            else:
                self.weather['fnspec'][f_name] = 'Unused'

        # Snow and snow depth are optional
        try:
            self.weather['fields']['snow'] = config.get(weather_sec,
                                                        'snow_field')
            if self.weather['fields']['snow'] is None or \
                    self.weather['fields']['snow'] == 'None':
                self.weather['fields']['snow'] = 'Snow'
                self.weather['units']['snow'] = 'mm/day'
                self.weather['fnspec']['snow'] = 'Estimated'
            else:
                try: self.weather['units']['snow'] = config.get(weather_sec,
                                                                'snow_units')
                except: self.weather['units']['snow'] = 'mm/day'
                try: self.weather['fnspec']['snow'] = config.get(weather_sec,
                                                                 'snow_name')
                except: self.weather['fnspec']['snow'] = \
                    self.weather['fields']['snow']
        except:
            self.weather['fields']['snow'] = 'Snow'
            self.weather['units']['snow'] = 'mm/day'
            self.weather['fnspec']['snow'] = 'Estimated'

        try:
            self.weather['fields']['snow_depth'] = config.get(weather_sec,
                                                              'depth_field')
            if self.weather['fields']['snow_depth'] is None or \
                    self.weather['fields']['snow_depth'] == 'None':
                self.weather['fields']['snow_depth'] = 'SDep'
                self.weather['units']['snow_depth'] = 'mm'
                self.weather['fnspec']['snow_depth'] = 'Estimated'
            else:
                try:
                    self.weather['units']['snow_depth'] = \
                    config.get(weather_sec, 'depth_units')
                except:
                    self.weather['units']['snow_depth'] = 'mm'
                try:
                    self.weather['fnspec']['snow_depth'] = \
                    config.get(weather_sec, 'depth_name')
                except:
                    self.weather['fnspec']['snow_depth'] = \
                    self.weather['fields']['snow_depth']
        except:
            self.weather['fields']['snow_depth'] = 'SDep'
            self.weather['units']['snow_depth'] = 'mm'
            self.weather['fnspec']['snow_depth'] = 'Estimated'

        # Dewpoint temperature can be set or computed from ea or q
        # Field that is provided is used to estimate rh_min for kc calcs
        for x in ['tdew', 'ea', 'q']:
            self.weather['fields'][x] = None
            self.weather['units'][x] = None
            # Check if field exists
            try:
                self.weather['fields'][x] = config.get(weather_sec, x +
                                                       '_field')
            except configparser.NoOptionError:
                logging.info(x + ' field not found in .ini. skipping.')
                continue
            # Check if field is None or ''
            if self.weather['fields'][x] is None or self.weather['fields'][
                    x].lower() in ['', 'None']:
                logging.info(x + 'field was not set or was set to none')
                continue
            # If field exists check for units
            if self.weather['fields'][x]:
                try:
                    self.weather['units'][x] = config.get(weather_sec,
                                                          x + '_units')
                except configparser.NoOptionError:
                    logging.info(x + ' units not found in .ini. skipping')
                    continue
            # Check if units is None or ''
            if self.weather['units'][x] is None or self.weather['units'][
                    x].lower() in ['', 'None']:
                logging.debug(x + 'units was not set or was set to none')
                continue
            # If field and units exist; add fnspec (field for now)
            # What is fnspec used for?
            if self.weather['fields'][x] and self.weather['units'][x]:
                self.weather['fnspec'][x] = self.weather['fields'][x]
                break

        # Check that at least one atmospheric moisture option is present in INI
        if not ((self.weather['fields']['tdew'] and
                 self.weather['units']['tdew']) or
                (self.weather['fields']['ea'] and
                 self.weather['units']['ea']) or
                (self.weather['fields']['q'] and
                 self.weather['units']['q'])):
            logging.error('tdew, ea, or q field and units must be specific in'
                          ' INI WEATHER section. \nExiting.')
            sys.exit()

        # CO2 correction factors are optional (default to None)
        self.weather['fields']['co2_grass'] = None
        self.weather['fields']['co2_tree'] = None
        self.weather['fields']['co2_c4'] = None
        self.weather['units']['co2_grass'] = None
        self.weather['units']['co2_tree'] = None
        self.weather['units']['co2_c4'] = None

        if self.co2_flag:
            logging.info('  CO2 correction')

            # For now, CO2 values in table will not be error checked

            # Get CO2 fields
            try:
                self.weather['fields']['co2_grass'] = \
                    config.get(weather_sec, 'co2_grass_field')
            except:
                self.weather['fields']['co2_grass'] = None
            try:
                self.weather['fields']['co2_tree'] = \
                    config.get(weather_sec, 'co2_tree_field')
            except:
                self.weather['fields']['co2_tree'] = None
            try:
                self.weather['fields']['co2_c4'] = \
                    config.get(weather_sec, 'co2_c4_field')
            except:
                self.weather['fields']['co2_c4'] = None

            # Check that at least one CO2 field was set in INI
            if (not self.weather['fields']['co2_grass'] and
                    not self.weather['fields']['co2_tree'] and
                    not self.weather['fields']['co2_c4']):
                logging.error(
                    '  ERROR: WEATHER CO2 field names must be set in ' +
                    'the INI if co2_flag = True')
                sys.exit()

        # CO2 correction factors are optional (default to None)
        self.weather['fields']['co2_grass'] = None
        self.weather['fields']['co2_tree'] = None
        self.weather['fields']['co2_c4'] = None
        self.weather['units']['co2_grass'] = None
        self.weather['units']['co2_tree'] = None
        self.weather['units']['co2_c4'] = None

        if self.co2_flag:
            logging.info('  CO2 correction')
            # For now, CO2 values in table will not be error checked

            # Get CO2 fields
            try:
                self.weather['fields']['co2_grass'] = config.get(
                    weather_sec, 'co2_grass_field')
            except:
                self.weather['fields']['co2_grass'] = None
            try:
                self.weather['fields']['co2_tree'] = config.get(
                    weather_sec, 'co2_tree_field')
            except:
                self.weather['fields']['co2_tree'] = None
            try:
                self.weather['fields']['co2_c4'] = config.get(
                    weather_sec, 'co2_c4_field')
            except:
                self.weather['fields']['co2_c4'] = None

            # Check that at least one CO2 field was set in INI
            if (not self.weather['fields']['co2_grass'] and
                    not self.weather['fields']['co2_tree'] and
                    not self.weather['fields']['co2_c4']):
                logging.error(
                    '  ERROR: WEATHER CO2 field names must be set in ' +
                    'the INI if co2_flag = True')
                sys.exit()

            # Get crop lists for each CO2 class
            try:
                self.co2_grass_crops = sorted(list(util.parse_int_set(
                    config.get(crop_et_sec, 'co2_grass_list'))))
            except:
                self.co2_grass_crops = []
                # # DEADBEEF - Make these the defaults?
                # self.co2_grass_crops = (
                #     1,6+1) + range(9,18+1) + range(21,67+1) +
                #     69,71,72,73,75,79,80,81,83,84,85])
            try:
                self.co2_tree_crops = sorted(list(util.parse_int_set(
                    config.get(crop_et_sec, 'co2_tree_list'))))
            except:
                self.co2_tree_crops = []
                # # DEADBEEF - Make these the defaults?
                # self.co2_tree_crops = [19, 20, 70, 74, 82]
            try:
                self.co2_c4_crops = sorted(list(util.parse_int_set(
                    config.get(crop_et_sec, 'co2_c4_list'))))
            except:
                self.co2_c4_crops = []
                # # DEADBEEF - Make these the defaults?
                # self.co2_c4_crops = [7, 8, 68, 76-78]
            logging.info('    Grass (C3): {}'.format(self.co2_grass_crops))
            logging.info('    Trees (C3): {}'.format(self.co2_tree_crops))
            logging.info('    C4: {}'.format(self.co2_c4_crops))

            # Check if data fields are present for all CO2 classes with crops
            if (self.co2_grass_crops and
                    not self.weather['fields']['co2_grass']):
                logging.error(
                    '  ERROR: WEATHER CO2 grass field name is not set in ' +
                    ' INI file but CO2 grass crops are listed')
                sys.exit()
            elif (self.co2_tree_crops and
                    not self.weather['fields']['co2_tree']):
                logging.error(
                    '  ERROR: WEATHER CO2 tree field name is not set in ' +
                    ' INI file but CO2 tree crops are listed')
                sys.exit()
            elif (self.co2_c4_crops and
                    not self.weather['fields']['co2_c4']):
                logging.error(
                    '  ERROR: WEATHER CO2 C4 field name is not set in ' +
                    ' INI file but CO2 C4 crops are listed')
                sys.exit()

        # Wind speeds measured at heights other than 2 meters will be scaled
        try:
            self.weather['wind_height'] = config.getfloat(weather_sec,
                                                          'wind_height')
        except:
            self.weather['wind_height'] = 2

        # Check weather parameters
        if not os.path.isdir(self.weather['ws']):
            logging.error(
                ('  ERROR:weather data folder does not exist'
                 '\n  %s') % self.weather['ws'])
            sys.exit()

        # REPLACE WITH CODE IN units.py TAKEN FROM WSWUP/REFET
        # Check units
        units_list = ['c', 'mm', 'mm/d', 'mm/day', 'm/d', 'm', 'meter',
                      'in*100', 'in', 'in/day', 'inches/day', 'kg/kg', 'kpa',
                      'mmhg', 'k', 'f', 'm/s', 'mps', 'mpd', 'miles/day',
                      'miles/d']
        for k, v in self.weather['units'].items():
            if v is not None and v.lower() not in units_list:
                logging.error(('  ERROR: {0} units {1} are not currently'
                               ' supported').format(k, v))
                sys.exit()

        # Read historic max and min temperatures to support constant phenology
        """
        INI [HIST] Section

        """

        if self.phenology_option > 0:
            # hist_temps parameters

            self.hist_temps = {}
            self.hist_temps['fields'] = {}
            self.hist_temps['units'] = {}
            self.hist_temps['ws'] = config.get(hist_temps_sec,
                                               'hist_temps_folder')
            self.hist_temps['fnspec'] = {}
            self.hist_temps['wsspec'] = {}

            # hist_temps folder could be full or relative path
            # Assume relative paths or from project folder
            if os.path.isdir(self.hist_temps['ws']):
                pass
            elif (not os.path.isdir(self.hist_temps['ws']) and
                  os.path.isdir(os.path.join(self.project_ws,
                                             self.hist_temps['ws']))):
                self.hist_temps['ws'] = os.path.join(self.project_ws,
                                                     self.hist_temps['ws'])
            else:
                logging.error('ERROR: Phenology option selected and  hist temp'
                              ' folder {} does not exist'
                              .format(self.hist_temps['ws']))
                sys.exit()
            try:
                self.hist_temps['file_type'] = config.get(
                    hist_temps_sec, 'file_type')
                if self.hist_temps['file_type'] is None or \
                        self.hist_temps['file_type'] == 'None':
                    self.hist_temps['file_type'] = 'csv'
            except:
                self.hist_temps['file_type'] = 'csv'
            self.hist_temps['name_format'] = config.get(hist_temps_sec,
                                                        'name_format')
            self.hist_temps['header_lines'] = config.getint(hist_temps_sec,
                                                            'header_lines')
            self.hist_temps['names_line'] = config.getint(hist_temps_sec,
                                                          'names_line')
            try:
                self.hist_temps['delimiter'] = config.get(hist_temps_sec,
                                                          'delimiter')
                if self.hist_temps['delimiter'] is None or \
                        self.hist_temps['delimiter'] == 'None':
                    self.hist_temps['delimiter'] = ','
                else:
                    if self.hist_temps['delimiter'] not in [' ', ',', '\\t']:
                        self.hist_temps['delimiter'] = ','
                    if "\\" in self.hist_temps['delimiter'] and \
                            "t" in self.hist_temps['delimiter']:
                        self.hist_temps['delimiter'] = \
                            self.hist_temps['delimiter'].replace('\\t', '\t')
            except:
                    self.hist_temps['delimiter'] = ','

            # Field names and units
            # Date can be read directly or computed from year, month, and day
            try:
                self.hist_temps['fields']['date'] = config.get(hist_temps_sec,
                                                               'date_field')
            except:
                self.hist_temps['fields']['date'] = None
            try:
                self.hist_temps['fields']['year'] = config.get(hist_temps_sec,
                                                               'year_field')
                self.hist_temps['fields']['month'] = config.get(hist_temps_sec,
                                                                'month_field')
                self.hist_temps['fields']['day'] = config.get(hist_temps_sec,
                                                              'day_field')
            except:
                self.hist_temps['fields']['year'] = None
                self.hist_temps['fields']['month'] = None
                self.hist_temps['fields']['day'] = None
            if self.hist_temps['fields']['date'] is not None:
                logging.debug('  hist_temps: Reading date from date column')
            elif (self.hist_temps['fields']['year'] is not None and
                  self.hist_temps['fields']['month'] is not None and
                  self.hist_temps['fields']['day'] is not None):
                logging.debug('  hist_temps: Reading date from year, month,'
                              ' and day columns')
            else:
                logging.error('  ERROR: hist_temps date_field (or year, month,'
                              ' and day fields) must be set in INI')
                sys.exit()

            # Field names
            field_list = ['mint', 'maxt']
            for f_name in field_list:
                try:
                    self.hist_temps['fields'][f_name] = config.get(
                        hist_temps_sec, f_name + '_field')
                except:
                    logging.error(('  ERROR: hist_temps {}_field must be set ' +
                         'in INI').format(f_name))
                    sys.exit()

            # Units
            for f_name in field_list:
                if f_name == 'date':
                    continue
                elif self.hist_temps['fields'][f_name] is not None:
                    try:
                        self.hist_temps['units'][f_name] = config.get(
                            hist_temps_sec, f_name + '_units')
                    except:
                        logging.error(
                            ('  ERROR: hist_temps {}_units must be set ' +
                             'in INI').format(f_name))
                        sys.exit()

            # fnspec - parameter extension to file name specification
            for f_name in field_list:
                if f_name == 'date':
                    continue
                elif self.hist_temps['fields'][f_name] is not None:
                    try:
                        self.hist_temps['fnspec'][f_name] = config.get(
                            hist_temps_sec, f_name + '_name')
                    except:
                        self.hist_temps['fnspec'][f_name] = f_name
                else:
                    self.hist_temps['fnspec'][f_name] = 'Unused'

            # Check hist_temps parameters
            if not os.path.isdir(self.hist_temps['ws']):
                logging.error(
                    ('  ERROR:hist_temps data folder does not ' +
                     'exist\n  %s') % self.hist_temps['ws'])
                sys.exit()

            # Check units

            units_list = (['c', 'k', 'f'])
            for k, v in self.hist_temps['units'].items():
                if v is not None and v.lower() not in units_list:
                    logging.error(
                        ('  ERROR: {0} units {1} are not ' +
                         'currently supported').format(k, v))
                    sys.exit()

        # Check if refet_type matches crop_coefs_name
        if self.refet['type'] not in self.crop_coefs_path.lower():
            logging.warning('\nRefET Type does not match crop_coefs file name.'
                            ' Check the ini')
            logging.info('  refet_type = {}'.format(self.refet['type']))
            logging.info('  crop_coefs_name = {}'.format(self.crop_coefs_path))
            input('Press ENTER to continue or Ctrl+C to exit')

    def set_crop_params(self):
        """ List of <CropParameter> instances

        Parameters
        ---------
        None

        Returns
        -------
        None

        """

        logging.info('  Reading crop parameters from\n' + self.crop_params_path)

        params_df = pd.read_csv(self.crop_params_path,
                                  delimiter=self.crop_params_delimiter,
                                  header=None,
                                  skiprows=self.crop_params_header_lines - 1,
                                  na_values=['NaN'])
        params_df.applymap(str)
        params_df.fillna('0', inplace=True)
        self.crop_params = {}
        for crop_i in range(2, len(list(params_df.columns))):
            crop_param_data = params_df[crop_i].values.astype(str)
            crop_num = abs(int(crop_param_data[1]))
            self.crop_params[crop_num] = \
                crop_parameters.CropParameters(crop_param_data)

        # Filter crop parameters based on skip and test lists
        # Filtering could happen in read_crop_parameters()

        if self.crop_skip_list or self.crop_test_list:
            # Leave bare soil "crop" parameters
            # Used in initialize_crop_cycle()

            non_crop_list = [44]
            # non_crop_list = [44,45,46,55,56,57]
            self.crop_params = {
                k: v for k, v in self.crop_params.items()
                if ((self.crop_skip_list and k not in self.crop_skip_list) or
                    (self.crop_test_list and k in self.crop_test_list) or
                    (k in non_crop_list))}

    def set_crop_coeffs(self):
        """ List of <CropCoeff> instances

        Parameters
        ---------
        None

        Returns
        -------
        None

        """

        logging.info('  Reading crop coefficients')
        self.crop_coeffs = \
            crop_coefficients.read_crop_coefs_txt(self)

    def set_crop_co2(self):
        """Set crop CO2 type using values in INI

        Parameters
        ---------
        None

        Returns
        -------
        None

        """

        for crop_num, crop_param in self.crop_params.items():
            if not self.co2_flag:
                crop_param.co2_type = None
            elif self.co2_grass_crops and crop_num in self.co2_grass_crops:
                crop_param.co2_type = 'GRASS'
            elif self.co2_tree_crops and crop_num in self.co2_tree_crops:
                crop_param.co2_type = 'TREE'
            elif self.co2_c4_crops and crop_num in self.co2_c4_crops:
                crop_param.co2_type = 'C4'
            else:
                logging.warning('  Crop {} not in INI CO2'
                                ' lists'.format(crop_num))
                crop_param.co2_type = None
            self.crop_params[crop_num] = crop_param

def console_logger(logger=logging.getLogger(''), log_level=logging.INFO):
    # Create console logger
    logger.setLevel(log_level)
    log_console = logging.StreamHandler(stream=sys.stdout)
    log_console.setLevel(log_level)
    log_console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(log_console)
    return logger

# CHECK THAT INI USED IN TESTS EXISTS AND IS UP TO DATE
def do_tests():
    # Simple testing of functions as developed
    # logger = console_logger(log_level = 'DEBUG')
    logger = console_logger(log_level=logging.DEBUG)
    ini_path = os.getcwd() + os.sep + "cet_template.ini"
    cfg = CropETData()
    cfg.read_cet_ini(ini_path, True)


if __name__ == '__main__':
    # testing during development
    do_tests()
