"""ref_et_data.py
Defines MetNodesData and MetNode class
Called by met_nodes.py

"""
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lib')))
import ref_et_data
import ret_utils


# MOVE TO UNIT CONVERSION SCRIPT / SECTION
mpdToMps = 3.2808399 * 5280 / 86400

class MetNodesData():
    """Functions for loading Met Node Meta Data from static text file
    Attributes
    ----------

    Notes
    -----

    """

    def __init__(self):
        """ """
        self.met_nodes_data = dict()
        self.met_nodes_input_met_data = {}
        self.mn_daily_refetalt_out_data = {}
        self.mn_monthly_refetalt_out_data = {}
        self.mn_annual_refetalt_out_data = {}

    def set_met_nodes_meta_data(self, cfg):
        """Extract Met Node Meta Data from specified file

        Parameters
        ----------
        cfg :
            configuration data from INI file

        Returns
        -------
        None

        Notes
        -----
        This function builds MetNode objects and must be run first

        """

        logging.debug('  Reading Met Nodes Meta Data from {0}'.format(cfg.met_nodes_meta_data_path))
        try:
            # Get list of 0 based line numbers to skip
            # Ignore header but assume header was set as 1's based index
            data_skip = [i for i in range(cfg.mnmd_header_lines) if i + 1 != cfg.mnmd_names_line]
            if '.xls' in cfg.met_nodes_meta_data_path.lower():
                df = pd.read_excel(cfg.met_nodes_meta_data_path,
                        sheet_name = cfg.met_nodes_meta_data_ws,
                        header = cfg.mnmd_names_line - len(data_skip) - 1,
                        skiprows = data_skip, na_values = ['NaN'])
            else:
#                df = pd.read_csv(cfg.met_nodes_meta_data_path, engine = 'python',
                df = pd.read_csv(cfg.met_nodes_meta_data_path, engine = 'python',
                        header = cfg.mnmd_names_line - len(data_skip) - 1,
                        skiprows = data_skip, sep = cfg.mnmd_delimiter, na_values = ['NaN'])
            uc_columns = list(df.columns)
            columns = [x.lower() for x in uc_columns]
            try:
                index_column = columns.index('met node index')
                index_name = uc_columns[index_column]
                columns.remove('met node index')
                df.drop(list(df.columns)[index_column], axis = 1, inplace = True)
                uc_columns.remove(index_name)
            except:
                pass

            # remove excess baggage from column names
            columns = [x.replace(' (feet)', '').replace('decimal ','') for x in columns]

            # parse met node met data for each node id
            for rc, row in df.iterrows():
                met_node = MetNode()
                if not(met_node.read_mnmd_from_row(row.tolist(), columns,
                        cfg.elev_units)): sys.exit()
                self.met_nodes_data[met_node.met_node_id] = met_node
        except:
            logging.error('Unable to read met nodes meta data from ' + cfg.met_nodes_meta_data_path)
            logging.error('\nERROR: ' + str(sys.exc_info()[0]) + 'occurred\n')
            sys.exit()

    def read_avg_monthly_data(self, cfg):
        """Reads average monthly tmax, tmin, Ko, and wind for met nodes

        Parameters
        ---------
        cfg :
            configuration data from INI file
        mnd : dict
            Met node data

        Returns
        -------
        :
            True
            False

        """

        if cfg.input_met['avgm_tmax_path'] is None:
            self.avg_monthly_tmax = None
        else:
            if '.xls' in cfg.input_met['avgm_tmax_path'].lower():
                success, self.avg_monthly_tmax = ret_utils.read_average_monthly_wb_data(cfg.input_met['avgm_tmax_path'],
                    cfg.input_met['avgm_tmax_ws'], cfg.input_met['avgm_tmax_header_lines'])    # degrees C
            else:
                success, self.avg_monthly_tmax = ret_utils.read_average_monthly_csv_data(cfg.input_met['avgm_tmax_path'],
                    cfg.input_met['avgm_tmax_header_lines'], cfg.input_met['avgm_tmax_delimitor'])    # degrees C
            if not success:
                logging.error('Unable to read avearge monthly maximum temperatures from ' + cfg.input_met['avgm_tmax_path'])
                sys.exit()
        if cfg.input_met['avgm_tmin_path'] is None:
            self.avg_monthly_tmin = None
        else:
            if '.xls' in cfg.input_met['avgm_tmin_path'].lower():
                success, self.avg_monthly_tmin = ret_utils.read_average_monthly_wb_data(cfg.input_met['avgm_tmin_path'],
                    cfg.input_met['avgm_tmin_ws'], cfg.input_met['avgm_tmin_header_lines'])    # degrees C
            else:
                success, self.avg_monthly_tmin = ret_utils.read_average_monthly_csv_data(cfg.input_met['avgm_tmin_path'],
                    cfg.input_met['avgm_tmin_header_lines'], cfg.input_met['avgm_tmin_delimitor'])    # degrees C
            if not success:
                logging.error('Unable to read avearge monthly minimum temperatures from ' + cfg.input_met['avgm_tmin_path'])
                sys.exit()
        if cfg.input_met['avgm_Ko_path'] is None:
            self.avg_monthly_Ko = None
        else:
            if '.xls' in cfg.input_met['avgm_Ko_path'].lower():
                success, self.avg_monthly_Ko = ret_utils.read_average_monthly_wb_data(cfg.input_met['avgm_Ko_path'],
                    cfg.input_met['avgm_Ko_ws'], cfg.input_met['avgm_Ko_header_lines'])    # degrees C
            else:
                success, self.avg_monthly_Ko = ret_utils.read_average_monthly_csv_data(cfg.input_met['avgm_Ko_path'],
                    cfg.input_met['avgm_Ko_header_lines'], cfg.input_met['avgm_Ko_delimitor'])    # degrees C
            if not success:
                logging.error('Unable to read avearge monthly Ko data from ' + cfg.input_met['avgm_Ko_path'])
                sys.exit()
        if cfg.input_met['avgm_wind_path'] is None:
            self.avg_monthly_wind = None
        else:
            if '.xls' in cfg.input_met['avgm_wind_path'].lower():
                success, self.avg_monthly_wind = ret_utils.read_average_monthly_wb_data(cfg.input_met['avgm_wind_path'],
                    cfg.input_met['avgm_wind_ws'], cfg.input_met['avgm_wind_header_lines'])    # mps
            else:
                success, self.avg_monthly_wind = ret_utils.read_average_monthly_csv_data(cfg.input_met['avgm_wind_path'],
                    cfg.input_met['avgm_wind_header_lines'], cfg.input_met['avgm_wind_delimitor'])    # mps
            if not success:
                logging.error('Unable to read avearge monthly wind data from ' + cfg.input_met['avgm_wind_path'])
                sys.exit()

class MetNode():
    """Class for managing times series data of one Met Node
    Attributes
    ----------

    Notes
    -----

    """

    def __init__(self):
        """ """

    def __str__(self):
        """ """
        return '<MetNode {0}, {1}>'.format(self.met_node_id, self.met_node_name)

    def read_mnmd_from_row(self, data, columns, elev_units = 'feet'):
        """ Parse row of data from Met Node Met Data file

        Order of columns:
            Older headers (columns) include Met Node Index
            Met Node ID, Met Node Name, Decimal Latitude, Decimal Longitude, Elevation,
            Wind Station No, Wind Station Name, Dewpoint Depression Station No, Dewpoint Depression Station Name
            MET Data Path, Ref ET Data Path, TR_b0, TR_b1, TR_b2, Source Met Id    # these are optional

        Parameters
        ---------
            data : list
                one row of Met Nodes Meta Data

        Returns
        -------
        : boolean
            True
            False

        """

        # met node id is node id for ref et computations
        # source met id is met id in input met data file
        # source met id is automatically set to met node id if not provided

        if 'met node id' in columns:
            self.met_node_id = data[columns.index('met node id')]
        else:
            logging.error('Unable to read met node id')
            return False
        if 'met node name' in columns:
            self.met_node_name = data[columns.index('met node name')]
        else:
            self.met_node_name = self.met_node_id
        if 'latitude' in columns:
            self.latitude = float(data[columns.index('latitude')])
        else:
            logging.error('Unable to read met node latitude')
            return False
        if 'longitude' in columns:
            self.longitude = float(data[columns.index('longitude')])
        else:
            logging.error('Unable to read met node longitude')
            return False
        if 'elevation' in columns:
            self.elevation = float(data[columns.index('elevation')])
        else:
            logging.error('Unable to read met node elevation')
            return False
        if elev_units == 'feet' or elev_units == 'ft': self.elevation *= 0.3048
        if 'wind station no' in columns:
            self.wind_id = data[columns.index('wind station no')]
        else:
            logging.error('Unable to read wind id')
            return False
        if 'ko id' in columns:
            self.Ko_id = data[columns.index('ko id')]
        else:
            if 'dewpoint depression station no' in columns:
                self.Ko_id = data[columns.index('dewpoint depression station no')]
            else:
                logging.error('Unable to read ko id')
                return False

        # additional optional meta data

        self.met_data_path = None
        if 'met data path' in columns:
            if not pd.isnull(data[columns.index('met data path')]):
                if len(data[columns.index('met data path')]) > 3:
                    self.met_data_path = data[columns.index('met data path')]
        self.ret_data_path = None
        if 'ret data path' in columns:
            if not pd.isnull(data[columns.index('ret data path')]):
                if len(data[columns.index('ret data path')]) > 3:
                    self.ret_data_path = data[columns.index('ret data path')]
        self.TR_b0 = None
        if 'tr_b0' in columns:
            if not pd.isnull(data[columns.index('tr_b0')]):
                if len(data[columns.index('tr_b0')]) > 3:
                    self.TR_b0 = float(data[columns.index('tr_b0')])
        self.TR_b1 = None
        if 'tr_b1' in columns:
            if not pd.isnull(data[columns.index('tr_b1')]):
                if len(data[columns.index('tr_b1')]) > 3:
                    self.TR_b1 = float(data[columns.index('tr_b1')])
        self.TR_b2 = None
        if 'tr_b2' in columns:
            if not pd.isnull(data[columns.index('tr_b2')]):
                if len(data[columns.index('tr_b2')]) > 3:
                    self.TR_b2 = float(data[columns.index('tr_b2')])
        self.source_met_id = self.met_node_id
        if 'source met id' in columns:
            if not pd.isnull(data[columns.index('source met id')]):
                if len(data[columns.index('source met id')]) > 0:
                    self.source_met_id = data[columns.index('source met id')]
        return True

    def read_and_fill_met_data(self, met_node_count, cfg, mnd):
        """Read meteorological/climate data for single station and fill missing values

        Parameters
        ---------
        met_node_count : int
            count of met node being processed
        cfg :
            configuration data from INI file
        mnd : dict
            Met node data

        Returns
        -------
        : boolean
            True
            False

        """

        logging.debug('Reading and filling meteorological data')
        success = self.input_met_data(cfg)
        if not success:
            logging.error('Unable to read and fill daily input meteorological data.')
            return False

        # Check/modify units
        for field_key, field_units in cfg.input_met['units'].items():
            if field_units is None:
                continue
            elif field_units.lower() in ['c', 'mm', 'mm/d', 'mm/day',
                    'm/s', 'mj/m2', 'mj/m^2', 'mj/m2/day', 'mj/m^2/day',
                    'mj/m2/d', 'mj/m^2/d', 'kg/kg', 'mps']:
                continue
            elif field_units.lower() == 'm':
                self.input_met_df[field_key] *= 1000.0
            elif field_units.lower() in ['m/d', 'm/day']:
                if field_key == 'wind':
                    self.input_met_df[field_key] /= 86400
                else:
                    self.input_met_df[field_key] *= 1000.0
            elif field_units.lower() == 'meter':
                self.input_met_df[field_key] *= 1000.0
            elif field_units.lower() == 'k':
                self.input_met_df[field_key] -= 273.15
            elif field_units.lower() == 'f':
                self.input_met_df[field_key] -= 32
                self.input_met_df[field_key] /= 1.8
            elif field_units.lower() == 'in*100':
                self.input_met_df[field_key] *= 0.254
            elif field_units.lower() == 'in*10':
                self.input_met_df[field_key] *= 2.54
            elif field_units.lower() in ['in', 'in/d', 'in/day', 'inches/day', 'inches']:
                self.input_met_df[field_key] *= 25.4
            elif field_units.lower() in ['w/m2', 'w/m^2']:
                self.input_met_df[field_key] *= 0.0864
            elif field_units.lower() in ['cal/cm2', 'cal/cm2/d', 'cal/cm2/day', 'cal/cm^2/d', 'cal/cm^2/day', 'langley']:
                self.input_met_df[field_key] *= 0.041868
            elif field_units.lower() in ['mpd', 'miles/d', 'miles/day']:
                self.input_met_df[field_key] *= mpdToMps
            else:
                logging.error('\n ERROR: Unknown {0} units {1}'.format(field_key, field_units) + ' input met data')
                return False

        # set date attributes

        self.input_met_df['doy'] = self.input_met_df.index.dayofyear
        self.input_met_df['year'] = self.input_met_df.index.year
        self.input_met_df['month'] = self.input_met_df.index.month
        self.input_met_df['day'] = self.input_met_df.index.day
        if cfg.time_step == 'hour':
            self.input_met_df['hour'] = self.input_met_df.index.hour
        input_met_columns = list(self.input_met_df.columns)

        # contrain max temperatures to 120 F and min temperature to 90 F

        self.input_met_df['tmax'] = self.input_met_df[['tmax']].apply(lambda s: ret_utils.max_max_temp(*s), axis = 1, raw = True, result_type='reduce')
        self.input_met_df['tmin'] = self.input_met_df[['tmin']].apply(lambda s: ret_utils.max_min_temp(*s), axis = 1, raw = True, result_type='reduce')

        # fill missing tmax data

        # interpolation fails if at least one value is not filled - used try and except to overcome

        try:    # fill by interpolation
            self.input_met_df['tmax'].interpolate(method = 'time', limit = 3, limit_direction = 'both', inplace = True)
        except: pass
        try:    # fill with average monthly values
            for dt, month in sorted(self.input_met_df['month'].items()):
                if pd.isnull(self.input_met_df.at[dt, 'tmax']):
                   self.input_met_df.at[dt, 'tmax'] = mnd.avg_monthly_tmax[self.met_node_id][month - 1]
        except: pass

        # fill missing tmin data

        try:    # fill by interpolation
            self.input_met_df['tmin'].interpolate(method = 'time', limit = 3, limit_direction = 'both', inplace = True)
        except: pass
        try:    # fill with average monthly values
            for dt, month in sorted(self.input_met_df['month'].items()):
                if pd.isnull(self.input_met_df.at[dt, 'tmin']):
                   self.input_met_df.at[dt, 'tmin'] = mnd.avg_monthly_tmin[self.met_node_id][month - 1]
        except: pass

        # don't allow tmin to be more than tmax

        self.input_met_df['tmax'] = self.input_met_df[['tmax', 'tmin']].apply(lambda s: max(*s), axis = 1, raw = True, result_type='reduce')

        # Scale wind height to 2m if necessary

        if 'wind' in input_met_columns:
            if cfg.input_met['wind_height'] != 2:
                self.input_met_df['wind'] = refet.calcs._wind_height_adjust(self.input_met_df['wind'], cfg.input_met['wind_height'])

            # fill missing wind by interpolation

            try:
                self.input_met_df['wind'].interpolate(method = 'time', limit = 3, limit_direction = 'both', inplace = True)
            except: pass
        else:
            self.input_met_df['wind'] = np.nan

        # fill missing wind with average monthly values

        try:    # fill with average monthly values
            for dt, month in sorted(self.input_met_df['month'].items()):
                if pd.isnull(self.input_met_df.at[dt, 'wind']):
                   self.input_met_df.at[dt, 'wind'] = mnd.avg_monthly_wind[self.wind_id][month - 1]
        except: pass

        # Add precip, snow and snow_depth if necessary; otherwise, fill missing values with zeros

        if 'ppt' not in self.input_met_df.columns:
            self.input_met_df['ppt'] = 0
        else:
            self.input_met_df['ppt'].fillna(0)
        if 'snow' not in self.input_met_df.columns:
            self.input_met_df['snow'] = 0
        else:
            self.input_met_df['snow'].fillna(0)
        if 'snow_depth' not in self.input_met_df.columns:
            self.input_met_df['snow_depth'] = 0
        else:
            self.input_met_df['snow_depth'].fillna(0)

        # Calculate TDew from specific humidity or TMin and Ko
        if 'tdew' in input_met_columns:
            try:    # fill by interpolation
                self.input_met_df['tdew'].interpolate(method = 'time', limit = 3, limit_direction = 'both', inplace = True)
            except: pass
        else:
            self.input_met_df['tdew'] = np.nan
            if 'q' in self.input_met_df.columns:
                self.input_met_df['tdew'] = ret_utils.tdew_from_ea(ret_utils.ea_from_q(
                    self.air_pressure, self.input_met_df['q'].values))
        try:    # fill missing values with tmin minus average monthly Ko
            for dt, month in sorted(self.input_met_df['month'].items()):
                if pd.isnull(self.input_met_df.at[dt, 'tdew']):
                   self.input_met_df.at[dt, 'tdew'] = self.input_met_df.at[dt, 'tmin'] - mnd.avg_monthly_Ko[self.Ko_id][month - 1]
        except:
            logging.error('Unable to develop dew point temperature data.')
            return False

        # compute solar radiation for missing values

        if 'rs' not in self.input_met_df.columns: self.input_met_df['rs'] = np.nan
        try:
            for dt, doy in sorted(self.input_met_df['doy'].items()):
                if pd.isnull(self.input_met_df.at[dt, 'rs']):
                    self.input_met_df.at[dt, 'rs'] = ret_utils.rs_daily(doy,
                        self.input_met_df.at[dt, 'tmax'],
                        self.input_met_df.at[dt,'tmin'],
                        self.input_met_df.at[dt, 'tdew'],
                        self.elevation,
                        self.latitude,
                        mnd.avg_monthly_tmax[self.met_node_id][self.input_met_df['month'][dt] - 1],
                        mnd.avg_monthly_tmin[self.met_node_id][self.input_met_df['month'][dt] - 1],
                        self.TR_b0,
                        self.TR_b1,
                        self.TR_b2)
        except:
            logging.error('Unable to develop solar radiation.' + str(dt) + str(doy))
            return False
        return True

    def input_met_data(self, cfg):
        """Read meteorological/climate data for single station in station files with all parameters

        Parameters
        ---------
        cfg :
            configuration data from INI file

        Returns
        -------
        : boolean
            True
            False

        """

        if self.met_data_path is not None:
            input_met_path = self.met_data_path
        else:
            input_met_path = os.path.join(cfg.input_met['ws'], cfg.input_met['name_format'] % self.source_met_id)
        if not os.path.isfile(input_met_path):
            logging.error('ERROR:  input met file {} does not exist'.format(input_met_path))
            return False
        logging.debug('  {0}'.format(input_met_path))
        # Get list of 0 based line numbers to skip
        # Ignore header but assume header was set as 1's based index
        data_skip = [i for i in range(cfg.input_met['header_lines']) if i + 1 != cfg.input_met['names_line']]
#        self.input_met_df = pd.read_csv(input_met_path, engine = 'python',
        self.input_met_df = pd.read_csv(input_met_path, engine = 'python',
                header = cfg.input_met['names_line'] - len(data_skip) - 1,
                skiprows = data_skip, sep = cfg.input_met['delimiter'],
                na_values = 'NaN')
        logging.debug('  Columns: {0}'.format(', '.join(list(self.input_met_df.columns))))

        # Check fields

        for field_key, field_name in cfg.input_met['fields'].items():
            if (field_name is not None and field_name not in self.input_met_df.columns):
                if cfg.input_met['fnspec'][field_key].lower() == 'estimated': continue
                if cfg.input_met['fnspec'][field_key].lower() == 'unused': continue
                logging.error(
                    ('\n  ERROR: Field "{0}" was not found in {1}\n'+
                     '    Check {2}_field value in INI file').format(
                    field_name, os.path.basename(input_met_path), field_key))
                return False
            # Rename dataframe fields
            self.input_met_df = self.input_met_df.rename(columns = {field_name:field_key})

        # Convert date strings to datetimes and index on date

        if cfg.input_met['fields']['date'] is not None:
            self.input_met_df['date'] = pd.to_datetime(self.input_met_df['date'])
        else:
            if cfg.time_step == 'day':
                self.input_met_df['date'] = self.input_met_df[['year', 'month', 'day']].apply(
                    lambda s : datetime.datetime(*s),axis = 1)
            else:
                self.input_met_df['date'] = self.input_met_df[['year', 'month', 'day', 'hour']].apply(
                    lambda s : datetime.datetime(*s),axis = 1)
        self.input_met_df.set_index('date', inplace = True)

        # verify period

        if cfg.start_dt is None:
            pydt = self.input_met_df.index[0]
            cfg.start_dt = pd.to_datetime(datetime.datetime(pydt.year, pydt.month, pydt.day, pydt.hour, pydt.minute))
        if cfg.end_dt is None:
            pydt = self.input_met_df.index[len(self.input_met_df.index) - 1]
            cfg.end_dt = pd.to_datetime(datetime.datetime(pydt.year, pydt.month, pydt.day, pydt.hour, pydt.minute))

        # truncate period

        try:
            self.input_met_df = self.input_met_df.truncate(before = cfg.start_dt, after = cfg.end_dt)
        except:
            logging.error('\nERROR: ' + str(sys.exc_info()[0]) + 'occurred truncating input met data')
            return False
        if len(self.input_met_df.index) < 1:
            logging.error('No values found reading met data')
            return False
        return True

    def calculate_and_post_ret_data(self, cfg):
        """Computes reference et and posts reference et and met data

        Parameters
        ---------
        cfg :
            configuration data from INI file

        Returns
        -------
        : boolean
            True
            False

        """

        logging.debug('Computing Reference ET and post ref et and and meteorological data')

        try:
            # construct ref et object and set up output

            retObj = ref_et_data.refET(cfg.input_met['TR_b0'], cfg.input_met['TR_b1'], cfg.input_met['TR_b2'])
            ret_df = ret_utils.make_ts_dataframe(cfg.time_step, cfg.ts_quantity, cfg.start_dt, cfg.end_dt)
            for fn in cfg.refet_out['refet_out_fields']: ret_df[fn] = np.nan
            if cfg.output_retalt_flag:
                retalt_df = ret_utils.make_ts_dataframe(cfg.time_step, cfg.ts_quantity, cfg.start_dt, cfg.end_dt)
                for fn in cfg.refetalt_out['refet_out_fields']: retalt_df[fn] = np.nan
            for dt, row in self.input_met_df.iterrows():
                HargreavesSamani = retObj.et_hargreaves_samani(dt.dayofyear, row['tmax'],
                    row['tmin'], self.latitude)

                penmans = retObj.compute_penmans(dt.year, dt.month, dt.day, dt.dayofyear,
                    cfg.time_step, row['tmax'], row['tmin'], row['tdew'],
                    row['rs'], row['wind'], self.elevation, self.latitude)
                Penman, PreTay, KimbPeng, ASCEr, ASCEo, FAO56PM, KimbPen = penmans

                # store ref et calcs
                ret_df['ascer'][dt] = ASCEr
                ret_df['asceg'][dt] = ASCEo

                # store alt ref et calcs
                    # ASCE for comparison
                retalt_df['ascer'][dt] = ASCEr
                retalt_df['asceg'][dt] = ASCEo
                    # Alt
                retalt_df['penm'][dt] = Penman
                retalt_df['kimo'][dt] = KimbPeng
                retalt_df['kimr'][dt] = KimbPen
                retalt_df['fao56'][dt] = FAO56PM
                retalt_df['pretay'][dt] = PreTay
                retalt_df['harg'][dt] = HargreavesSamani

            # merge ref et calcs with met data for output
            try:
                daily_refet_df = pd.merge(self.input_met_df, ret_df, left_index = True, right_index = True)
            except:
                logging.error('\nERROR: ' + str(sys.exc_info()[0]) + 'occurred merging input met with ref et dataframe.\n')
                return False

            # merge alt ref et calcs with met data for output
            if cfg.output_retalt_flag:
                try:
                    daily_refetalt_df = pd.merge(self.input_met_df, retalt_df, left_index = True, right_index = True)
                except:
                    logging.error('\nERROR: ' + str(sys.exc_info()[0]) + 'occurred merging input met with alt ref et dataframe.\n')
                    return False

            # Check/modify units
            for field_key, field_units in cfg.refet_out['units'].items():
                if field_units is None: continue
                elif field_units.lower() in ['c', 'mm', 'mm/d', 'mm/day',
                        'm/s', 'mj/m2', 'mj/m^2', 'mj/m2/day', 'mj/m^2/day',
                        'mj/m2/d', 'mj/m^2/d', 'kg/kg', 'mps']:
                    continue
                elif field_units.lower() == 'k':
                   daily_refet_df[field_key] += 273.15
                elif field_units.lower() == 'f':
                    daily_refet_df[field_key] *= 1.8
                    daily_refet_df[field_key] += 32
                elif field_units.lower() == 'm':
                    daily_refet_df[field_key] *= 0.001
                elif field_units.lower() in ['m/d', 'm/day']:
                    if field_key == 'wind':
                        daily_refet_df[field_key] *= 86400
                    else:
                        daily_refet_df[field_key] *= 1000.0
                elif field_units.lower() == 'meter':
                    daily_refet_df[field_key] *= 0.001
                elif field_units.lower() == 'in*100':
                    daily_refet_df[field_key] /= 0.254
                elif field_units.lower() == 'in*10':
                    daily_refet_df[field_key] /= 2.54
                elif field_units.lower() in ['in', 'in/d', 'in/day', 'inches/day', 'inches']:
                    daily_refet_df[field_key] /= 25.4
                elif field_units.lower() in ['w/m2', 'w/m^2']:
                    daily_refet_df[field_key] /= 0.0864
                elif field_units.lower() in ['cal/cm2', 'cal/cm2/d', 'cal/cm2/day', 'cal/cm^2/d', 'cal/cm^2/day', 'langley']:
                    daily_refet_df[field_key] /= 0.041868
                elif field_units.lower() in ['mpd', 'miles/d', 'miles/day']:
                    daily_refet_df[field_key] /= mpdToMps
                else:
                    logging.error('\n ERROR: Unknown {0} units {1}'.format(field_key, field_units) + 'converting ref et output')
            data_fields = list(daily_refet_df.columns)
            for fn in data_fields:
                if fn == "date": continue
                if fn in cfg.used_refet_out_fields:
                    daily_refet_df = daily_refet_df.rename(columns={fn:cfg.refet_out['fields'][fn]})
                else:
                    daily_refet_df.drop(fn, axis = 1, inplace = True)
            if 'date' in cfg.refet_out['fields'] and cfg.refet_out['fields']['date'] is not None:
                try:
                    daily_refet_df.index.set_names(cfg.refet_out['fields']['date'], inplace = True)
                except: pass    # Index is probably already named 'Date'
            data_fields = list(daily_refet_df.columns)

            # set up aggregations
            aggregation_func = {}
            for fn in data_fields:
                fc = cfg.refet_out['out_data_fields'].index(fn)
                field_name = cfg.refet_out['data_out_fields'][fc]
                if "tmax" in field_name: aggregation_func.update({fn: np.mean})
                elif "tmin" in field_name: aggregation_func.update({fn: np.mean})
                elif "tavg" in field_name: aggregation_func.update({fn: np.mean})
                elif "tdew" in field_name: aggregation_func.update({fn: np.mean})
                elif "q" in field_name: aggregation_func.update({fn: np.mean})
                elif "wind" in field_name: aggregation_func.update({fn: np.mean})
                elif "rs" in field_name: aggregation_func.update({fn: np.mean})
                elif "solar" in field_name: aggregation_func.update({fn: np.mean})
                else: aggregation_func.update({fn: np.sum})
            if cfg.monthly_refet_flag:
                monthly_refet_df = daily_refet_df.resample('MS').apply( aggregation_func)
            if cfg.annual_refet_flag:
                annual_refet_df = daily_refet_df.resample('AS').apply( aggregation_func)

            # set up output fields
            if cfg.daily_refet_flag:
                adj_daily_fields = []
                if 'year' in cfg.used_refet_out_fields:
                    adj_daily_fields.append(cfg.refet_out['fields']['year'])
                    daily_refet_df[cfg.refet_out['fields']['year']] = daily_refet_df.index.year
                if 'month' in cfg.used_refet_out_fields:
                    adj_daily_fields.append(cfg.refet_out['fields']['month'])
                    daily_refet_df[cfg.refet_out['fields']['month']] = daily_refet_df.index.month
                if 'day' in cfg.used_refet_out_fields:
                    adj_daily_fields.append(cfg.refet_out['fields']['day'])
                    daily_refet_df[cfg.refet_out['fields']['day']] = daily_refet_df.index.day
                if 'doy' in cfg.used_refet_out_fields:
                    adj_daily_fields.append(cfg.refet_out['fields']['doy'])
                    daily_refet_df[cfg.refet_out['fields']['doy']] = daily_refet_df.index.doy
                adj_daily_fields.extend(cfg.refet_out['out_data_fields'])
            if cfg.monthly_refet_flag:
                adj_monthly_fields = []
                if 'year' in cfg.used_refet_out_fields:
                    adj_monthly_fields.append(cfg.refet_out['fields']['year'])
                    monthly_refet_df[cfg.refet_out['fields']['year']] = monthly_refet_df.index.year
                if 'month' in cfg.used_refet_out_fields:
                    adj_monthly_fields.append(cfg.refet_out['fields']['month'])
                    monthly_refet_df[cfg.refet_out['fields']['month']] = monthly_refet_df.index.month
                adj_monthly_fields.extend(cfg.refet_out['out_data_fields'])
            if cfg.annual_refet_flag:
                adj_annual_fields = []
                if 'year' in cfg.used_refet_out_fields:
                    adj_annual_fields.append(cfg.refet_out['fields']['year'])
                    annual_refet_df[cfg.refet_out['fields']['year']] = annual_refet_df.index.year
                adj_annual_fields.extend(cfg.refet_out['out_data_fields'])
            if cfg.daily_refet_flag:
                # format date attributes if values are formatted

                if cfg.refet_out['daily_float_format'] is not None:
                    if 'year' in cfg.used_refet_out_fields:
                        daily_refet_df[cfg.refet_out['fields']['year']] = \
                            daily_refet_df[cfg.refet_out['fields']['year']].map(lambda x: ' %4d' % x)
                    if 'month' in cfg.used_refet_out_fields:
                        daily_refet_df[cfg.refet_out['fields']['month']] = \
                            daily_refet_df[cfg.refet_out['fields']['month']].map(lambda x: ' %2d' % x)
                    if 'day' in cfg.used_refet_out_fields:
                        daily_refet_df[cfg.refet_out['fields']['day']] = \
                            daily_refet_df[cfg.refet_out['fields']['day']].map(lambda x: ' %2d' % x)
                if 'doy' in cfg.used_refet_out_fields:
                        daily_refet_df[cfg.refet_out['fields']['doy']] = \
                            daily_refet_df[cfg.refet_out['fields']['doy']].map(lambda x: ' %3d' % x)

                # post daily output

                daily_refet_path = os.path.join(cfg.daily_refet_ws, cfg.refet_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(daily_refet_path))
                with open(daily_refet_path, 'w', newline='') as daily_refet_f:
                    daily_refet_f.write(cfg.refet_out['daily_header1'] + '\n')
                    if cfg.refet_out['header_lines'] == 2:
                        daily_refet_f.write(cfg.refet_out['daily_header2'] + '\n')
                    if cfg.refet_out['daily_float_format'] is None:
                        if 'date' in cfg.used_refet_out_fields:
                            daily_refet_df.to_csv(daily_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['daily_date_format'],
                                columns = adj_daily_fields)
                        else:
                            daily_refet_df.to_csv(daily_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_daily_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refet_out_fields:
                            daily_refet_df.to_csv(daily_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['daily_date_format'],
                                float_format = cfg.refet_out['daily_float_format'],
                                columns = adj_daily_fields)
                        else:
                            daily_refet_df.to_csv(daily_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_daily_fields,
                                float_format = cfg.refet_out['daily_float_format'])
                del daily_refet_df, daily_refet_path, adj_daily_fields
            if cfg.monthly_refet_flag:
                if cfg.refet_out['monthly_float_format'] is not None:
                    if 'year' in cfg.used_refet_out_fields:
                        monthly_refet_df[cfg.refet_out['fields']['year']] = \
                            monthly_refet_df[cfg.refet_out['fields']['year']].map(lambda x: ' %4d' % x)
                    if 'month' in cfg.used_refet_out_fields:
                        monthly_refet_df[cfg.refet_out['fields']['month']] = \
                            monthly_refet_df[cfg.refet_out['fields']['month']].map(lambda x: ' %2d' % x)

                # post monthly output
                monthly_refet_path = os.path.join(cfg.monthly_refet_ws, cfg.refet_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(monthly_refet_path))
                with open(monthly_refet_path, 'w', newline='') as monthly_refet_f:
                    monthly_refet_f.write(cfg.refet_out['monthly_header1'] + '\n')
                    if cfg.refet_out['header_lines'] == 2:
                        monthly_refet_f.write(cfg.refet_out['monthly_header2'] + '\n')
                    if cfg.refet_out['monthly_float_format'] is None:
                        if 'date' in cfg.used_refet_out_fields:
                            monthly_refet_df.to_csv(monthly_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['monthly_date_format'],
                                columns = adj_monthly_fields)
                        else:
                            monthly_refet_df.to_csv(monthly_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_monthly_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refet_out_fields:
                            monthly_refet_df.to_csv(monthly_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['monthly_date_format'],
                                float_format = cfg.refet_out['monthly_float_format'],
                                columns = adj_monthly_fields)
                        else:
                            monthly_refet_df.to_csv(monthly_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_monthly_fields,
                                float_format = cfg.refet_out['monthly_float_format'])
                del monthly_refet_df, monthly_refet_path, adj_monthly_fields
            if cfg.annual_refet_flag:
                # format date attributes if values are formatted

                if cfg.refet_out['annual_float_format'] is not None:
                    if 'year' in cfg.used_refet_out_fields:
                        annual_refet_df[cfg.refet_out['fields']['year']] = \
                            annual_refet_df[cfg.refet_out['fields']['year']].map(lambda x: ' %4d' % x)

                # post annual output
                annual_refet_path = os.path.join(cfg.annual_refet_ws, cfg.refet_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(annual_refet_path))
                with open(annual_refet_path, 'w', newline='') as annual_refet_f:
                    annual_refet_f.write(cfg.refet_out['annual_header1'] + '\n')
                    if cfg.refet_out['header_lines'] == 2:
                        annual_refet_f.write(cfg.refet_out['annual_header2'] + '\n')
                    if cfg.refet_out['annual_float_format'] is None:
                        if 'date' in cfg.used_refet_out_fields:
                            annual_refet_df.to_csv(annual_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['annual_date_format'],
                                columns = adj_annual_fields)
                        else:
                            annual_refet_df.to_csv(annual_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_annual_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refet_out_fields:
                            annual_refet_df.to_csv(annual_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, date_format = cfg.refet_out['annual_date_format'],
                                float_format = cfg.refet_out['annual_float_format'],
                                columns = adj_annual_fields)
                        else:
                            annual_refet_df.to_csv(annual_refet_f, sep = cfg.refet_out['delimiter'],
                                header = False, index = False, columns = adj_annual_fields,
                                float_format = cfg.refet_out['annual_float_format'])
                del annual_refet_df, annual_refet_path, adj_annual_fields

            ## Post Alternative Ref ET Calcs
            # Check/modify units

            for field_key, field_units in cfg.refetalt_out['units'].items():
                if field_units is None: continue
                elif field_units.lower() in ['c', 'mm', 'mm/d', 'mm/day',
                        'm/s', 'mj/m2', 'mj/m^2', 'mj/m2/day', 'mj/m^2/day',
                        'mj/m2/d', 'mj/m^2/d', 'kg/kg', 'mps']:
                    continue
                elif field_units.lower() == 'k':
                   daily_refetalt_df[field_key] += 273.15
                elif field_units.lower() == 'f':
                    daily_refetalt_df[field_key] *= 1.8
                    daily_refetalt_df[field_key] += 32
                elif field_units.lower() == 'm':
                    daily_refetalt_df[field_key] *= 0.001
                elif field_units.lower() in ['m/d', 'm/day']:
                    if field_key == 'wind':
                        daily_refetalt_df[field_key] *= 86400
                    else:
                        daily_refetalt_df[field_key] *= 1000.0
                elif field_units.lower() == 'meter':
                    daily_refetalt_df[field_key] *= 0.001
                elif field_units.lower() == 'in*100':
                    daily_refetalt_df[field_key] /= 0.254
                elif field_units.lower() == 'in*10':
                    daily_refetalt_df[field_key] /= 2.54
                elif field_units.lower() in ['in', 'in/d', 'in/day', 'inches/day', 'inches']:
                    daily_refetalt_df[field_key] /= 25.4
                elif field_units.lower() in ['w/m2', 'w/m^2']:
                    daily_refetalt_df[field_key] /= 0.0864
                elif field_units.lower() in ['cal/cm2', 'cal/cm2/d', 'cal/cm2/day', 'cal/cm^2/d', 'cal/cm^2/day', 'langley']:
                    daily_refetalt_df[field_key] /= 0.041868
                elif field_units.lower() in ['mpd', 'miles/d', 'miles/day']:
                    daily_refetalt_df[field_key] /= mpdToMps
                else:
                    logging.error('\n ERROR: Unknown {0} units {1}'.format(field_key, field_units) + 'converting ref et output')
            data_fields = list(daily_refetalt_df.columns)
            for fn in data_fields:
                if fn == "date": continue
                if fn in cfg.used_refetalt_out_fields:
                    daily_refetalt_df = daily_refetalt_df.rename(columns={fn:cfg.refetalt_out['fields'][fn]})
                else:
                    daily_refetalt_df.drop(fn, axis = 1, inplace = True)
            if 'date' in cfg.refetalt_out['fields'] and cfg.refetalt_out['fields']['date'] is not None:
                try:
                    daily_refetalt_df.index.set_names(cfg.refetalt_out['fields']['date'], inplace = True)
                except: pass    # Index is probably already named 'Date'
            data_fields = list(daily_refetalt_df.columns)

            # set up aggregations

            aggregation_func = {}
            for fn in data_fields:
                fc = cfg.refetalt_out['out_data_fields'].index(fn)
                field_name = cfg.refetalt_out['data_out_fields'][fc]
                if "tmax" in field_name: aggregation_func.update({fn: np.mean})
                elif "tmin" in field_name: aggregation_func.update({fn: np.mean})
                elif "tavg" in field_name: aggregation_func.update({fn: np.mean})
                elif "tdew" in field_name: aggregation_func.update({fn: np.mean})
                elif "q" in field_name: aggregation_func.update({fn: np.mean})
                elif "wind" in field_name: aggregation_func.update({fn: np.mean})
                elif "rs" in field_name: aggregation_func.update({fn: np.mean})
                elif "solar" in field_name: aggregation_func.update({fn: np.mean})
                else: aggregation_func.update({fn: np.sum})
            if cfg.monthly_refetalt_flag:
                monthly_refetalt_df = daily_refetalt_df.resample('MS').apply( aggregation_func)
            if cfg.annual_refetalt_flag:
                annual_refetalt_df = daily_refetalt_df.resample('AS').apply( aggregation_func)

            # set up output fields

            if cfg.daily_refetalt_flag:
                adj_daily_fields = []
                if 'year' in cfg.used_refetalt_out_fields:
                    adj_daily_fields.append(cfg.refetalt_out['fields']['year'])
                    daily_refetalt_df[cfg.refetalt_out['fields']['year']] = daily_refetalt_df.index.year
                if 'month' in cfg.used_refetalt_out_fields:
                    adj_daily_fields.append(cfg.refetalt_out['fields']['month'])
                    daily_refetalt_df[cfg.refetalt_out['fields']['month']] = daily_refetalt_df.index.month
                if 'day' in cfg.used_refetalt_out_fields:
                    adj_daily_fields.append(cfg.refetalt_out['fields']['day'])
                    daily_refetalt_df[cfg.refetalt_out['fields']['day']] = daily_refetalt_df.index.day
                if 'doy' in cfg.used_refetalt_out_fields:
                    adj_daily_fields.append(cfg.refetalt_out['fields']['doy'])
                    daily_refetalt_df[cfg.refetalt_out['fields']['doy']] = daily_refetalt_df.index.doy
                adj_daily_fields.extend(cfg.refetalt_out['out_data_fields'])
            if cfg.monthly_refetalt_flag:
                adj_monthly_fields = []
                if 'year' in cfg.used_refetalt_out_fields:
                    adj_monthly_fields.append(cfg.refetalt_out['fields']['year'])
                    monthly_refetalt_df[cfg.refetalt_out['fields']['year']] = monthly_refetalt_df.index.year
                if 'month' in cfg.used_refetalt_out_fields:
                    adj_monthly_fields.append(cfg.refetalt_out['fields']['month'])
                    monthly_refetalt_df[cfg.refetalt_out['fields']['month']] = monthly_refetalt_df.index.month
                adj_monthly_fields.extend(cfg.refetalt_out['out_data_fields'])
            if cfg.annual_refetalt_flag:
                adj_annual_fields = []
                if 'year' in cfg.used_refetalt_out_fields:
                    adj_annual_fields.append(cfg.refetalt_out['fields']['year'])
                    annual_refetalt_df[cfg.refetalt_out['fields']['year']] = annual_refetalt_df.index.year
                adj_annual_fields.extend(cfg.refetalt_out['out_data_fields'])
            if cfg.daily_refetalt_flag:
                # format date attributes if values are formatted

                if cfg.refetalt_out['daily_float_format'] is not None:
                    if 'year' in cfg.used_refetalt_out_fields:
                        daily_refetalt_df[cfg.refetalt_out['fields']['year']] = \
                            daily_refetalt_df[cfg.refetalt_out['fields']['year']].map(lambda x: ' %4d' % x)
                    if 'month' in cfg.used_refetalt_out_fields:
                        daily_refetalt_df[cfg.refetalt_out['fields']['month']] = \
                            daily_refetalt_df[cfg.refetalt_out['fields']['month']].map(lambda x: ' %2d' % x)
                    if 'day' in cfg.used_refetalt_out_fields:
                        daily_refetalt_df[cfg.refetalt_out['fields']['day']] = \
                            daily_refetalt_df[cfg.refetalt_out['fields']['day']].map(lambda x: ' %2d' % x)
                if 'doy' in cfg.used_refetalt_out_fields:
                        daily_refetalt_df[cfg.refetalt_out['fields']['doy']] = \
                            daily_refetalt_df[cfg.refetalt_out['fields']['doy']].map(lambda x: ' %3d' % x)

                # post daily output

                daily_refetalt_path = os.path.join(cfg.daily_refetalt_ws, cfg.refetalt_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(daily_refetalt_path))
                with open(daily_refetalt_path, 'w', newline='') as daily_refetalt_f:
                    daily_refetalt_f.write(cfg.refetalt_out['daily_header1'] + '\n')
                    if cfg.refetalt_out['header_lines'] == 2:
                        daily_refetalt_f.write(cfg.refetalt_out['daily_header2'] + '\n')
                    if cfg.refetalt_out['daily_float_format'] is None:
                        if 'date' in cfg.used_refetalt_out_fields:
                            daily_refetalt_df.to_csv(daily_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['daily_date_format'],
                                columns = adj_daily_fields)
                        else:
                            daily_refetalt_df.to_csv(daily_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_daily_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refetalt_out_fields:
                            daily_refetalt_df.to_csv(daily_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['daily_date_format'],
                                float_format = cfg.refetalt_out['daily_float_format'],
                                columns = adj_daily_fields)
                        else:
                            daily_refetalt_df.to_csv(daily_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_daily_fields,
                                float_format = cfg.refetalt_out['daily_float_format'])
                del daily_refetalt_df, daily_refetalt_path, adj_daily_fields
            if cfg.monthly_refetalt_flag:
                if cfg.refetalt_out['monthly_float_format'] is not None:
                    if 'year' in cfg.used_refetalt_out_fields:
                        monthly_refetalt_df[cfg.refetalt_out['fields']['year']] = \
                            monthly_refetalt_df[cfg.refetalt_out['fields']['year']].map(lambda x: ' %4d' % x)
                    if 'month' in cfg.used_refetalt_out_fields:
                        monthly_refetalt_df[cfg.refetalt_out['fields']['month']] = \
                            monthly_refetalt_df[cfg.refetalt_out['fields']['month']].map(lambda x: ' %2d' % x)

                # post monthly output

                monthly_refetalt_path = os.path.join(cfg.monthly_refetalt_ws, cfg.refetalt_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(monthly_refetalt_path))
                with open(monthly_refetalt_path, 'w', newline='') as monthly_refetalt_f:
                    monthly_refetalt_f.write(cfg.refetalt_out['monthly_header1'] + '\n')
                    if cfg.refetalt_out['header_lines'] == 2:
                        monthly_refetalt_f.write(cfg.refetalt_out['monthly_header2'] + '\n')
                    if cfg.refetalt_out['monthly_float_format'] is None:
                        if 'date' in cfg.used_refetalt_out_fields:
                            monthly_refetalt_df.to_csv(monthly_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['monthly_date_format'],
                                columns = adj_monthly_fields)
                        else:
                            monthly_refetalt_df.to_csv(monthly_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_monthly_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refetalt_out_fields:
                            monthly_refetalt_df.to_csv(monthly_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['monthly_date_format'],
                                float_format = cfg.refetalt_out['monthly_float_format'],
                                columns = adj_monthly_fields)
                        else:
                            monthly_refetalt_df.to_csv(monthly_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_monthly_fields,
                                float_format = cfg.refetalt_out['monthly_float_format'])
                del monthly_refetalt_df, monthly_refetalt_path, adj_monthly_fields
            if cfg.annual_refetalt_flag:
                # format date attributes if values are formatted

                if cfg.refetalt_out['annual_float_format'] is not None:
                    if 'year' in cfg.used_refetalt_out_fields:
                        annual_refetalt_df[cfg.refetalt_out['fields']['year']] = \
                            annual_refetalt_df[cfg.refetalt_out['fields']['year']].map(lambda x: ' %4d' % x)

                # post annual output

                annual_refetalt_path = os.path.join(cfg.annual_refetalt_ws, cfg.refetalt_out['name_format'] % self.met_node_id)
                logging.debug('  {0}'.format(annual_refetalt_path))
                with open(annual_refetalt_path, 'w', newline='') as annual_refetalt_f:
                    annual_refetalt_f.write(cfg.refetalt_out['annual_header1'] + '\n')
                    if cfg.refetalt_out['header_lines'] == 2:
                        annual_refetalt_f.write(cfg.refetalt_out['annual_header2'] + '\n')
                    if cfg.refetalt_out['annual_float_format'] is None:
                        if 'date' in cfg.used_refetalt_out_fields:
                            annual_refetalt_df.to_csv(annual_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['annual_date_format'],
                                columns = adj_annual_fields)
                        else:
                            annual_refetalt_df.to_csv(annual_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_annual_fields)
                    else:    # formatted output causes loss of precision in crop et computations
                        if 'date' in cfg.used_refetalt_out_fields:
                            annual_refetalt_df.to_csv(annual_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, date_format = cfg.refetalt_out['annual_date_format'],
                                float_format = cfg.refetalt_out['annual_float_format'],
                                columns = adj_annual_fields)
                        else:
                            annual_refetalt_df.to_csv(annual_refetalt_f, sep = cfg.refetalt_out['delimiter'],
                                header = False, index = False, columns = adj_annual_fields,
                                float_format = cfg.refetalt_out['annual_float_format'])
                del annual_refetalt_df, annual_refetalt_path, adj_annual_fields
            return True;
        except:
            logging.error('\nERROR: ' + str(sys.exc_info()[0]) + 'occurred computing reference ET for {0}', format(self.met_node_id))
            return False
