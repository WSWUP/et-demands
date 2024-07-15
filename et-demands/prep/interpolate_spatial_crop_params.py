# --------------------------------
# Name:         interpolate_spatial_crop_params.py
# Purpose:      Interpolate spatial parameter files for ET-Demands
# --------------------------------

import argparse
from collections import defaultdict
import datetime as dt
import logging
import math
import os
import pprint
import re
import sys

from osgeo import gdal, ogr, osr

import _arcpy
import _util as util


def main(ini_path):
    """Interpolate Preliminary Calibration Zones to All Zones

    Args:
        ini_path (str): file path of the project INI file
    Returns:
        None

    """
    logging.info('\nInterpolating Calibration Data from Subset Point Data')

    #  INI path
    crop_et_sec = 'CROP_ET'
    config = util.read_ini(ini_path, section=crop_et_sec)

    try:
        project_ws = config.get(crop_et_sec, 'project_folder')
    except:
        logging.error('project_folder parameter must be set in the INI file, '
                      'exiting')
        return False
    try:
        gis_ws = config.get(crop_et_sec, 'gis_folder')
    except:
        logging.error('gis_folder parameter must be set in the INI file, '
                      'exiting')
        return False
    try:
        et_cells_path = config.get(crop_et_sec, 'cells_path')
    except:
        logging.error('et_cells_path parameter must be set in the INI file, '
                      'exiting')
        return False
    try:
        calibration_ws = config.get(crop_et_sec, 'spatial_cal_folder')
    except:
        calibration_ws = os.path.join(project_ws, 'calibration')

    try:
        crop_params_name = config.get(crop_et_sec, 'crop_params_name')
    except:
        logging.error('crop_params_name parameter must be set in the INI file, '
                      'exiting')
        return False

    # Sub folder names
    static_ws = os.path.join(project_ws, 'static')
    crop_params_path = os.path.join(static_ws, crop_params_name)
    crop_et_ws = config.get(crop_et_sec, 'crop_et_folder')
    bin_ws = os.path.join(crop_et_ws, 'bin')

    # Check input folders
    if not os.path.exists(calibration_ws):
        logging.critical('\nERROR: The calibration folder does not exist. '
                         '\n  Run build_spatial_crop_params.py')
        sys.exit()

    # Check input folders
    if not os.path.isdir(project_ws):
        logging.critical('\nERROR: The project folder does not exist'
                         '\n  {}'.format(project_ws))
        sys.exit()
    elif not os.path.isdir(gis_ws):
        logging.critical('\nERROR: The GIS folder does not exist'
                         '\n  {}'.format(gis_ws))
        sys.exit()

    logging.info('\nGIS Workspace:      {}'.format(gis_ws))

    # ET cells field names
    cell_id_field = 'CELL_ID'
    cell_station_id_field = 'STATION_ID'
    # cell_name_field = 'CELL_NAME'
    # crop_acres_field = 'CROP_ACRES'

    # Do distance calculations in decimal degrees to match Arc script
    gcs_osr = osr.SpatialReference()
    gcs_osr.ImportFromEPSG(4326)

    # Read in the cell locations and values
    et_cells_data = defaultdict(dict)
    input_driver = _arcpy.get_ogr_driver(et_cells_path)
    input_ds = input_driver.Open(et_cells_path, 0)
    input_lyr = input_ds.GetLayer()
    input_osr = input_lyr.GetSpatialRef()
    # gcs_osr = input_osr.CloneGeogCS()
    for input_ftr in input_lyr:
        input_fid = input_ftr.GetFID()
        logging.debug('  FID: {}'.format(input_fid))
        input_id = input_ftr.GetField(input_ftr.GetFieldIndex(cell_id_field))
        input_geom = input_ftr.GetGeometryRef()
        centroid_geom = input_geom.Clone()
        # Do distance calculations in decimal degrees to match Arc script
        centroid_geom.Transform(
            osr.CoordinateTransformation(input_osr, gcs_osr))
        centroid_geom = centroid_geom.Centroid()
        et_cells_data[input_id]['X'] = centroid_geom.GetX()
        et_cells_data[input_id]['Y'] = centroid_geom.GetY()
    input_ds = None

    # Read crop parameters using ET Demands functions/methods
    logging.info('\nReading default crop parameters')
    sys.path.append(bin_ws)
    import crop_parameters
    crop_param_dict = crop_parameters.read_crop_parameters(crop_params_path)

    logging.info(_arcpy.list_fields(et_cells_path))
    
    # Get list of crops specified in ET cells
    crop_field_list = [
        field for field in _arcpy.list_fields(et_cells_path)
        if re.match('CROP_\d{2}', field)]
        
    crop_number_list = [int(f.split('_')[1]) for f in crop_field_list]
    logging.info('Cell crop numbers: {}'.format(
        ', '.join(list(util.ranges(crop_number_list)))))
    logging.debug('Cell crop fields: {}'.format(', '.join(crop_field_list)))

    # Get Crop Names for each Crop in crop_number_list
    crop_name_list = []
    logging.debug('\nBuilding crop name list')
    for crop_num in crop_number_list:
        try:
            crop_param = crop_param_dict[crop_num]
        except:
            continue
        # logging.info('{:>2d} {}'.format(crop_num, crop_param.name))
        logging.debug('{}'.format(crop_param))
        # Replace other characters with spaces, then remove multiple spaces
        crop_name = re.sub('[-"().,/~]', ' ', str(crop_param.name).lower())
        crop_name = ' '.join(crop_name.strip().split()).replace(' ', '_')
        crop_name_list.append(crop_name)

    # Location of preliminary calibration .shp files (ADD AS INPUT ARG?)
    prelim_calibration_ws = os.path.join(calibration_ws,
                                         'preliminary_calibration')

    logging.info('\nInterpolating calibration parameters')
    for crop_num, crop_name in zip(crop_number_list, crop_name_list):
        # Preliminary calibration .shp
        subset_cal_file = os.path.join(
            prelim_calibration_ws,
            'crop_{0:02d}_{1}{2}').format(crop_num, crop_name, '.shp')
        final_cal_file = os.path.join(
            calibration_ws,
            'crop_{0:02d}_{1}{2}').format(crop_num, crop_name, '.shp')

        if not _arcpy.exists(subset_cal_file):
            logging.info(
                '\nCrop No: {} Preliminary calibration file not found. '
                'skipping.'.format(crop_num))
            continue
        logging.info('\nInterpolating Crop: {:02d}'.format(crop_num))

        # Params to Interpolate
        # param_list = ['T30_CGDD', 'CGDD_EFC', 'CGDD_TERM', 'KillFrostC']
        param_list = ['MAD_Init', 'MAD_Mid', 'T30_CGDD',
            'PL_GU_Date', 'CGDD_Tbase', 'CGDD_EFC',
            'CGDD_Term', 'Time_EFC', 'Time_Harv', 'KillFrostC']

        # Read in the calibration locations and values
        subset_cal_data = defaultdict(dict)
        input_driver = _arcpy.get_ogr_driver(subset_cal_file)
        input_ds = input_driver.Open(subset_cal_file, 0)
        input_lyr = input_ds.GetLayer()
        input_osr = input_lyr.GetSpatialRef()
        # gcs_osr = input_osr.CloneGeogCS()
        for input_ftr in input_lyr:
            input_fid = input_ftr.GetFID()
            logging.debug('  FID: {}'.format(input_fid))
            input_id = input_ftr.GetField(input_ftr.GetFieldIndex(
                cell_id_field))
            input_geom = input_ftr.GetGeometryRef()
            centroid_geom = input_geom.Clone()
            # Do distance calculations in decimal degrees to match Arc script
            centroid_geom.Transform(
                osr.CoordinateTransformation(input_osr, gcs_osr))
            centroid_geom = centroid_geom.Centroid()
            subset_cal_data[input_id]['X'] = centroid_geom.GetX()
            subset_cal_data[input_id]['Y'] = centroid_geom.GetY()
            for f in param_list:
                subset_cal_data[input_id][f] = input_ftr.GetField(
                    input_ftr.GetFieldIndex(f))
        input_ds = None

        # Compute interpolated calibration parameters
        final_cal_data = defaultdict(dict)
        for cell_id, cell_dict in et_cells_data.items():
            final_cal_data[cell_id] = {}
            logging.debug('  {}'.format(cell_id))

            # Precompute distances to all subset cells
            weight = {}
            for subset_id, subset_dict in subset_cal_data.items():
                distance = math.sqrt(
                    (subset_dict['X'] - cell_dict['X']) ** 2 +
                    (subset_dict['Y'] - cell_dict['Y']) ** 2)
                try:
                    weight[subset_id] = distance ** -2.0
                except:
                    weight[subset_id] = 0
                weight_total = sum(weight.values())

            # Brute force IDW using all subset cell
            for param in param_list:
                # If any weight is zero, use the values directly
                # There is probably a better way of flagging these
                d0 = [id for id, w in weight.items() if w == 0]
                if d0:
                    final_cal_data[cell_id][param] = subset_cal_data[
                        d0[0]][param]
                else:
                    final_cal_data[cell_id][param] = sum([
                        data[param] * weight[id]
                        for id, data in subset_cal_data.items()])
                    final_cal_data[cell_id][param] /= weight_total

        # Overwrite values in calibration .shp with interpolated values
        output_ds = input_driver.Open(final_cal_file, 1)
        output_lyr = output_ds.GetLayer()
        for output_ftr in output_lyr:
            output_id = output_ftr.GetField(
                output_ftr.GetFieldIndex(cell_id_field))
            for param in param_list:
                output_ftr.SetField(
                    input_ftr.GetFieldIndex(param),
                    round(final_cal_data[output_id][param], 1))
            output_lyr.SetFeature(output_ftr)
        output_ds = None


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET-Demands Interpolate Spatial Crop Parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type=lambda x: util.is_valid_file(parser, x), help='Input file')
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

    main(ini_path=args.ini)
