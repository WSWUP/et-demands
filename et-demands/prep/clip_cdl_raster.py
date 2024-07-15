#--------------------------------
# Name:         clip_cdl_raster.py
# Purpose:      Clip CDL rasters to the ET zones extent
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import shutil
import subprocess
import sys

from osgeo import gdal, ogr

import _gdal_common as gdc
import _util as util


def main(ini_path, overwrite_flag=False):
    """Clip CDL rasters to a target extent and rebuild color table

    Parameters
    ----------
    ini_path : str
        File path of the parameter INI file.
    overwrite_flag (bool):
        If True, overwrite the output raster (the default is False).

    Returns
    -------
    None

    Notes
    -----
    The script will attempt to clip into the project/gis/cdl folder.
    If the CDL raster is already in this folder, it will be overwritten.

    """
    logging.debug('INI: {}'.format(ini_path))
    config = util.read_ini(ini_path, section='CROP_ET')
    zones_path = config.get('CROP_ET', 'cells_path')
    gis_ws = config.get('CROP_ET', 'gis_folder')
    cdl_input_ws = config.get('CROP_ET', 'cdl_folder')
    cdl_year = int(config.get('CROP_ET', 'cdl_year'))
    cdl_format = config.get('CROP_ET', 'cdl_format')

    cdl_output_ws = os.path.join(gis_ws, 'cdl')

    cdl_input_path = os.path.join(
        cdl_input_ws, cdl_format.format(cdl_year, 'img'))
    cdl_output_path = os.path.join(
        cdl_output_ws, cdl_format.format(cdl_year, 'img'))

    # Keep the CDL raster in the default IMG format
    output_format = 'HFA'

    pyramids_flag = True
    stats_flag = True

    # if pyramids_flag:
    levels = '2 4 8 16 32 64 128'
    # gdal.SetConfigOption('USE_RRD', 'YES')
    # gdal.SetConfigOption('HFA_USE_RRD', 'YES')
    # gdal.SetConfigOption('HFA_COMPRESS_OVR', 'YES')

    if os.name == 'posix':
        shell_flag = False
    else:
        shell_flag = True

    # Check input folders
    if not os.path.isfile(zones_path):
        logging.error(
            '\nERROR: The ET zone shapefile doesn\'t exist, exiting\n'
            '  {}'.format(zones_path))
        sys.exit()
    elif not os.path.isfile(cdl_input_path):
        logging.error(
            '\nERROR: The input CDL raster doesn\'t exist, exiting\n'
            '  {}'.format(cdl_input_path))
        sys.exit()

    if not os.path.isdir(cdl_output_ws):
        os.makedirs(cdl_output_ws)

    logging.info('\nGIS Workspace: {}'.format(gis_ws))
    logging.info('CDL Input Path:  {}'.format(cdl_input_path))
    logging.info('CDL Output Path: {}'.format(cdl_output_path))

    # TODO: Add logic to handle doing the clip inplace
    if cdl_input_path == cdl_output_path:
        logging.error('\nThe script does not currently handle clipping the '
                      'CDL raster in place, exiting')
        sys.exit()

    # CDL Raster Properties
    cdl_ds = gdal.Open(cdl_input_path)
    cdl_proj = cdl_ds.GetProjection()
    cdl_osr = gdc.proj_osr(cdl_proj)
    cdl_geo = cdl_ds.GetGeoTransform()
    cdl_x, cdl_y = gdc.geo_origin(cdl_geo)
    cdl_cs = gdc.geo_cellsize(cdl_geo, x_only=True)
    logging.debug('\nCDL Input Raster Properties')
    logging.debug('  Geo:        {}'.format(cdl_geo))
    logging.debug('  Snap:       {} {}'.format(cdl_x, cdl_y))
    logging.debug('  Cellsize:   {}'.format(cdl_cs))
    logging.debug('  Projection: {}'.format(cdl_osr.ExportToWkt()))
    # logging.debug('  OSR: {}'.format(cdl_osr))
    # logging.debug('  Extent: {}'.format(zones_extent))

    # Reference all output rasters zone raster
    zones_ds = ogr.Open(zones_path, 0)
    zones_lyr = zones_ds.GetLayer()
    zones_osr = zones_lyr.GetSpatialRef()
    # zones_wkt = gdc.osr_proj(zones_ds)
    zones_extent = gdc.feature_lyr_extent(zones_lyr)
    zones_ds = None
    logging.debug('\nET Zones Shapefile Properties')
    logging.debug('  Extent:     {}'.format(zones_extent))
    logging.debug('  Projection: {}'.format(zones_osr.ExportToWkt()))
    # logging.debug('  OSR:    {}'.format(zones_osr))

    # Subset/clip properties
    # Project the extent to the CDL spatial reference
    logging.debug('\nCDL Output Raster Properties')
    clip_extent = zones_extent.project(zones_osr, cdl_osr)
    logging.debug('  Projected:  {}'.format(clip_extent))
    # Adjust the clip extent to the CDL snap point and cell size
    clip_extent.buffer(10 * cdl_cs)
    clip_extent.adjust_to_snap(snap_x=cdl_x, snap_y=cdl_y, cs=cdl_cs,
                               method='EXPAND')
    logging.debug('  Snapped:    {}'.format(clip_extent))
    # Limit the subset extent to CDL extent
    clip_extent.clip(clip_extent)
    logging.debug('  Clipped:    {}'.format(clip_extent))
    clip_ullr = clip_extent.ul_lr_swap()
    logging.debug('  Clipped UL/LR: {}'.format(clip_ullr))

    # Overwrite
    if os.path.isfile(cdl_output_path) or overwrite_flag:
        logging.info('\nDeleting existing raster')
        logging.debug('  {}'.format(cdl_output_path))
        # subprocess.run(
        subprocess.check_output(
            ['gdalmanage', 'delete', '-f', output_format, cdl_output_path],
            shell=shell_flag)
        # remove_file(cdl_output_path)

    # Clip
    if not os.path.isfile(cdl_output_path):
        logging.info('\nClipping CDL raster')
        logging.debug('  {}\n  {}'.format(cdl_input_path, cdl_output_path))
        # subprocess.run(
        subprocess.check_output(
            ['gdal_translate', '-of', output_format, '-co', 'COMPRESSED=YES'] +
            ['-projwin'] + str(clip_ullr).split() +
            ['-a_ullr'] + str(clip_ullr).split() +
            [cdl_input_path, cdl_output_path],
            shell=shell_flag)
        if os.path.isfile(cdl_input_path.replace('.img', '.img.vat.dbf')):
            shutil.copyfile(
                cdl_input_path.replace('.img', '.img.vat.dbf'),
                cdl_output_path.replace('.img', '.img.vat.dbf')
            )

    # Statistics
    if stats_flag and os.path.isfile(cdl_output_path):
        logging.info('\nComputing statistics')
        logging.debug('  {}'.format(cdl_output_path))
        # subprocess.run(
        subprocess.check_output(
            ['gdalinfo', '-stats', '-nomd', '-noct', '-norat',
             cdl_output_path],
            shell=shell_flag)

    # Pyramids
    if pyramids_flag and os.path.isfile(cdl_output_path):
        logging.info('\nBuilding pyramids')
        logging.debug('  {}'.format(cdl_output_path))
        # subprocess.run(
        subprocess.check_output(
            ['gdaladdo', '-ro', cdl_output_path] + levels.split(),
            shell=shell_flag)
        # args = ['gdaladdo', '-ro']
        # if cdl_output_path.endswith('.img'):
        #     args.extend([
        #         '--config', 'USE_RRD YES',
        #         '--config', 'HFA_USE_RRD YES',
        #         '--config', 'HFA_COMPRESS_OVR YES'])
        # args.append(cdl_output_path)
        # args.extend(levels.split())
        # subprocess.check_output(args, shell=shell_flag)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Clip the CDL Raster to the cell extent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, metavar='INI',
        type=lambda x: util.is_valid_file(parser, x),
        help='Input INI File')
    parser.add_argument(
        '-o', '--overwrite', default=None, action='store_true',
        help='Overwrite existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.ini and os.path.isdir(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)

    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Start Time:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
