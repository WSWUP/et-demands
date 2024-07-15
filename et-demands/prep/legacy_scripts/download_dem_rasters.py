#--------------------------------
# Name:         download_dem_rasters.py
# Purpose:      Download NED tiles
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys
import zipfile

from osgeo import gdal, ogr

import _gdal_common as gdc
import _util as util


def main(gis_ws, tile_ws, dem_cs, mask_flag=False, overwrite_flag=False):
    """Download NED tiles that intersect the study_area

    Script assumes DEM data is in 1x1 WGS84 degree tiles
    Download 10m (1/3 arc-second) or 30m (1 arc-second) versions from:
        10m: rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/13/IMG
        30m: rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG
    For this example, only download 30m DEM

    Args:
        gis_ws (str): Folder/workspace path of the GIS data for the project
        tile_ws (str): Folder/workspace path of the DEM tiles
        dem_cs (int): DEM cellsize (10 or 30m)
        mask_flag (bool): If True, only download tiles intersecting zones mask
        overwrite_flag (bool): If True, overwrite existing files

    """
    logging.info('\nDownload DEM tiles')

    zip_fmt = 'n{0:02d}w{1:03d}.zip'
    site_url = 'rockyftp.cr.usgs.gov'
    if dem_cs == 10:
        site_folder = 'vdelivery/Datasets/Staged/Elevation/13/IMG'
    elif dem_cs == 30:
        site_folder = 'vdelivery/Datasets/Staged/Elevation/1/IMG'
    else:
        logging.error('\nERROR: The input cell size must be 10 or 30\n')
        sys.exit()

    # Use 1 degree snap point and "cellsize" to get 1x1 degree tiles
    tile_osr = gdc.epsg_osr(4269)
    tile_buffer = 0.5
    tile_x, tile_y, tile_cs = 0, 0, 1

    scratch_ws = os.path.join(gis_ws, 'scratch')
    zone_raster_path = os.path.join(scratch_ws, 'zone_raster.img')
    zone_polygon_path = os.path.join(scratch_ws, 'zone_polygon.shp')

    # Error checking
    if not os.path.isfile(zone_raster_path):
        logging.error(
            '\nERROR: The zone raster {} does not exist'
            '\n  Try re-running "build_study_area_raster.py"'.format(
                zone_raster_path))
        sys.exit()
    if mask_flag and not os.path.isfile(zone_polygon_path):
        logging.error(
            '\nERROR: The zone polygon {} does not exist and mask_flag=True'
            '\n  Try re-running "build_study_area_raster.py"'.format(
                zone_raster_path))
        sys.exit()
    if not os.path.isdir(tile_ws):
        os.makedirs(tile_ws)

    # Reference all output rasters zone raster
    zone_raster_ds = gdal.Open(zone_raster_path)
    output_osr = gdc.raster_ds_osr(zone_raster_ds)
    # output_wkt = gdc.raster_ds_proj(zone_raster_ds)
    output_cs = gdc.raster_ds_cellsize(zone_raster_ds)[0]
    output_x, output_y = gdc.raster_ds_origin(zone_raster_ds)
    output_extent = gdc.raster_ds_extent(zone_raster_ds)
    output_ullr = output_extent.ul_lr_swap()
    zone_raster_ds = None
    logging.debug('\nStudy area properties')
    logging.debug('  Output OSR: {}'.format(output_osr))
    logging.debug('  Output Extent: {}'.format(output_extent))
    logging.debug('  Output cellsize: {}'.format(output_cs))
    logging.debug('  Output UL/LR: {}'.format(output_ullr))

    if mask_flag:
        # Keep tiles that intersect zone polygon
        lat_lon_list = polygon_tiles(
            zone_polygon_path, tile_osr, tile_x, tile_y, tile_cs,
            tile_buffer=0)
    else:
        # Keep tiles that intersect zone raster extent
        # Project study area extent to DEM tile coordinate system
        tile_extent = output_extent.project(output_osr, tile_osr)
        logging.debug('Output Extent: {}'.format(tile_extent))

        # Extent needed to select 1x1 degree tiles
        tile_extent.buffer(tile_buffer)
        tile_extent.adjust_to_snap(tile_x, tile_y, tile_cs, method='EXPAND')
        logging.debug('Tile Extent: {}'.format(tile_extent))

        # Get list of available tiles that intersect the extent
        lat_lon_list = sorted(list(set([
            (lat, -lon)
            for lon in range(int(tile_extent.xmin), int(tile_extent.xmax))
            for lat in range(int(tile_extent.ymax), int(tile_extent.ymin), -1)])))

    # Retrieve a list of files available on the FTP server (keyed by lat/lon)
    zip_files = {
        m.group(1): x
        for x in util.ftp_file_list(site_url, site_folder)
        for m in [re.search('[\w]*(n\d{2}w\d{3})[\w]*.zip', x)] if m}
    # logging.debug(zip_files)

    # Attempt to download the tiles
    logging.debug('Downloading')
    for lat_lon in lat_lon_list:
        logging.info('  {}'.format(lat_lon))
        lat_lon_key = 'n{:02d}w{:03d}'.format(*lat_lon)

        try:
            zip_name = zip_files[lat_lon_key]
        except KeyError:
            logging.exception('Error finding zip file for {}'.format(lat_lon))
        zip_url = '/'.join([site_url, site_folder, zip_name])
        zip_path = os.path.join(tile_ws, zip_name)

        tile_path = os.path.join(tile_ws, '{}.img'.format(lat_lon_key))

        logging.debug(zip_url)
        logging.debug(zip_path)
        logging.debug('  {}'.format(tile_path))
        if os.path.isfile(tile_path):
            if not overwrite_flag:
                logging.debug('  Skipping')
                continue
            else:
                logging.debug('  Removing')
                os.remove(tile_path)

        logging.debug('  Downloading')
        util.ftp_download(site_url, site_folder, zip_name, zip_path)

        logging.debug('  Extracting')
        try:
            zip_f = zipfile.ZipFile(zip_path)
            img_name = [x for x in zip_f.namelist()
                        if re.search('[\w]*(n\d{2}w\d{3})[\w]*.img$', x)][0]
            img_path = os.path.join(tile_ws, img_name)
            zip_f.extract(img_name, tile_ws)
            zip_f.close()
            os.rename(img_path, tile_path)
        except Exception as e:
            logging.info('  Unhandled exception: {}'.format(e))

        try:
            os.remove(zip_path)
        except Exception as e:
            logging.info('  Unhandled exception: {}'.format(e))


def polygon_tiles(input_path, tile_osr=gdc.epsg_osr(4269),
                  tile_x=0, tile_y=0, tile_cs=1, tile_buffer=0.5):
    """"""
    lat_lon_list = []
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    input_ds = shp_driver.Open(input_path, 0)
    input_layer = input_ds.GetLayer()
    input_osr = input_layer.GetSpatialRef()
    input_ftr = input_layer.GetNextFeature()
    while input_ftr:
        input_fid = input_ftr.GetFID()
        logging.debug('  {0}'.format(input_fid))
        input_geom = input_ftr.GetGeometryRef()

        # This finds the tiles that intersect the extent of each feature
        input_extent = gdc.extent(input_geom.GetEnvelope())
        input_extent = input_extent.ogrenv_swap()
        logging.debug('  Feature Extent: {}'.format(input_extent))

        # Project feature extent to the DEM tile coordinate system
        tile_extent = input_extent.project(input_osr, tile_osr)
        logging.debug('  Feature Extent: {}'.format(tile_extent))

        # Extent needed to select 1x1 degree tiles
        tile_extent.buffer(tile_buffer)
        tile_extent.adjust_to_snap(tile_x, tile_y, tile_cs, method='EXPAND')
        logging.debug('  Tile Extent: {}'.format(tile_extent))

        # Get list of available tiles that intersect the extent
        lat_lon_list.extend([
            (lat, -lon)
            for lon in range(int(tile_extent.xmin), int(tile_extent.xmax))
            for lat in range(int(tile_extent.ymax), int(tile_extent.ymin), -1)])
        del input_extent, tile_extent

        # # This finds the tiles that intersect the geometry of each feature
        # # Project the feature geometry to the DEM tile coordinate system
        # output_geom = input_geom.Clone()
        # output_geom.Transform(tx)
        # output_geom = output_geom.Buffer(tile_buffer)
        # logging.debug('  Geometry type: {}'.format(output_geom.GetGeometryName()))
        #
        # # Compute the upper left tile coordinate for each feature vertex
        # output_json = json.loads(output_geom.ExportToJson())
        # # DEADBEEF - Add a point adjust_to_snap method
        # if output_geom.GetGeometryName() == 'POLYGON':
        #     _list = sorted(list(set([
        #        (int(math.ceil((pnt[1] - tile_y) / tile_cs) * tile_cs + tile_y),
        #         -int(math.floor((pnt[0] - tile_x) / tile_cs) * tile_cs + tile_x))
        #        for ring in output_json['coordinates']
        #        for pnt in ring])))
        # elif output_geom.GetGeometryName() == 'MULTIPOLYGON':
        #     _list = sorted(list(set([
        #        (int(math.ceil((pnt[1] - tile_y) / tile_cs) * tile_cs + tile_y),
        #         -int(math.floor((pnt[0] - tile_x) / tile_cs) * tile_cs + tile_x))
        #        for poly in output_json['coordinates']
        #        for ring in poly
        #        for pnt in ring])))
        # else:
        #     .error('Invalid geometry type')
        #     .exit()
        # lat_lon_list.extend(output_list)
        # del output_geom, output_list

        # Cleanup
        input_geom = None
        del input_fid, input_geom
        input_ftr = input_layer.GetNextFeature()
    del input_ds

    return sorted(list(set(lat_lon_list)))


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Download NED tiles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--gis', nargs='?', default=os.path.join(os.getcwd(), 'gis'),
        type=lambda x: util.is_valid_directory(parser, x),
        help='GIS workspace/folder', metavar='FOLDER')
    parser.add_argument(
        '--tiles', metavar='FOLDER', required=True,
        help='Coommon DEM tiles workspace/folder')
    parser.add_argument(
        '-cs', '--cellsize', default=30, metavar='INT', type=int,
        choices=(10, 30), help='DEM cellsize (10 or 30m)')
    parser.add_argument(
        '--mask', default=None, action='store_true',
        help='Download tiles intersecting zones mask')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.gis and os.path.isdir(os.path.abspath(args.gis)):
        args.gis = os.path.abspath(args.gis)
    if args.tiles and os.path.isdir(os.path.abspath(args.tiles)):
        args.tiles = os.path.abspath(args.tiles)
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

    main(gis_ws=args.gis, tile_ws=args.tiles, dem_cs=args.cellsize,
         mask_flag=args.mask, overwrite_flag=args.overwrite)
