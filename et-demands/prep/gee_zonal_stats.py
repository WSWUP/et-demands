#--------------------------------
# Name:         gee_zonal_stats.py
# Purpose:      Calculate zonal stats for ET zones using GEE
#--------------------------------

import ee
import argparse
import logging
import os
import argparse
from collections import defaultdict
import datetime as dt
import logging
import os
import pprint
import sys
import _util as util


ee.Initialize()


def main(ini_path, overwrite_flag=False):
    """Extract zonal statistics for ET Demands from rasters using Google Earth Engine.

    Args:
        ini_path (str): file path of the project INI file
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nExtracting ET Demands zonal statistics')

    # Read INI file
    config = util.read_ini(ini_path, section='CROP_ET')
    logging.debug('INI: {}'.format(ini_path))

    gis_ws = config.get('CROP_ET', 'gis_folder')
    soil_crop_mask_flag = config.getboolean('CROP_ET', 'soil_crop_mask_flag')
    save_crop_mask_flag = config.getboolean('CROP_ET', 'save_crop_mask_flag')
    crosswalk_path = config.get('CROP_ET', 'crosswalk_path')

    # Set GEE inputs
    et_cells_path = config.get('CROP_ET_GEE', 'cells_path')
    crop = config.get('CROP_ET_GEE', 'crop_path')
    awc = config.get('CROP_ET_GEE', 'awc_path')
    clay = config.get('CROP_ET_GEE', 'clay_path')
    sand = config.get('CROP_ET_GEE', 'sand_path')
    crop_field = config.get('CROP_ET_GEE', 'crop_field')

    # TODO: Read field names from INI
    cell_lat_field = 'LAT'
    cell_lon_field = 'LON'
    # cell_id_field = 'CELL_ID'
    # cell_name_field = 'CELL_NAME'
    # cell_station_id_field = 'STATION_ID'
    acreage_field = 'AG_ACRES'
    awc_field = 'AWC'
    clay_field = 'CLAY'
    sand_field = 'SAND'
    awc_in_ft_field = 'AWC_IN_FT'
    hydgrp_num_field = 'HYDGRP_NUM'
    hydgrp_field = 'HYDGRP'

    field_names = {'lat_field': cell_lat_field, 'lon_field': cell_lon_field, 'acreage_field': acreage_field,
                   'awc_field': awc_field, 'clay_field': clay_field, 'sand_field': sand_field,
                   'awc_in_ft_field': awc_in_ft_field, 'hydgrp_num_field': hydgrp_num_field, 'hydgrp_field': hydgrp_field}

    # +/- buffer distance (in zone units)
    simplify_threshold = 0.01

    sqm_2_acres = 0.000247105381
    sqft_2_acres = 0.0000229568

    # Initialize Earth Engine feature collections and images
    et_cells = ee.FeatureCollection(et_cells_path)
    crop_img = ee.Image(crop)

    if soil_crop_mask_flag:
        # Mask the crop layer for agricultural pixels only
        crop_mask = crop_img.remap(
            ee.List.sequence(0, 254),
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).rename('crop_mask')

        crop_img = crop_img.updateMask(crop_mask)

    # Perform zonal statistics
    zonal_stats = calculate_zonal_statistics(et_cells, crop_img, awc, clay, sand, field_names)

    # Export results
    export_results(zonal_stats, 'ET_Demands_Zonal_Stats')

    logging.info('\nDONE!')


def calculate_zonal_statistics(features, cdl, awc_fc, clay_fc, sand_fc, field_names):
    """Calculate zonal statistics using GEE"""
    cell_lat_field = field_names['lat_field']
    cell_lon_field = field_names['lon_field']
    awc_field = field_names['awc_field']
    clay_field = field_names['clay_field']
    sand_field = field_names['sand_field']
    hydgrp_num_field = field_names['hydgrp_num_field']
    hydgrp_field = field_names['hydgrp_field']

    def per_feature(feature):
        # Calculate statistics for each feature

        # Calculate centroid and reproject to WGS84
        centroid = feature.geometry().centroid().transform('EPSG:4326')

        # Extract lat/lon from centroid
        lon = centroid.coordinates().get(0)
        lat = centroid.coordinates().get(1)

        cdl_hist = cdl.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=feature.geometry(),
            scale=30,
            maxPixels=1e9
        ).get('cropland')

        # Convert the histogram to a list of crops
        cdl_crops = ee.List(ee.Dictionary(cdl_hist).keys())

        # Find intersecting features for AWC, clay, and sand
        intersecting_awc = awc_fc.filterBounds(feature.geometry())
        intersecting_clay = clay_fc.filterBounds(feature.geometry())
        intersecting_sand = sand_fc.filterBounds(feature.geometry())

        # TODO: How to set up so uses mask for the soil features? Is there raster data available?
        # Calculate mean values for intersecting features
        awc_mean = intersecting_awc.aggregate_mean('AWC')
        clay_mean = intersecting_clay.aggregate_mean('Clay')
        sand_mean = intersecting_sand.aggregate_mean('Sand')

        # Calculate AWC in in/ft
        awc_in_ft = ee.Number(awc_mean).multiply(12)

        # Calculate hydrologic group using a single comparison
        hydgrp_num = ee.Number(sand_mean).gt(50).multiply(1).add(
            ee.Number(clay_mean).gt(40).multiply(3)).add(2).clamp(1, 3)

        # Map hydrologic group number to letter
        hydgrp = ee.Dictionary({1: 'A', 2: 'B', 3: 'C'}).get(hydgrp_num)

        return feature.set({
            'cdl_crops': cdl_crops,
            awc_field: awc_mean,
            clay_field: clay_mean,
            sand_field: sand_mean,
            hydgrp_num_field: hydgrp_num,
            hydgrp_field: hydgrp,
            cell_lat_field: lat,
            cell_lon_field: lon,
        })

    return features.map(per_feature)


def export_results(results, output_path):
    """Export results to Google Drive"""
    task = ee.batch.Export.table.toDrive(
        collection=results,
        description='ET_Demands_Zonal_Stats',
        fileFormat='CSV',
        folder='ET_Demands_Output'
    )
    task.start()
    print(f"Export task started: {output_path}")


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='ET-Demands Zonal Stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, metavar='INI',
        type=lambda x: util.is_valid_file(parser, x),
        help='Input INI File')
    parser.add_argument(
        '-o', '--overwrite', default=None, action='store_true',
        help='Overwrite existing file')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.ini and os.path.isdir(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)

    return args


if __name__ == '__main__':
    # Add argument parsing here if needed
    ini_path = 'path/to/your/ini/file.ini'
    main(ini_path, overwrite_flag=False)

    if __name__ == '__main__':
        args = arg_parse()

        logging.basicConfig(level=args.loglevel, format='%(message)s')
        logging.info('\n{}'.format('#' * 80))
        logging.info('{0:<20s} {1}'.format(
            'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
        logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
        logging.info('{0:<20s} {1}'.format(
            'Script:', os.path.basename(sys.argv[0])))

        main(ini_path=args.ini, overwrite_flag=args.overwrite)