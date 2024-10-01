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
    crop_coll = config.get('CROP_ET_GEE', 'crop_path')
    # awc = ee.FeatureCollection(config.get('CROP_ET_GEE', 'awc_path'))
    # clay = ee.FeatureCollection(config.get('CROP_ET_GEE', 'clay_path'))
    # sand = ee.FeatureCollection(config.get('CROP_ET_GEE', 'sand_path'))
    awc = ee.Image(config.get('CROP_ET_GEE', 'awc_path'))
    clay = ee.Image(config.get('CROP_ET_GEE', 'clay_path'))
    sand = ee.Image(config.get('CROP_ET_GEE', 'sand_path'))
    # crop_field = config.get('CROP_ET_GEE', 'crop_field')

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
    # Get the most recent crop layer
    crop_img = ee.ImageCollection(crop_coll).sort("system:time_start", False)
    crop_img = ee.Image(crop_img.first())

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


def mask_crops(image):
    # Mask for agricultural crops
    ag_mask = image.remap(
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
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).rename('ag_mask')

    masked_image = image.updateMask(ag_mask)
    return masked_image
    # return masked_image.reduceRegion(
    #     reducer=ee.Reducer.frequencyHistogram(),
    #     geometry=feature.geometry(),
    #     scale=30,
    #     maxPixels=1e9
    # ).get('cropland')



def calculate_zonal_statistics(features, crop_coll, awc_fc, clay_fc, sand_fc, field_names):
    """Calculate zonal statistics using GEE for each gridmet cell (or specified input feature)"""
    cell_lat_field = field_names['lat_field']
    cell_lon_field = field_names['lon_field']
    awc_field = field_names['awc_field']
    clay_field = field_names['clay_field']
    sand_field = field_names['sand_field']
    hydgrp_num_field = field_names['hydgrp_num_field']
    hydgrp_field = field_names['hydgrp_field']

    # Combine all crop images into a single ImageCollection
    crop_images = ee.ImageCollection(crop_coll)

    crop_images = crop_images.map(mask_crops)

    def process_image(image):
        date = ee.Date(image.get('system:time_start')).get('year')
        cdl_year = ee.String('crop_').cat(ee.String(date.format()))

        def add_crop_list(feature):
            histogram = ee.Dictionary(feature.get('cropland'))
            crop_list = histogram.keys()
            return feature.set(cdl_year, crop_list)

        return image.reduceRegions(
            collection=features,
            reducer=ee.Reducer.frequencyHistogram(),
            scale=30
        ).map(add_crop_list)

    # Map over each image in the collection
    histograms = crop_images.map(process_image).flatten()

    print("\ntest histogram", histograms.first().getInfo())
    cropProperties = histograms.first().propertyNames()
    print("\ntest cropProperties", cropProperties.getInfo())

    # Get all unique crop IDs across all features and years
    # all_crop_ids = histograms.aggregate_array('cropland').flatten().distinct()

    # Get all unique crop IDs across all features and years
    all_crop_properties = histograms.first().propertyNames().filter(ee.Filter.stringStartsWith('item', 'crop_'))
    all_crop_ids = all_crop_properties.iterate(
        lambda prop, prev: ee.List(prev).cat(histograms.aggregate_array(prop)),
        ee.List([])
    )
    all_crop_ids = ee.List(all_crop_ids).flatten().distinct().sort()

    print("\ntest all_crop_ids", all_crop_ids.getInfo())
    pause = input("Press Enter to continue...")


    # def extract_crop_codes(feature):
    #     # Get all property names containing 'crop_' and extract the crop codes
    #     cropProperties = feature.propertyNames().filter(ee.Filter.stringContains('item', 'crop_'))
    #     def extract_crop_codes(prop):
    #         return feature.get(prop)
    #
    #     cropcodes = cropProperties.map(extract_crop_codes).flatten()
    #     return ee.Feature(None, {"all_crops": cropcodes})
    #
    # cropCodeFeatures = histograms.map(extract_crop_codes)
    #
    # # print("test histograms", histograms.first().getInfo())
    # print("test cropCodeFeatures", cropCodeFeatures.first().getInfo())

    def per_feature(feature):

        feature = ee.Feature(feature)
        # Calculate centroid and reproject to WGS84
        centroid = feature.geometry().centroid().transform('EPSG:4326')
        lon = centroid.coordinates().get(0)
        lat = centroid.coordinates().get(1)

        # Get the histogram result for the cropland
        # Get all property names containing 'crop_'
        crop_properties = feature.propertyNames().filter(ee.Filter.stringStartsWith('item', 'crop_'))

        # Extract and flatten all crop lists
        feature_crops = crop_properties.map(lambda prop: feature.get(prop)).flatten().distinct()

        # Create an initial crop presence dictionary with all values set to 0
        crop_presence_default = all_crop_ids.map(lambda crop_id:
                                                 ee.String('CROP_').cat(ee.String(crop_id))
                                                 )

        default_values = ee.List.repeat(0, all_crop_ids.size())
        crop_presence_default_dict = ee.Dictionary.fromLists(crop_presence_default, default_values)

        # Create a dictionary with crop presence (1 or 0) for crops in the feature
        def map_crop_presence(crop_id):
            is_present = feature_crops.contains(ee.String(crop_id).slice(5))
            return ee.List([crop_id, ee.Number.expression('b ? 1 : 0', {'b': is_present})])

        crop_presence = crop_presence_default.map(map_crop_presence)

        crop_presence_dict = ee.Dictionary(crop_presence.flatten())

        # Combine the two dictionaries, keeping the non-zero values
        final_crop_presence = crop_presence_dict.combine(crop_presence_default_dict, False)

        # Join the crop IDs into a comma-separated string
        feature_crops = feature_crops.join(',')

        # Find intersecting features for AWC, clay, and sand
        # intersecting_awc = awc_fc.filterBounds(feature.geometry())
        # intersecting_clay = clay_fc.filterBounds(feature.geometry())
        # intersecting_sand = sand_fc.filterBounds(feature.geometry())

        # Calculate mean values for intersecting features
        # awc_mean = intersecting_awc.aggregate_mean('AWC')
        # clay_mean = intersecting_clay.aggregate_mean('Clay')
        # sand_mean = intersecting_sand.aggregate_mean('Sand')

        props = {"reducer": ee.Reducer.mean(),
                 "geometry": feature.geometry(),
                 "scale": 30,
                 "maxPixels": 1e9}


        awc_mean = awc_fc.reduceRegion(**props).get('b1')
        clay_mean = clay_fc.reduceRegion(**props).get('b1')
        sand_mean = sand_fc.reduceRegion(**props).get('b1')

        # Calculate AWC in in/ft
        awc_in_ft = ee.Number(awc_mean).multiply(12)

        # Hydrologic group calculation
        hydgrp_num = ee.Number(sand_mean).gt(50).multiply(1).add(
            ee.Number(clay_mean).gt(40).multiply(3)).add(2).clamp(1, 3)
        hydgrp_num = hydgrp_num.int().format()  # Convert to integer and format as string

        # Map hydrologic group number to letter
        hydgrp = ee.Dictionary({1: 'A', 2: 'B', 3: 'C'}).get(hydgrp_num)

        return ee.Feature(feature.geometry()).set({
            'cdl_crops': feature_crops,
            field_names['awc_field']: awc_mean,
            field_names['clay_field']: clay_mean,
            field_names['sand_field']: sand_mean,
            field_names['hydgrp_num_field']: hydgrp_num,
            field_names['hydgrp_field']: hydgrp,
            field_names['lat_field']: lat,
            field_names['lon_field']: lon,
        }).set(final_crop_presence)

    return histograms.map(per_feature)




def export_results(results, output_path):
    first = results.first().getInfo()
    print("test first result", first)
    pause = input("Press Enter to continue...")

    """Export results to Google Drive"""
    date = dt.datetime.now().strftime('%Y%m%d')
    print("date", date)
    descrip = 'ET_Demands_Zonal_Stats_{}'.format(date)

    task = ee.batch.Export.table.toDrive(
        collection=results,
        description= descrip,
        fileFormat='SHP',
        folder='ET_Demands_Output'
    )
    task.start()
    print(f"Export task started: {descrip}")


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
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)