#--------------------------------
# Name:         et_demands_zonal_stats.py
# Purpose:      Calculate zonal stats for ET zones
#--------------------------------

import argparse
from collections import defaultdict
import datetime as dt
import logging
import os
import pprint
import sys

from osgeo import ogr, osr
import pandas as pd
import rtree
from shapely.ops import unary_union
from shapely.wkt import loads

import _arcpy
import _gdal_common as gdc
import _util as util



def main(ini_path, overwrite_flag=False):
    """Calculate zonal statistics needed to run ET-Demands model

    Parameters
    ----------
    ini_path : str
        File path of the parameter INI file.
    overwrite_flag : boolean
        True : overwrite existing shapefile
        False : default

    Returns
    -------
    None

    """
    logging.info('\nComputing ET-Demands Zonal Stats')

    logging.debug('INI: {}'.format(ini_path))
    config = util.read_ini(ini_path, section='CROP_ET')

    gis_ws = config.get('CROP_ET', 'gis_folder')
    zone_path = config.get('CROP_ET', 'cells_path')
    crop_path = config.get('CROP_ET', 'crop_path')
    awc_path = config.get('CROP_ET', 'awc_path')
    clay_path = config.get('CROP_ET', 'clay_path')
    sand_path = config.get('CROP_ET', 'sand_path')

    crop_field = config.get('CROP_ET', 'crop_field')

    crosswalk_path = config.get('CROP_ET', 'crosswalk_path')

    soil_crop_mask_flag = config.getboolean('CROP_ET', 'soil_crop_mask_flag')
    save_crop_mask_flag = config.getboolean('CROP_ET', 'save_crop_mask_flag')

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

    # +/- buffer distance (in zone units)
    simplify_threshold = 0.01

    sqm_2_acres = 0.000247105381
    sqft_2_acres = 0.0000229568

    # Check if crosswalk file exists
    if not os.path.isfile(zone_path):
        logging.error(
            '\nERROR: The ET zone shapefile doesn\'t exist, exiting\n'
            '  {}'.format(zone_path))
        sys.exit()
    elif not os.path.isfile(crop_path):
        logging.error(
            '\nERROR: The crop shapefile doesn\'t exist, exiting\n'
            '  {}'.format(crop_path))
        sys.exit()
    elif not os.path.isfile(crosswalk_path):
        logging.error(
            '\nERROR: The CDL Crosswalk file does not exist, exiting\n'
            ' Check the filename:  {}'.format(crosswalk_path))
        sys.exit()

    # Scratch files
    scratch_ws = os.path.join(gis_ws, 'scratch')
    if not os.path.isdir(scratch_ws):
        os.makedirs(scratch_ws)
    zone_crop_path = os.path.join(scratch_ws, 'zone_crop.shp')

    # if os.name == 'posix':
    #     shell_flag = False
    # else:
    #     shell_flag = True

    shp_driver = ogr.GetDriverByName('ESRI Shapefile')

    # Master zonal stats dictionary
    crop_stats = defaultdict(dict)
    zone_stats = defaultdict(dict)

    # Link zones to crops (and crops to zones, but this isn't being used)
    zone_crops = defaultdict(list)
    # crop_zones = defaultdict(list)

    # # Copy the zone_path
    # if overwrite_flag and _arcpy.exists(et_cells_path):
    #     _arcpy.delete(et_cells_path)
    # # Just copy the input shapefile
    # if not _arcpy.exists(et_cells_path):
    #     _arcpy.copy(zone_path, et_cells_path)

    # Add lat/lon fields
    logging.info('\nAdding Fields')
    zone_field_list = _arcpy.list_fields(zone_path)
    if cell_lat_field not in zone_field_list:
        logging.debug('  {}'.format(cell_lat_field))
        _arcpy.add_field(zone_path, cell_lat_field, ogr.OFTReal)
    if cell_lon_field not in zone_field_list:
        logging.debug('  {}'.format(cell_lon_field))
        _arcpy.add_field(zone_path, cell_lon_field, ogr.OFTReal)

    # # Cell ID/name
    # if cell_id_field not in zone_field_list:
    #     logging.debug('  {}'.format(cell_id_field))
    #     _arcpy.add_field(zone_path, cell_id_field, ogr.OFTString, width=24)
    # if cell_name_field not in zone_field_list:
    #     logging.debug('  {}'.format(cell_name_field))
    #     _arcpy.add_field(zone_path, cell_name_field, ogr.OFTString,
    #                      width=48)

    # Add soil fields
    if awc_field not in zone_field_list:
        logging.debug('  {}'.format(awc_field))
        _arcpy.add_field(zone_path, awc_field, ogr.OFTReal)
    if clay_field not in zone_field_list:
        logging.debug('  {}'.format(clay_field))
        _arcpy.add_field(zone_path, clay_field, ogr.OFTReal)
    if sand_field not in zone_field_list:
        logging.debug('  {}'.format(sand_field))
        _arcpy.add_field(zone_path, sand_field, ogr.OFTReal)
    if awc_in_ft_field not in zone_field_list:
        logging.debug('  {}'.format(awc_in_ft_field))
        _arcpy.add_field(zone_path, awc_in_ft_field, ogr.OFTReal,
                         width=8, precision=4)
    if hydgrp_num_field not in zone_field_list:
        logging.debug('  {}'.format(hydgrp_num_field))
        _arcpy.add_field(zone_path, hydgrp_num_field, ogr.OFTInteger)
    if hydgrp_field not in zone_field_list:
        logging.debug('  {}'.format(hydgrp_field))
        _arcpy.add_field(zone_path, hydgrp_field, ogr.OFTString, width=1)

    if acreage_field not in zone_field_list:
        logging.debug('  {}'.format(acreage_field))
        _arcpy.add_field(zone_path, acreage_field, ogr.OFTReal)

    # Crop fields are only added for needed crops below
    # for crop_num in crop_num_list:
    #     field_name = 'CROP_{0:02d}'.format(crop_num)
    #     if field_name not in zone_field_list:
    #         logging.debug('  {}'.format(field_name))
    #         _arcpy.add_field(zone_path, field_name, ogr.OFTInteger)

    # Rebuild the field list
    zone_field_list = _arcpy.list_fields(zone_path)

    # Update field width/precision
    logging.debug('\nUpdating ET zone field width and precision')
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    input_ds = shp_driver.Open(zone_path, 1)
    input_lyr = input_ds.GetLayer()
    input_defn = input_lyr.GetLayerDefn()
    logging.debug('  {:<10} {:<10} {:<5} {:<9}'.format(
        'Name', 'Type', 'Width', 'Precision'))
    for i in range(input_defn.GetFieldCount()):
        fieldDefn = input_defn.GetFieldDefn(i)
        copyDefn = ogr.FieldDefn(fieldDefn.GetName(), fieldDefn.GetType())
        logging.debug('  {:<10s} {:<10s} {:>5d} {:>9d}'.format(
            fieldDefn.GetName(),
            fieldDefn.GetFieldTypeName(fieldDefn.GetType()),
            fieldDefn.GetWidth(), fieldDefn.GetPrecision()
        ))
        if (fieldDefn.GetFieldTypeName(fieldDefn.GetType()) == 'Real' and
                fieldDefn.GetWidth() < 24 and
                fieldDefn.GetPrecision() > 0):
            copyDefn.SetWidth(24)
            copyDefn.SetPrecision(15)
        else:
            continue
        input_lyr.AlterFieldDefn(i, copyDefn, (ogr.ALTER_WIDTH_PRECISION_FLAG))
    input_ds = None

    # Calculate lat/lon
    logging.info('\nCalculating ET zone lat/lon')
    cell_lat_lon_func(zone_path, cell_lat_field, cell_lon_field)

    # Load the ET zones shapefile geometries into memory
    # Build the spatial index in the zone spatial reference
    logging.debug('\nReading ET zone shapefile features')
    zone_full_rtree = rtree.index.Index()
    zone_full_wkt_dict = dict()
    zone_full_ds = shp_driver.Open(zone_path, 0)
    zone_full_lyr = zone_full_ds.GetLayer()
    zone_full_osr = zone_full_lyr.GetSpatialRef()

    # Check that the ET zones shapefile is in a projected coordinate system
    if zone_full_osr.IsGeographic():
        logging.error('\nERROR: The ET zones shapefile must be in a '
                      'projected coordinate system, exiting')
        sys.exit()
    zone_full_unit = zone_full_osr.GetLinearUnitsName()
    if zone_full_unit.upper() not in ['METER', 'METERS','METRE']:
        logging.error('\nERROR: Unsupported unit type: {}'.format(
            zone_full_unit))
        sys.exit()

    for zone_ftr in zone_full_lyr:
        zone_fid = zone_ftr.GetFID()
        zone_geom = zone_ftr.GetGeometryRef()
        zone_geom = zone_geom.Buffer(0)
        zone_extent = gdc.Extent(zone_geom.GetEnvelope())
        zone_extent = zone_extent.ogrenv_swap()
        zone_full_rtree.insert(zone_fid, list(zone_extent))
        zone_full_wkt_dict[zone_fid] = zone_geom.ExportToWkt()
    zone_full_ds = None

    # DEADBEEF - Commented out for testing
    # Read the crop shapefile and identify intersecting features
    logging.debug('\nReading crop shapefile features')
    crop_dict = defaultdict(dict)
    crop_ds = shp_driver.Open(crop_path, 0)
    crop_lyr = crop_ds.GetLayer()
    crop_osr = crop_lyr.GetSpatialRef()
    crop_tx = osr.CoordinateTransformation(crop_osr, zone_full_osr)
    # crop_lyr_name = zones_lyr.GetName()
    for crop_ftr in crop_lyr:
        crop_fid = crop_ftr.GetFID()
        # if crop_fid % 100000 == 0 and crop_fid != 0:
            # print('test')
            # logging.info('FID: {}'.format(crop_fid))
        crop_geom = crop_ftr.GetGeometryRef()
        proj_geom = crop_geom.Clone()
        proj_geom.Transform(crop_tx)
        proj_geom = proj_geom.Buffer(0)
        proj_extent = gdc.Extent(proj_geom.GetEnvelope())
        proj_extent = proj_extent.ogrenv_swap()
        zone_fid_list = list(zone_full_rtree.intersection(list(proj_extent)))
        if not zone_fid_list:
            continue

        # Link zones to crops and crops to zones
        for zone_fid in zone_fid_list:
            zone_crops[zone_fid].append(crop_fid)
        # crop_zones[crop_fid] = zone_fid_list

        crop_dict[crop_fid] = {
            'fid': crop_fid,
            'wkt' :proj_geom.ExportToWkt(),
            'value': crop_ftr.GetField(crop_field),
        }
    crop_ds = None

    # Read ET demands crop number crosswalk
    # Link ET demands crop number (1-84) with input crop values (i.e. CDL)
    # Key is input crop number, value is crop number, ignore comment
    # note that crosswalk list cannot contain NaN or empty entries
    # NaN in etd_no forces data to type float and breaks str/int logic below (line 284)
    logging.info('\nReading Crop Crosswalk File\n  {}'.format(crosswalk_path))
    cross_df = pd.read_csv(crosswalk_path)
    cross_dict = dict()
    for index, row in cross_df.iterrows():
        # cross_dict[int(row.cdl_no)] = list(map(int, str(row.etd_no).split(',')))
        cross_dict[row.cdl_no] = list(map(int, str(row.etd_no).split(',')))
    # logging.debug(crop_num_dict)

    # Build the crop list
    # Because the spatial index is extent based,
    #   this may include crops that don't intersect the zones.
    input_crops = sorted(list(set(c['value'] for c in crop_dict.values())))
    try:
        etd_crops = sorted(list(set(
            x for c in crop_dict.values() for x in cross_dict[c['value']])))
    except KeyError as e:
        logging.error('\nError: Input crop not found in crosswalk file. '
                      'Missing Crop: {}\n Exiting.'.format(e))
        sys.exit()

    logging.info('\nInput Crops: {}'.format(', '.join(map(str, input_crops))))
    logging.info('Demands Crops: {}'.format(', '.join(map(str, etd_crops))))

    # Build the crop clipped ET zones shapefile
    # The shapefile only needs to be saved if the soils are being masked to
    #   the agricultural areas.  It would probably be possibly to avoid saving
    #   and keep the geometries in memory instead.
    if save_crop_mask_flag:
        logging.info('\nBuilding crop clipped zone shapefile')
        if os.path.exists(zone_crop_path):
            shp_driver.DeleteDataSource(zone_crop_path)
        zone_crop_ds = shp_driver.CreateDataSource(zone_crop_path)
        zone_crop_lyr_name = os.path.splitext(
            os.path.basename(zone_crop_path))[0]
        zone_crop_lyr = zone_crop_ds.CreateLayer(
            zone_crop_lyr_name, geom_type=ogr.wkbPolygon)
        zone_crop_lyr.CreateField(ogr.FieldDefn('ZONE_FID', ogr.OFTInteger))

    if soil_crop_mask_flag:
        zone_crop_rtree = rtree.index.Index()
        zone_crop_wkt_dict = dict()

    # Process crops (by zone) and compute area weighted stats
    # Write clipped zones (if necessary)
    logging.info('\nComputing crop area/type zonal stats')
    for zone_fid, crop_fid_list in sorted(zone_crops.items()):
        # if zone_fid % 1000 == 0 and zone_fid != 0:
        # logging.info('ZONE FID: {}'.format(zone_fid))
        logging.info('ZONE FID: {}'.format(zone_fid))
        logging.debug('CROP FID: {}'.format(crop_fid_list))
        if not crop_fid_list:
            logging.debug('  No crop FIDs, skipping zone')
            continue

        zone_poly = loads(zone_full_wkt_dict[zone_fid])
        zone_crop_polys = []
        zone_crop_area = 0

        # Initialize zonal stats crop acreages
        for etd_crop in etd_crops:
            field = 'CROP_{:02d}'.format(etd_crop)
            crop_stats[zone_fid][field] = 0
        crop_stats[zone_fid][acreage_field] = 0

        # Process all intersecting/neighboring crops
        for crop_fid in crop_fid_list:
            input_crop_dict = crop_dict[crop_fid]
            crop_value = input_crop_dict['value']
            crop_poly = loads(input_crop_dict['wkt'])

            clip_poly = zone_poly.intersection(crop_poly)

            if not clip_poly or clip_poly.is_empty:
                continue
            elif not clip_poly.is_valid:
                logging.error('\nERROR: Invalid clip geometry')
                input('ENTER')

            clip_area = clip_poly.area
            if not clip_area or clip_area <= 0:
                continue

            zone_crop_area += clip_area
            zone_crop_polys.append(clip_poly)

            for etd_crop in cross_dict[crop_value]:
                field = 'CROP_{:02d}'.format(etd_crop)
                crop_stats[zone_fid][field] += clip_area

        if soil_crop_mask_flag or save_crop_mask_flag:
            # Combine all polygons/multipolygons into a single multipolygon
            zone_crop_poly = unary_union(zone_crop_polys)\
                .buffer(simplify_threshold).buffer(-simplify_threshold)
            # .simplify(simplify_threshold, preserve_topology=False)\
            # zone_crop_poly = cascaded_union(zone_crop_polys)

            if zone_crop_poly.is_empty:
                logging.debug('  ZONE FID: {} - empty polygon, skipping'.format(
                    zone_fid))
                continue

            if soil_crop_mask_flag:
                # Save the crop masked zone to memory
                zone_crop_rtree.insert(zone_fid, list(zone_crop_poly.bounds))
                zone_crop_wkt_dict[zone_fid] = zone_crop_poly.wkt

            if save_crop_mask_flag:
                # Write the crop masked zone to shapefile
                zone_ftr = ogr.Feature(zone_crop_lyr.GetLayerDefn())
                zone_ftr.SetField('ZONE_FID', zone_fid)
                zone_ftr.SetGeometry(ogr.CreateGeometryFromWkt(zone_crop_poly.wkt))
                zone_crop_lyr.CreateFeature(zone_ftr)
                zone_ftr = None

    if save_crop_mask_flag:
        zone_crop_ds.ExecuteSQL("RECOMPUTE EXTENT ON {}".format(
            zone_crop_lyr_name))
        zone_crop_ds = None

        # Write projection/spatial reference to prj file
        # Format OSR as ESRI WKT
        prj_osr = zone_full_osr.Clone()
        prj_osr.MorphToESRI()
        with open(zone_crop_path.replace('.shp', '.prj'), 'w') as prj_f:
            prj_f.write(prj_osr.ExportToWkt())

    # Rebuild the crop list from the stats
    crop_field_list = sorted(list(set([
        crop_field for zone_crop_dict in crop_stats.values()
        for crop_field in zone_crop_dict.keys()])))
    logging.info('\nCrop Fields: {}'.format(', '.join(map(str, etd_crops))))


    logging.debug('\nAdding crop fields to zones shapefile')
    for crop_field in crop_field_list:
        if crop_field not in zone_field_list:
            logging.debug('  Field: {}'.format(crop_field))
            _arcpy.add_field(zone_path, crop_field, ogr.OFTReal)

    logging.debug('\nConverting crop areas to acres (if needed)')
    zone_ds = shp_driver.Open(zone_path, 0)
    zone_lyr = zone_ds.GetLayer()
    zone_osr = zone_lyr.GetSpatialRef()
    zone_unit = zone_osr.GetLinearUnitsName()
    if zone_unit.upper() not in ['METER', 'METERS', 'METRE']:
        raise ValueError('Unsupported unit type: {}'.format(zone_unit))
    for zone_fid, crop_stat_dict in crop_stats.items():
        for crop_field, crop_area in crop_stat_dict.items():
            if crop_area < 0:
                continue
            elif crop_field not in crop_field_list:
                continue
            elif zone_unit.upper() in ['METER', 'METERS', 'METRE']:
                crop_stats[zone_fid][crop_field] = crop_area * sqm_2_acres
            # elif zone_unit in ['Feet']:
            #     crop_stats[zone_fid][crop_field] = crop_area * sqft_2_acres

        # Compute total crop acreage per zone
        crop_stats[zone_fid][acreage_field] = sum(crop_stats[zone_fid].values())

    logging.debug('\nWriting crop zonal stats to zones shapefile')
    _arcpy.update_cursor(zone_path, crop_stats)

    # NOTE - Defining here to avoid passing zone_stats as an input
    def zonal_stats(input_path, input_field, zone_wkt_dict, zone_rtree):
        logging.debug('\nComputing {} zonal stats'.format(input_field))
        total_dict = dict()
        area_dict = dict()

        # Read the soil shapefile and identify intersecting features
        input_ds = shp_driver.Open(input_path, 0)
        input_lyr = input_ds.GetLayer()
        input_osr = input_lyr.GetSpatialRef()
        input_tx = osr.CoordinateTransformation(input_osr, zone_osr)
        for input_ftr in input_lyr:
            # input_fid = input_ftr.GetFID()
            input_value = input_ftr.GetField(input_field)
            #  added .Buffer(0) to clean up soil .shp intersecting geom 8/19/20
            input_geom = input_ftr.GetGeometryRef().Buffer(0)
            # input_geom = input_ftr.GetGeometryRef()
            proj_geom = input_geom.Clone()
            proj_geom.Transform(input_tx)
            input_poly = loads(proj_geom.ExportToWkt())

            proj_extent = gdc.Extent(proj_geom.GetEnvelope())
            proj_extent = proj_extent.ogrenv_swap()
            zone_fid_list = list(zone_rtree.intersection(list(proj_extent)))
            if not zone_fid_list:
                continue

            # Process all intersecting/neighboring features
            for zone_fid in zone_fid_list:
                try:
                    zone_poly = loads(zone_wkt_dict[zone_fid])
                except KeyError:
                    continue
                clip_poly = zone_poly.intersection(input_poly)
                if not clip_poly or clip_poly.is_empty:
                    continue
                elif not clip_poly.is_valid:
                    logging.error('\nERROR: Invalid clip geometry')
                    input('ENTER')
                elif not clip_poly.area or clip_poly.area <= 0:
                    continue

                if zone_fid not in total_dict.keys():
                    total_dict[zone_fid] = 0
                    area_dict[zone_fid] = 0

                if clip_poly.area > 0:
                    total_dict[zone_fid] += input_value * clip_poly.area
                    area_dict[zone_fid] += clip_poly.area
        input_ds = None

        # Compute area weighted values and save to master zonal stats dict
        for zone_fid, zone_total in total_dict.items():
            zone_stats[zone_fid][input_field] = zone_total / area_dict[zone_fid]

    if soil_crop_mask_flag:
        # # Load the crop masked zone shapefile
        # logging.debug('\nReading zone crop mask shapefile features into
        #  memory')
        # zone_crop_rtree = rtree.index.Index()
        # zone_crop_wkt_dict = dict()
        # zone_crop_ds = shp_driver.Open(zone_crop_path, 0)
        # zone_crop_lyr = zone_crop_ds.GetLayer()
        # for zone_crop_ftr in zone_crop_lyr:
        #     zone_crop_fid = zone_crop_ftr.GetFID()
        #     zone_fid = zone_crop_ftr.GetField('ZONE_FID')
        #     zone_crop_geom = zone_crop_ftr.GetGeometryRef()
        #     if not zone_crop_geom:
        #         continue
        #     zone_crop_geom = zone_crop_geom.Buffer(0)
        #     zone_crop_extent = gdc.Extent(zone_crop_geom.GetEnvelope())
        #     zone_crop_extent = zone_crop_extent.ogrenv_swap()
        #     zone_crop_rtree.insert(zone_crop_fid, list(zone_crop_extent))
        #     zone_crop_wkt_dict[zone_fid] = zone_crop_geom.ExportToWkt()
        # zone_crop_ds = None

        # Compute soil zonal stats for the crop masked ET zones
        # Process files separately even though geometries are probably the same
        zonal_stats(awc_path, awc_field, zone_crop_wkt_dict, zone_crop_rtree)
        zonal_stats(clay_path, clay_field, zone_crop_wkt_dict, zone_crop_rtree)
        zonal_stats(sand_path, sand_field, zone_crop_wkt_dict, zone_crop_rtree)

    else:
        # Compute soil zonal stats for the full ET zones
        # Process files separately even though geometries are probably the same
        zonal_stats(awc_path, awc_field, zone_full_wkt_dict, zone_full_rtree)
        zonal_stats(clay_path, clay_field, zone_full_wkt_dict, zone_full_rtree)
        zonal_stats(sand_path, sand_field, zone_full_wkt_dict, zone_full_rtree)

    logging.debug('\nWriting soil zonal stats to zones shapefile')
    _arcpy.update_cursor(zone_path, zone_stats)

    # Calculate AWC in in/feet
    logging.info('Calculating AWC in in/ft')
    _arcpy.calculate_field(
        zone_path, awc_in_ft_field, '!{}! * 12'.format(awc_field))

    # Calculate hydrologic group
    logging.info('Calculating hydrologic group')
    fields = (clay_field, sand_field, hydgrp_num_field, hydgrp_field)
    values = _arcpy.search_cursor(zone_path, fields)
    for fid, row in values.items():
        if row[sand_field] > 50:
            values[fid][hydgrp_num_field], values[fid][hydgrp_field] = 1, 'A'
        elif row[clay_field] > 40:
            values[fid][hydgrp_num_field], values[fid][hydgrp_field] = 3, 'C'
        else:
            values[fid][hydgrp_num_field], values[fid][hydgrp_field] = 2, 'B'
    _arcpy.update_cursor(zone_path, values)


#     # Save by FID/feature for easier writing to shapefile
#     logging.debug('\nParsing crop zonal stats')
#     zone_crop_dict = {}
#     crop_field_fmt = 'CROP_{:02d}'
#     for fid, ftr in enumerate(zs):
#         # logging.debug('FID: {}'.format(i))
#         zone_crop_dict[fid] = {}
#         for cdl_str, cdl_pixels in ftr.items():
#             # logging.debug('  {} {}'.format(cdl_str, cdl_pixels))
#             cdl_number = int(cdl_str)
#             # Only 'crops' have a crop number (no shrub, water, urban, etc.)
#             if cdl_number not in crop_num_dict.keys():
#                 if cdl_number != 0:
#                     logging.debug('  Skipping CDL {}'.format(cdl_number))
#                 continue
#
#             # Crop number can be an integer or list of integers (double crops)
#             crop_number = crop_num_dict[cdl_number]
#
#             # Crop numbers of -1 are for crops that haven't been linked
#             #   to a CDL number
#             if not crop_number or crop_number == -1:
#                 logging.warning('  Missing CDL {}'.format(cdl_number))
#                 continue
#
#             crop_acreage = cdl_pixels * sqm_2_acres * snap_cs ** 2
#
#             # Save acreage for both double crops
#             for c in crop_number:
#                 crop_field = crop_field_fmt.format(c)
#                 if crop_field not in zone_crop_dict[fid].keys():
#                     zone_crop_dict[fid][crop_field] = crop_acreage
#                 else:
#                     zone_crop_dict[fid][crop_field] += crop_acreage


def cell_lat_lon_func(input_path, lat_field, lon_field):
    """"""

    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    input_ds = shp_driver.Open(input_path, 1)
    input_lyr = input_ds.GetLayer()
    input_osr = input_lyr.GetSpatialRef()
    # input_osr.MorphToESRI()
    output_osr = input_osr.CloneGeogCS()
    # output_osr = osr.SpatialReference()
    # output_osr.ImportFromEPSG(4326)

    for input_ftr in input_lyr:
        input_fid = input_ftr.GetFID()
        # logging.debug('  FID: {}'.format(input_fid))
        input_geom = input_ftr.GetGeometryRef()
        # Project geometry to 4326 then extract geometry centroid
        centroid_geom = input_geom.Clone()
        centroid_geom.Transform(
            osr.CoordinateTransformation(input_osr, output_osr))
        centroid_geom = centroid_geom.Centroid()
        input_ftr.SetField(lat_field, centroid_geom.GetY())
        input_ftr.SetField(lon_field, centroid_geom.GetX())
        input_lyr.SetFeature(input_ftr)
    input_ds = None


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
