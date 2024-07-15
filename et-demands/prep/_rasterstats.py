#--------------------------------
# Name:         gdal_common.py
# Purpose:      Mimic rasterstats module
#               https://github.com/perrygeo/python-rasterstats
#--------------------------------

import logging
import math

import numpy as np
from osgeo import gdal, ogr, osr

logger = logging.getLogger(__name__)


def zonal_stats(*args, **kwargs):
    """"""
    return list(gen_zonal_stats(*args, **kwargs))


def gen_zonal_stats(
        vectors, raster,
        band=1,
        stats=None,
        categorical=False,
        **kwargs):
    """"""
    logger.debug('Computing Zonal Statistics')

    # Should the raster be kept open or reopened for each feature?
    raster_ds = gdal.Open(raster, 0)
    raster_band = raster_ds.GetRasterBand(band)
    raster_nodata = raster_band.GetNoDataValue()
    raster_proj = raster_ds.GetProjection()
    raster_osr = osr.SpatialReference()
    raster_osr.ImportFromWkt(raster_proj)
    raster_geo = raster_ds.GetGeoTransform()
    raster_rows = raster_ds.RasterYSize
    raster_cols = raster_ds.RasterXSize
    cs = abs(raster_geo[1])
    raster_x = raster_geo[0]
    raster_y = raster_geo[3]
    raster_extent = [raster_x, raster_y - raster_rows * cs,
                     raster_x + raster_cols * cs, raster_y]

    # For now, hardcode to only process shapefile vector files
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_ds = vector_driver.Open(vectors, 0)
    vector_lyr = vector_ds.GetLayer()
    vector_osr = vector_lyr.GetSpatialRef()

    # Project vector geometry to the raster spatial reference
    vector_tx = osr.CoordinateTransformation(vector_osr, raster_osr)

    logger.debug('Raster: {}'.format(raster))
    logger.debug('  WKT: {}'.format(raster_osr.ExportToWkt()))
    logger.debug('  Rows:     {}'.format(raster_rows))
    logger.debug('  Cols:     {}'.format(raster_cols))
    logger.debug('  Extent:   {}'.format(raster_extent))
    logger.debug('  Geo:      {}'.format(raster_geo))
    logger.debug('  Cellsize: {}'.format(cs))
    logger.debug('  Snap X:   {}'.format(raster_x))
    logger.debug('  Snap Y:   {}'.format(raster_y))
    logger.debug('Vectors: {}'.format(vectors))
    logger.debug('  WKT: {}'.format(vector_osr.ExportToWkt()))

    # Iterate through the features
    for vector_ftr in vector_lyr:
        fid = vector_ftr.GetFID()

        # Project the geometry
        vector_geom = vector_ftr.GetGeometryRef()
        v_geom = vector_geom.Clone()
        v_geom.Transform(vector_tx)

        # Get the projected geometry extent
        extent = list(v_geom.GetEnvelope())

        # Convert to an OGR style extent (xmin, ymin, xmax, ymax)
        extent = [extent[0], extent[2], extent[1], extent[3]]

        # Expand the vector extent to the raster transform
        extent[0] = math.floor((extent[0] - raster_x) / cs) * cs + raster_x
        extent[1] = math.floor((extent[1] - raster_y) / cs) * cs + raster_y
        extent[2] = math.ceil((extent[2] - raster_x) / cs) * cs + raster_x
        extent[3] = math.ceil((extent[3] - raster_y) / cs) * cs + raster_y

        # TODO: Check if zone extent intersects the raster extent

        # Clip the zone extent to the raster extent
        extent[0] = max(extent[0], raster_extent[0])
        extent[1] = max(extent[1], raster_extent[1])
        extent[2] = min(extent[2], raster_extent[2])
        extent[3] = min(extent[3], raster_extent[3])

        # Compute raster properties
        cols = int((abs(extent[2] - extent[0]) / cs) + 0.5)
        rows = int((abs(extent[3] - extent[1]) / cs) + 0.5)
        geo = [extent[0], cs, 0, extent[3], 0, -cs]
        i = int(round((geo[0] - raster_geo[0]) / cs, 0))
        j = int(round((geo[3] - raster_geo[3]) / -cs, 0))

        # logger.debug('FID: {}'.format(fid))
        # logger.debug('  Rows:   {}'.format(rows))
        # logger.debug('  Cols:   {}'.format(cols))
        # logger.debug('  Extent: {}'.format(extent))
        # logger.debug('  Geo:    {}'.format(geo))
        # logger.debug('  i:      {}'.format(i))
        # logger.debug('  j:      {}'.format(j))

        # Create an in-memory dataset/layer for each feature
        v_driver = ogr.GetDriverByName('Memory')
        v_ds = v_driver.CreateDataSource('out')
        v_lyr = v_ds.CreateLayer('poly', geom_type=ogr.wkbPolygon,
                                 srs=raster_osr)
        v_feat = ogr.Feature(v_lyr.GetLayerDefn())
        v_feat.SetGeometryDirectly(v_geom)
        v_lyr.CreateFeature(v_feat)

        # Create an in-memory raster to set from the vector data
        mask_driver = gdal.GetDriverByName('MEM')
        mask_ds = mask_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetProjection(raster_proj)
        mask_ds.SetGeoTransform(geo)
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)
        mask_band.SetNoDataValue(0)
        gdal.RasterizeLayer(mask_ds, [1], v_lyr, burn_values=[1])

        # Read the vector mask array
        mask = mask_band.ReadAsArray(0, 0, cols, rows).astype(np.bool)

        # Read the data array
        array = raster_band.ReadAsArray(i, j, cols, rows)

        # Mask nodata pixels
        if (raster_nodata is not None and
                array.dtype in [np.float32, np.float64]):
            array[array == raster_nodata] = np.nan

        # Apply the zone mask
        # This might contribute to memory issues
        array = array[mask]

        if categorical and array.dtype not in [np.float32, np.float64]:
            # Compute categorical stats
            ftr_stats = dict(zip(*np.unique(array, return_counts=True)))
        else:
            ftr_stats = {stat: None for stat in stats}

            # Remove all nan values before computing statistics
            if np.any(np.isnan(array)):
                array = array[np.isfinite(array)]

            for stat in stats:
                if not np.any(array):
                    continue
                elif stat == 'mean':
                    ftr_stats[stat] = float(np.mean(array))
                elif stat == 'max':
                    ftr_stats[stat] = float(np.max(array))
                elif stat == 'min':
                    ftr_stats[stat] = float(np.min(array))
                elif stat == 'median':
                    ftr_stats[stat] = float(np.median(array))
                elif stat == 'sum':
                    ftr_stats[stat] = float(np.sum(array))
                elif stat == 'std':
                    ftr_stats[stat] = float(np.std(array))
                elif stat == 'var':
                    ftr_stats[stat] = float(np.var(array))
                elif stat == 'count':
                    ftr_stats[stat] = float(np.sum(np.isfinite(array)))
                else:
                    raise ValueError('Stat {} not supported'.format(stat))

        # Cleanup
        del array, mask
        v_ds = None
        mask_band = None
        mask_ds = None
        del v_ds, v_lyr, v_feat, v_geom, v_driver
        del mask_ds, mask_band, mask_driver

        yield ftr_stats

    # Cleanup
    vector_lyr = None
    vector_ds = None
    raster_band = None
    raster_ds = None
    del vector_ds, vector_lyr
    del raster_ds, raster_band
