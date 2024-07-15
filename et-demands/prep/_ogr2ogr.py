import os
import subprocess


def clip(input_path, clip_path, output_path):
    """Clip a feature dataset using ogr2ogr

    Parameters
    ----------
    input_path : str
    clip_path : str
    output_path : str

    """
    if os.name == 'posix':
        shell_flag = False
    else:
        shell_flag = True

    # subprocess.run(
    subprocess.check_output(
        ['ogr2ogr', '-f', 'ESRI Shapefile', '-overwrite',
            '-preserve_fid', '-unsetFieldWidth',
            '-clipsrc', clip_path, output_path, input_path],
        shell=shell_flag)


def copy(input_path, output_path):
    """Copy a feature or raster dataset

    Parameters
    ----------
    input_path : str
    output_path : str

    """
    if os.name == 'posix':
        shell_flag = False
    else:
        shell_flag = True

    # if not is_shapefile(input_path):
    #     raise Exception('input must be a .shp type')

    # DEADBEEF - Extra command(s) remove the field type warning but make
    # the DBF quite a bit larger since all the field types are doubles.
    # subprocess.run(
    subprocess.check_output(
        ['ogr2ogr', '-f', 'ESRI Shapefile', '-overwrite', '-preserve_fid',
            '-unsetFieldWidth', output_path, input_path],
        shell=shell_flag)

    # # DEADBEEF - Only runs on Python 3 but suppresses field width warning
    # subprocess.run(
    #     ['ogr2ogr', '-f', 'ESRI Shapefile', output_path, input_path],
    #     shell=shell_flag, stderr=subprocess.DEVNULL)


def delete(input_path):
    """Delete a raster dataset

    Parameters
    ----------
    input_path : str

    """
    if os.name == 'posix':
        shell_flag = False
    else:
        shell_flag = True

    # TODO: Add a format type function to gdal_common
    if input_path.upper().endswith('.IMG'):
        input_format = 'HFA'
        input_type = 'raster'
    elif input_path.upper().endswith('.TIF'):
        input_format = 'GTiff'
        input_type = 'raster'
    elif input_path.upper().endswith('.TIFF'):
        input_format = 'GTiff'
        input_type = 'raster'

    if input_type == 'raster':
        # subprocess.run(
        subprocess.check_output(
            ['gdalmanage', 'delete', '-f', input_format, input_path],
            shell=shell_flag)
    # elif input_type == 'vector':
    else:
        raise ValueError('unsupported format for delete')


def project(input_path, output_path, output_wkt):
    """Project a feature dataset to a new projection

    Parameters
    ----------
    input_path : str
    output_path : str
    output_wkt : str

    """
    if os.name == 'posix':
        shell_flag = False
    else:
        shell_flag = True

    # subprocess.run(
    subprocess.check_output(
        ['ogr2ogr', '-f', 'ESRI Shapefile', '-overwrite',
            '-preserve_fid', '-unsetFieldWidth', '-t_srs', output_wkt,
            output_path, input_path],
        shell=shell_flag)
