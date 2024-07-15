#--------------------------------
# Name:         download_cdl_raster.py
# Purpose:      Download national CDL zips
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile

import _util as util


def main(ini_path, overwrite_flag=False):
    """Download CONUS CDL zips

    Parameters
    ----------
    ini_path : str
        File path of the parameter INI file.
    overwrite_flag : bool
        If True, overwrite existing files (the default is False).

    Returns
    -------
    None

    """
    logging.info('\nDownload and extract CONUS CDL rasters')

    logging.debug('INI: {}'.format(ini_path))
    config = util.read_ini(ini_path, section='CROP_ET')

    site_url = 'ftp.nass.usda.gov'
    site_folder = 'download/res'

    cdl_ws = config.get('CROP_ET', 'cdl_folder')
    cdl_year = int(config.get('CROP_ET', 'cdl_year'))
    cdl_format = config.get('CROP_ET', 'cdl_format')

    logging.info('Year: {}'.format(cdl_year))
    zip_name = cdl_format.format(cdl_year, 'zip')
    zip_url = site_url + '/' + zip_name
    zip_path = os.path.join(cdl_ws, zip_name)

    cdl_path = os.path.join(cdl_ws, cdl_format.format(cdl_year, 'img'))

    # zip_url_size = remote_size(zip_url)
    # if os.path.isfile(zip_path):
    #     zip_path_size = local_size(zip_path)
    # if not os.path.isfile(zip_path):
    #     zip_path_size = 0

    # if zip_url_size == zip_path_size:
    #     size_flag = False
    # if zip_url_size != zip_path_size:
    #     size_flag = True
    size_flag = False

    if not os.path.isdir(cdl_ws):
        os.makedirs(cdl_ws)

    if os.path.isfile(zip_path) and (overwrite_flag or size_flag):
        os.remove(zip_path)

    if not os.path.isfile(zip_path):
        logging.info('  Download CDL files')
        logging.debug('    {}'.format(zip_url))
        logging.debug('    {}'.format(zip_path))

        util.ftp_download(site_url, site_folder, zip_name, zip_path)
        # util.url_download(zip_url, zip_path)

    if os.path.isfile(cdl_path) and overwrite_flag:
        util.remove_file(cdl_path)

    if os.path.isfile(zip_path) and not os.path.isfile(cdl_path):
        logging.info('  Extracting CDL files')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cdl_ws)

                
def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Download CDL raster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, metavar='INI',
        type=lambda x: util.is_valid_file(parser, x),
        help='Input INI File')
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert input file to an absolute path
    if args.ini and os.path.isdir(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)

    return args


# def reporthook(count, block_size, total_size):
#     global start_time
#     if count == 0:
#         start_time = time.time()
#         return
#     duration = time.time() - start_time
#     progress_size = int(count * block_size)
#     speed = int(progress_size / (1024 * duration))
#     percent = int(count * block_size * 100 / total_size)
#     sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
#                     (percent, progress_size / (1024 * 1024), speed, duration))
#     sys.stdout.flush()


# # CGM - Need to see if this can be done through the FTP module
# def remote_size(link):
#     site = urllib.urlopen(link)
#     meta = site.info()
#     size = meta.getheaders("Content-Length")[0]
#     return size


# def local_size(path):
#     file = open(path, "rb")
#     size = len(file.read())
#     file.close()
#     return size


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Start Time:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
