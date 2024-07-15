#--------------------------------
# Name:         util.py
# Purpose:      Utility functions for ET-Demands prep scripts
#--------------------------------

import configparser
from ftplib import FTP
import glob
from itertools import groupby
import logging
import os
import sys


def ftp_download(site_url, site_folder, file_name, output_path):
    """"""
    try:
        ftp = FTP()
        ftp.connect(site_url)
        ftp.login()
        ftp.cwd('{}'.format(site_folder))
        logging.debug('  Beginning download')
        ftp.retrbinary('RETR %s' % file_name, open(output_path, 'wb').write)
        logging.debug('  Download complete')
        ftp.quit()
    except Exception as e:
        logging.info('  Unhandled exception: {}'.format(e))


def ftp_file_list(site_url, site_folder):
    """"""
    try:
        ftp = FTP()
        ftp.connect(site_url)
        ftp.login()
        ftp.cwd('{}'.format(site_folder))
        files = ftp.nlst()
        ftp.quit()
    except Exception as e:
        logging.info('  Unhandled exception: {}'.format(e))
        files = []
    return files


def is_valid_file(parser, arg):
    """"""
    if os.path.isfile(arg):
        return arg
    elif os.path.isfile(os.path.abspath(arg)):
        return os.path.abspath(arg)
    else:
        parser.error('\nThe file {} does not exist!'.format(arg))


def is_valid_directory(parser, arg):
    """"""
    if os.path.isdir(arg):
        return arg
    elif os.path.isdir(os.path.abspath(arg)):
        return os.path.abspath(arg)
    else:
        parser.error('\nThe directory {} does not exist!'.format(arg))


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()
    # tokens are comma separated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def read_ini(ini_path, section):
    """Open the INI file and check for obvious errors

    Notes
    -----

    """
    logging.info('  INI: {}'.format(os.path.basename(ini_path)))
    config = configparser.ConfigParser()

    try:
        config.read_file(open(ini_path, 'r'))
        # This doesn't raise an exception when the file doesn't exist
        # config.read(ini_path)
    except [FileNotFoundError, IOError] as e:
        logging.error('\nERROR: INI file does not exist\n'
                      '  {}\n'.format(ini_path))
        sys.exit()
    except configparser.MissingSectionHeaderError:
        logging.error('\nERROR: INI file is missing a section header'
                      '\n  Please make sure the following line is at the '
                      'beginning of the file\n[{}]\n'.format(section))
        sys.exit()
    except Exception as e:
        logging.error('\nERROR: Unhandled exception reading INI file:'
                      '\n  {}\n'.format(ini_path, e))
        logging.error('{}\n'.format(e))
        sys.exit()

    return config


def ranges(i):
    """Join sequential values into ranges"""
    for a, b in groupby(enumerate(sorted(i)), lambda t: t[1] - t[0]):
        b = list(b)
        if b[0][1] == b[-1][1]:
            yield str(b[0][1])
        else:
            yield '{}-{}'.format(b[0][1], b[-1][1])
        # yield b[0][1], b[-1][1]


def remove_file(file_path):
    """Remove a feature/raster and all of its anciallary files"""
    file_ws = os.path.dirname(file_path)
    for file_name in glob.glob(os.path.splitext(file_path)[0] + ".*"):
        logging.debug('  Remove: {}'.format(os.path.join(file_ws, file_name)))
        os.remove(os.path.join(file_ws, file_name))


def url_download(download_url, output_path, verify=True):
    """Download file from a URL using requests module"""
    import requests
    response = requests.get(download_url, stream=True, verify=verify)
    if response.status_code != 200:
        logging.error('  HTTPError: {}'.format(response.status_code))
        return False

    logging.debug('  Beginning download')
    with (open(output_path, "wb")) as output_f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:  # filter out keep-alive new chunks
                output_f.write(chunk)
    logging.debug('  Download complete')
    return True
