"""run_cet.py
wrapper for running area et model
called by user from command line

"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import tkinter as tk
import tkinter.filedialog

def main(ini_path, bin_ws = '', verbose_flag = False, etcid_to_run = 'ALL', debug_flag = False, mp_procs = 1):
    """Wrapper for running crop ET model

    Arguments
    ----------
    ini_path : str
        file path of the project INI file
    verbose_flag : boolean
        True : print info level comments
        False : default
    etcid_to_run :
        ET Cell id to run in lieu of 'ALL'
    debug_flag boolean :
        True : write debug level comments to debug.txt
        False : default
    mp_procs : int
        number of cores to use

    Returns
    -------
    None

    Notes
    -----
    -i, --ini, ini_path : INI file path
    -b, --bin, bin_ws : source code directory path
    -v, --verbose, verbose_flag : print info level comments
    -c, --etcid, mnid_to_run : user specified et cell id to run
    -d, --debug, debug_flag : save debug level comments to debug.txt
    -mp, --multiprocessing, mp_procs : number of processers to use


    """

    # Area et demands python function
    if bin_ws is None:
        print('Source code directory path (-b) not provided')
        sys.exit()
    else:
        # script_path = os.path.join(bin_ws, 'mod_area_et.py')
        script_path = os.path.join(bin_ws, 'mod_area_et.py')

    # Check input folder/path
    if not os.path.isfile(ini_path):
        print('Area ET configuration file does not exist\n  %s' % (ini_path))
        sys.exit()
    elif not os.path.isdir(bin_ws):
        print('Code workspace does not exist\n  %s' % (bin_ws))
        sys.exit()
    elif not os.path.isfile(script_path):
        print('Area ET main script does not exist\n  %s' % (script_path))
        sys.exit()

    # Run Area ET Model

    args_list = ['python', script_path, '-i', ini_path]
    args_list.append('-c')
    args_list.append(etcid_to_run)
    if debug_flag:
        args_list.append('-d')
    if verbose_flag:
        args_list.append('-v')
    if mp_procs > 1:
        args_list.extend(['-mp', str(mp_procs)])
    # print "command line is "
    print(args_list)
    subprocess.call(args_list)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Reference ET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type = lambda x: is_valid_file(parser, x), help = 'Input file')
    parser.add_argument(
        '-b', '--bin', metavar='DIR',
        type = lambda x: is_valid_directory(parser, x), help = 'Source code directory path')
    parser.add_argument(
        '-d', '--debug', action = "store_true", default = False,
        help = "Save debug level comments to debug.txt")
    parser.add_argument(
        '-c', '--etcid', metavar='etcid_to_run', default='ALL',
        help = "User specified et cell id to run")
    parser.add_argument(
        '-v', '--verbose', action = "store_true", default = False,
        help = "Print info level comments")
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar = 'N', nargs = '?', const = mp.cpu_count(),
        help = 'Number of processers to use')
    args = parser.parse_args()

    # Convert INI path to an absolute path if necessary

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    # print "\nargs are\n", args, "\n"
    return args

def get_ini_path(workspace):
    """parses user-entered ini_path

    Parameters
    ---------
    workspace :


    Returns
    -------f
    ini_path : str
        absolute file path of INI file

    Notes
    -----
    Uses tkinter and tkinter.filedialog modules
    Updated syntax and packages for Python 3.x

    """
    root = tk.Tk()
    ini_path = tkinter.filedialog.askopenfilename(
        initialdir = workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select INI file')
    root.destroy()
    return ini_path

def is_valid_file(parser, arg):
    """checks if file is valid
    Parameters
    ---------
    parser : argparse.ArgumentParser instance

    arg : str
        absolute file path

    Returns
    -------
    args : argparser.parse_args method


    Notes
    -----
    Uses the argparse module
    Also defined in mod_ref_et.py

    """

    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg

def is_valid_directory(parser, arg):
    """checks if directory is valid

    Parameters
    ---------
    parser : argparse.ArgumentParser instance

    arg : str
        absolute directory path

    Returns
    -------
    args : argparser.parse_args method


    Notes
    -----
    Uses the argparse module
    Also defined in mod_ref_et.py

    """

    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg

if __name__ == '__main__':
    args = parse_args()
    if args.ini:
        ini_path = args.ini
    else:
        # ini_path = get_ini_path(os.getcwd())

        # todo: upgrade this with path call
        ini_path = 'D:/projects/et-demands/milk_case/msm_aet.ini'
        args.bin = 'D:/projects/et-demands/et-demands/areaET/bin'
        args.multiprocessing = 4

    main(ini_path, bin_ws = args.bin, verbose_flag = args.verbose, etcid_to_run = args.etcid,
        debug_flag = args.debug, mp_procs = args.multiprocessing)
