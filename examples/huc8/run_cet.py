#!/usr/bin/env python

import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import sys

def main(ini_path, bin_ws = '', verbose_flag = False,
        etcid_to_run = 'ALL', cal_flag = False,
        debug_flag = False, mp_procs = 1):
    """Wrapper for running crop et model

    Arguments
    ---------
    ini_path : str
        file path of the project INI file
    bin_ws : str
        path of source code directory
    verbose_flag : boolean
        True : print info level comments
    etcid_to_run :
        ET Cell id to run in lieu of 'ALL'
    cal_flag : boolean
        True :
        False :
    debug_flag : boolean
        True : write debug level comments to debug.txt
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
    -m, --metid, mnid_to_run : user specified met id to run
    -d, --debug, debug_flag : save debug level comments to debug.txt
    -mp, --multiprocessing, mp_procs : number of processers to use
    --cal, cal_flag : display mean annual start/end dates to screen
    """
    # print ini_path

    bin_ws = r'..\..\et-demands\cropET\bin'

    # Crop ET demands python function

    script_path = os.path.join(bin_ws, 'mod_crop_et.py')
    print('bin_ws - ' + bin_ws)
    print('ini_path - ' + ini_path)
    # Check input folder/path

    if not os.path.isfile(ini_path):
        print('Crop ET demands input file does not exist\n  %s' % (ini_path))
        sys.exit()
    elif not os.path.isdir(bin_ws):
        print('Code workspace does not exist\n  %s' % (bin_ws))
        sys.exit()
    elif not os.path.isfile(script_path):
        print('Crop ET demands main script does not exist\n  %s' % (script_path))
        sys.exit()

    # Run Crop ET Demands Model

    args_list = ['python', script_path, '-i', ini_path]
    args_list.append('-c')
    args_list.append(etcid_to_run)
    if debug_flag:
        args_list.append('-d')
    if verbose_flag:
        args_list.append('-v')
    if mp_procs > 1:
        args_list.extend(['-mp', str(mp_procs)])
    subprocess.call(args_list)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop ET-Demands',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', metavar='PATH',
        type = lambda x: is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-b', '--bin', metavar='DIR',
        type = lambda x: is_valid_directory(parser, x), help = 'Source code directory path')
    parser.add_argument(
        '-d', '--debug', action="store_true", default=False,
        help = "Save debug level comments to debug.txt")
    parser.add_argument(
        '-c', '--etcid', metavar='etcid_to_run', default='ALL',
        help = "User specified et cell id to run")
    parser.add_argument(
        '-v', '--verbose', action="store_true", default=False,
        help = "Print info level comments")
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processers to use')
    parser.add_argument(
        '--cal', action = 'store_true', default = False,
        help = "Display mean annual start/end dates to screen")
    args = parser.parse_args()

    # Convert INI path to an absolute path if necessary
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)

    # Convert source code dir to an absolute path if necessary
    if args.bin and os.path.isfile(os.path.abspath(args.bin)):
        args.bin = os.path.abspath(args.bin)

    return args

def get_ini_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    root.destroy()
    return ini_path

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg
def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg

if __name__ == '__main__':
    args = parse_args()
    if args.ini:
        ini_path = args.ini
    else:
        ini_path = get_ini_path(os.getcwd())
    main(ini_path, bin_ws = args.bin, verbose_flag=args.verbose,
        etcid_to_run = args.etcid, cal_flag = args.cal,
        debug_flag = args.debug, mp_procs=args.multiprocessing)
