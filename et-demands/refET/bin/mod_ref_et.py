"""mod_ref_et.py
Main ref et model code
called by run_ret.py

"""
import argparse
import datetime
import logging
import multiprocessing as mp
import os
import sys
import shutil
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../lib')))
import ret_utils
import ret_config
import met_nodes

def main(ini_path, log_level = logging.WARNING, mnid_to_run = 'ALL',
        debug_flag = False, mp_procs = 1):
    """ Main function for running Reference ET model

    Args:
        ini_path : str
            absolute file path ofproject INI file
        log_level : logging.lvl

        mnid_to_run :
            Met node id to run in lieu of 'ALL'
        debug_flag : boolean
            True : write debug level comments to debug.txt
            False
        mp_procs : int
            number of cores to use for multiprocessing

    Returns
    -------
    None

    Notes
    -----

    """

    clock_start = time.perf_counter()

    # Start console logging immediately

    logger = ret_utils.console_logger(log_level = log_level)
    logging.warning('\nPython REFET')
    if debug_flag and mp_procs > 1:
        logging.warning('  Debug mode, disabling multiprocessing')
        mp_procs = 1
    if mp_procs > 1:
        logging.warning('  Multiprocessing mode, {0} cores'.format(mp_procs))

    # Read INI file

    cfg = ret_config.RefETConfig()
    cfg.read_refet_ini(ini_path, debug_flag)

    # Start file logging once INI file has been read in

    if debug_flag:
        logger = ret_utils.file_logger(logger, log_level=logging.DEBUG,
                                       output_ws=cfg.project_ws)

    # Read Met Nodes Meta Data

    mnd = met_nodes.MetNodesData()
    mnd.set_met_nodes_meta_data(cfg)

    # Read average monthly data
    mnd.read_avg_monthly_data(cfg)

    # Set up average monthly output if flagged

    if cfg.avg_monthly_flag:
        avg_monthly_header = 'Met Node ID,Met Node Name,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec'
        if "xls" in cfg.input_met['avgm_tmax_path'].lower():
            avgTMaxRev_path = cfg.input_met['avgm_tmax_path'].replace(".xlsx", "avgTMaxMonRev.txt.").replace(".xls", "avgTMaxMonRev.txt.")
        else:
            avgTMaxRev_path = cfg.input_met['avgm_tmax_path']
        avgTMaxRev_hand = open(avgTMaxRev_path, 'w')
        avgTMaxRev_hand.write(avg_monthly_header + "\n")
        if "xls" in cfg.input_met['avgm_tmin_path'].lower():
            avgTMinRev_path = cfg.input_met['avgm_tmin_path'].replace(
                ".xlsx", "avgTMinMonRev.txt.").replace(".xls", "avgTMinMonRev.txt.")
        else:
            avgTMinRev_path = cfg.input_met['avgm_tmin_path']
        avgTMinRev_hand = open(avgTMinRev_path, 'w')
        avgTMinRev_hand.write(avg_monthly_header + "\n")

    # Multiprocessing set up
    node_mp_list = []
    node_mp_flag = False
    if mp_procs > 1:
        if not cfg.avg_monthly_flag:
            if mnid_to_run == 'ALL':
                nodes_count = len(mnd.met_nodes_data.keys())
                logging.warning('  nodes count: {}'.format(nodes_count))
                if nodes_count > 1:
                    logging.warning("  Multiprocessing by node")
                    node_mp_flag = True
            else:
                logging.warning("Multiprocessing can only be used for"
                                " multiple nodes.")
                mp_procs = 1
        else:
            logging.warning("Multiprocessing can not be used when posting"
                            " average monthly data.")
            mp_procs = 1

    # loop thru met nodes meta data

    logging.warning("\n")
    met_node_count = 0
    for met_node_id, met_node in sorted(mnd.met_nodes_data.items()):
        if mnid_to_run == 'ALL' or mnid_to_run == met_node_id:
            logging.info('  Processing node id' + met_node_id + ' with name ' + met_node.met_node_name)
        if met_node.TR_b0 is None: met_node.TR_b0 = cfg.input_met['TR_b0']
        if met_node.TR_b1 is None: met_node.TR_b1 = cfg.input_met['TR_b1']
        if met_node.TR_b2 is None: met_node.TR_b2 = cfg.input_met['TR_b2']

        # read input met data

        met_node_count += 1
        if node_mp_flag and met_node_count > 1:
            node_mp_list.append([met_node_count, cfg, met_node, mnd])
        else:
            if not met_node.read_and_fill_met_data(met_node_count, cfg, mnd):
                if cfg.avg_monthly_flag:
                    avgTMaxRev_hand.close()
                    avgTMinRev_hand.close()
                sys.exit()

            # calculate and post refet et and requested met output

            if cfg.refet_out_flag:
                if not met_node.calculate_and_post_ret_data(cfg):
                    if cfg.avg_monthly_flag:
                        avgTMaxRev_hand.close()
                        avgTMinRev_hand.close()
                    sys.exit()

            # post updated average monthly temperatures if requested -

            if cfg.avg_monthly_flag:
                avgTMaxRev_string = met_node_id + cfg.input_met['avgm_tmax_delimitor'] + met_node.met_node_name
                avgTMinRev_string = avgTMaxRev_string
                for month, row in met_node.input_met_df.groupby(['month'])['tmax'].agg([np.mean]).iterrows():
                    avgTMaxRev_string = avgTMaxRev_string + cfg.input_met['avgm_tmax_delimitor'] + str(row['mean'])
                for month, row in met_node.input_met_df.groupby(['month'])['tmin'].agg([np.mean]).iterrows():
                    avgTMinRev_string = avgTMinRev_string + cfg.input_met['avgm_tmax_delimitor'] + str(row['mean'])
                avgTMaxRev_hand.write(avgTMaxRev_string + "\n")
                avgTMinRev_hand.write(avgTMinRev_string + "\n")

            # setup output met data posting

            # if cfg.output_retalt_flag:
            #     if not met_node.setup_refetalt_out_data(met_node_count, cfg, mnd):
            #         if cfg.avg_monthly_flag:
            #             avgTMaxRev_hand.clos
            #             avgTMinRev_hand.close()
            #         sys.exit()
            # del met_node.input_met_df
    if cfg.avg_monthly_flag:
        avgTMaxRev_hand.close()
        avgTMinRev_hand.close()

    # Multiprocess all nodes
    results = []
    if node_mp_list:
        pool = mp.Pool(mp_procs)
        results = pool.imap(node_mp, node_mp_list, chunksize=1)
        pool.close()
        pool.join()
        del pool, results

    logging.warning('\nREFET Run Completed')
    logging.info('\n{} seconds'.format(time.perf_counter()-clock_start))


def node_mp(tup):
    """Pool multiprocessing friendly function
    Parameters
    ---------
    tup :

    Returns
    -------
    :

    Notes
    ------
    mp.Pool needs all inputs are packed into single tuple
    Tuple is unpacked and and single processing version of function is called

    """
    return node_sp(*tup)


def node_sp(met_node_count, cfg, met_node, mnd):
    """Compute output for each node
    Parameters
    ---------
    met_node_count : int
        count of node being processed
    cfg :
        configuration data
    met_node :
        MetNode instance
    mnd :
        MetNodesData instance

    Returns
    -------
    None

    """

    if not met_node.read_and_fill_met_data(met_node_count, cfg, mnd):
        sys.exit()

    # calculate and post refet et and requested met output

    # calculate and post refet et and requested met output
    if cfg.refet_out_flag:
        if not met_node.calculate_and_post_ret_data(cfg):
            sys.exit()

    # setup output met data posting

    if cfg.output_retalt_flag:
        if not met_node.setup_refetalt_out_data(met_node_count, cfg, mnd):
            sys.exit()
    del met_node.input_met_df

def parse_args():
    """initialize parser

    Parameters
    ---------
    None

    Returns
    -------
    args : argparser.parse_args method


    Notes
    -----
    Uses the argparse module

    """

    parser = argparse.ArgumentParser(
        description='Reference ET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, metavar='PATH',
        type=lambda x: is_valid_file(parser, x), help='Input file')
    parser.add_argument(
        '-d', '--debug', action="store_true", default=False,
        help="Save debug level comments to debug.txt")
    parser.add_argument(
        '-m', '--metid', metavar='mnid_to_run', default='ALL',
        help="User specified met node id to run")
    parser.add_argument(
        '-v', '--verbose', action="store_const",
        dest='log_level', const=logging.INFO, default=logging.WARNING,
        help="Print info level comments")
    parser.add_argument(
        '-mp', '--multiprocessing', default = 1, type = int,
        metavar = 'N', nargs = '?', const = mp.cpu_count(),
        help = 'Number of processers to use')
    args = parser.parse_args()

    # Convert INI path to an absolute path if necessary

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    # print "\nargs are\n", args, "\n"
    return args

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

    """

if __name__ == '__main__':
    args = parse_args()

    main(ini_path=args.ini, log_level = args.log_level, mnid_to_run = args.metid,
         debug_flag = args.debug, mp_procs = args.multiprocessing)
