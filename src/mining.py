#encoding=utf-8


import argparse
import time
from pretrain import bugsource

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # argument parser for preprocessing
    parser = argparse.ArgumentParser()

    # the type of preprocessing to use
    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-url")
    parser.add_argument("-url_platform")
    parser.add_argument("-mozilla", default=False)
    parser.add_argument("-mozilla_products")
    parser.add_argument("-finetune_ids_file")
    parser.add_argument("-save_path", default='../../data/platform/')
    parser.add_argument("-args_save_path", default='../../pretrain/')

    # nargs=? -> allows optional arguments to be provided, if option string is present but not followed by a
    # command-line argument, the value in const will be used. If no argument is provided at all, the value in default
    # will be used
    parser.add_argument('-n_cpus', default=8, type=int)

    args = parser.parse_args()
    eval('bugsource.'+ args.mode + '(args)') # parse function and execute
