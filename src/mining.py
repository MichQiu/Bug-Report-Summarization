#encoding=utf-8


import argparse
import time

from others.logging import init_logger
from pretrain import data_source


def do_source_data(args):
    print(time.clock())
    data_source.source_data(args)
    print(time.clock())


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
    parser.add_argument("-url", default='https://bugzilla.mozilla.org')
    parser.add_argument("-url_platform", default='https://bugs.kde.org/query.cgi?format=advanced')
    parser.add_argument("-mozilla", default=False)
    parser.add_argument("-mozilla_products", default='../../data/mozilla_products.txt')
    parser.add_argument("finetune_ids", default='../../prepro/bug_ids.pkl')
    parser.add_argument("-save_path", default='../../data/platform/')

    # nargs=? -> allows optional arguments to be provided, if option string is present but not followed by a
    # command-line argument, the value in const will be used. If no argument is provided at all, the value in default
    # will be used
    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('-log_file', default='../../logs/data_source.log')
    parser.add_argument('-n_cpus', default=4, type=int)

    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_source.'+args.mode + '(args)') # parse function and execute
