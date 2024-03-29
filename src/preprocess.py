#encoding=utf-8


import argparse
import time

from others.logging import init_logger
from prepro import data_builder


def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
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
    parser.add_argument("-mode", default='', type=str) # format_to_bert, format_to_lines, tokenize
    parser.add_argument("-select_mode", default='greedy', type=str)
    parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-raw_path_annotated", default='../../line_data_annotated')
    parser.add_argument("-save_path", default='../../data/')
    parser.add_argument("-id_save_path", )
    parser.add_argument("-vocab_file", default='')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    # nargs=? -> allows optional arguments to be provided, if option string is present but not followed by a
    # command-line argument, the value in const will be used. If no argument is provided at all, the value in default
    # will be used
    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-log_file', default='../../logs/bug.log')

    parser.add_argument('-split_ratio', default=0.7, type=int)
    parser.add_argument('-seed', default=516, type=int)

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.'+args.mode + '(args)') # parse function and execute
