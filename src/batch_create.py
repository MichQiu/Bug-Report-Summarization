# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import multiprocessing
import os
import pprint
import subprocess
from os import listdir


def main(args):
    #working_dir = os.environ['BERT_PREP_WORKING_DIR']
    working_dir = args.working_dir

    print('Working Directory:', working_dir)
    print('Action:', args.action)
    print('Dataset Name:', args.dataset)

    if args.input_files:
        args.input_files = args.input_files.split(',')

    directory_structure = {
        'hdf5': working_dir + '/hdf5',
        'phase1': working_dir + '/phase1'
    }

    print('\nDirectory Structure:')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(directory_structure)
    print('')

    if args.action == 'create_hdf5_files':
        last_process = None

        if not os.path.exists(directory_structure['hdf5'] + "/" + args.dataset):
            os.makedirs(directory_structure['hdf5'] + "/" + args.dataset)

        def create_record_worker(filename_prefix, shard_id, input_file, output_format='hdf5'):
            bert_preprocessing_command = 'python' + ' ' + working_dir + '/Bug-Report-Summarization/src' + '/create_pretraining_data.py'
            bert_preprocessing_command += ' --input_file=' + input_file
            bert_preprocessing_command += ' --output_file=' + directory_structure['hdf5'] + '/' + args.dataset + '/' + filename_prefix + '_' + str(shard_id) + '.' + output_format
            bert_preprocessing_command += ' --vocab_file=' + args.vocab_file
            bert_preprocessing_command += ' --do_lower_case' if args.do_lower_case else ''
            bert_preprocessing_command += ' --max_seq_length=' + str(args.max_seq_length)
            bert_preprocessing_command += ' --max_predictions_per_seq=' + str(args.max_predictions_per_seq)
            bert_preprocessing_command += ' --masked_lm_prob=' + str(args.masked_lm_prob)
            bert_preprocessing_command += ' --random_seed=' + str(args.random_seed)
            bert_preprocessing_command += ' --dupe_factor=' + str(args.dupe_factor)
            bert_preprocessing_process = subprocess.Popen(bert_preprocessing_command, shell=True)

            last_process = bert_preprocessing_process

            # This could be better optimized (fine if all take equal time)
            if shard_id % args.n_processes == 0 and shard_id > 0:
                bert_preprocessing_process.wait()
            return last_process

        output_file_prefix = args.dataset

        files = [directory_structure['phase1'] + '/' + file for file in listdir(directory_structure['phase1'])]
        for i in range(args.n_training_shards):
            last_process = create_record_worker(output_file_prefix + '_phase1_', i, files[i])

        last_process.wait()
        '''
        if args.n_test_shards:
            for i in range(args.n_test_shards):
                last_process = create_record_worker(output_file_prefix + '_phase2_', i)

            last_process.wait()
        '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything BERT-related'
    )

    parser.add_argument(
        '--working_dir',
        type=str,
        help='Specify the current working directory',
    )

    parser.add_argument(
        '--action',
        type=str,
        help='Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords',
        choices={
            'create_hdf5_files' # Turn each shard into a HDF5 file with masking and next sentence prediction info
        }
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
        choices={
            'regular',
            'heuristics'
        }
    )

    parser.add_argument(
        '--input_files',
        type=str,
        help='Specify the input files in a comma-separated list (no spaces)',
        required=False
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=256
    )

    parser.add_argument(
        '--n_test_shards',
        type=int,
        help='Specify the number of test shards to generate',
        default=256,
        required=False
    )

    parser.add_argument(
        '--fraction_test_set',
        type=float,
        help='Specify the fraction (0..1) of the data to withhold for the test data split (based on number of sequences)',
        default=0.1,
        required=False
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=10
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Specify the base seed to use for any random number generation',
        default=12345
    )

    parser.add_argument(
        '--dupe_factor',
        type=int,
        help='Specify the duplication factor',
        default=10
    )

    parser.add_argument(
        '--masked_lm_prob',
        type=float,
        help='Specify the probability for masked lm',
        default=0.15
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=128
    )

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        help='Specify the maximum number of masked words per sequence',
        default=20
    )

    parser.add_argument(
        '--do_lower_case',
        type=int,
        help='Specify whether it is cased (0) or uncased (1) (any number greater than 0 will be treated as uncased)',
        default=0
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    parser.add_argument(
        '--interactive_json_config_generator',
        type=str,
        help='Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords'
    )

    args = parser.parse_args()
    main(args)