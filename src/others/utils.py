import os
import re
import shutil
import time
import torch

from os import listdir
from os.path import join as pjoin
from copy import deepcopy
from collections import Counter, OrderedDict
from others import pyrouge
from others.tokenization import _is_control, _is_whitespace
from transformers import BertModel, BertConfig

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir): # make directory if tmp_dir is not a directory
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i]) # write candidates summaries to each file
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir) # initiate pyrouge
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate() # convert summaries and evaluate rouge
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir) # remove the temporary directory
    return results_dict


def test_rouge(temp_dir, cand, ref):
    # list of summary sentences
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates)) # number of sentences
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    In other words, repeat tensor in every batch by count number of times, proceeding from the first batch to the last
    E.g. [[[1, 2, 3],                   [[1, 2, 3],
           [2, 3, 4]],     count = 2     [2, 3, 4]],
          [[6, 7, 8],       ----->      [[1, 2, 3],
           [7, 8, 9]]]                   [2, 3, 4]],
                                        [[6, 7, 8],
                                         [7, 8, 9]],
                                        [[6, 7, 8],
                                         [7, 8, 9]
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size) # [count*x.size(0), x.size(1), ..., x.size(-1)]
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

# convert rouge results to string output
def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )

def _clean_text(text): #tested
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def custom_split(input_string, split_chars): #tested
    """
    Split text based on special punctuation rules
    """
    start_idx = 0
    end_idx = 0
    split_string_list = []
    char_list = list(input_string)
    for idx, char in enumerate(char_list):
        if idx == (len(char_list) - 1):
            end_idx = idx + 1
            char_list_selection = char_list[start_idx:end_idx]
            split_string = ''.join(char_list_selection)
            split_string_list.append(split_string)
        elif char in split_chars and (char_list[idx+1] == ' ' or char_list[idx+1] == '('):
            end_idx = idx
            char_list_selection = char_list[start_idx:end_idx+1]
            split_string = ''.join(char_list_selection)
            split_string_list.append(split_string)
            if char_list[idx+1] == ' ':
                start_idx = idx + 2
            elif char_list[idx+1] == '(':
                start_idx = idx + 1
    return split_string_list

def _batch_data(data_dir, save_dir, lower_bound, upper_bound):
    """
    Batch data into a certain size range while maintaining dict order and keys
    """
    batch_data = {}
    cache = {}
    current_pos = len(batch_data)
    data_label = 0
    save_count = 1
    files = [file for file in listdir(data_dir)]
    for idx, file in enumerate(files):
        if idx == len(files) - 1:
            data = torch.load(data_dir + file)
            for data_idx, item in enumerate(list(data.items())):
                batch_data[data_idx + current_pos] = item[-1]
            current_pos += len(data)
            if cache:
                for cache_idx, item in enumerate(list(cache.items())):
                    batch_data[cache_idx + current_pos] = item[-1]
            save_path = pjoin(save_dir, 'pretrain_' + str(data_label) + '_bert.pt')
            torch.save(batch_data, save_path)
        elif lower_bound < getsize(batch_data) < upper_bound:
            save_path = pjoin(save_dir, 'pretrain_' + str(data_label) + '_bert.pt')
            torch.save(batch_data, save_path)
            print('Minibatch {} saved'.format(save_count))
            save_count += 1
            data_label += 1
            batch_data = {}
            current_pos = 0
        elif getsize(batch_data) <= lower_bound:
            if cache:
                cache_copy = deepcopy(cache)
                for i in cache_copy.keys():
                    cache_value_size = getsize(cache_copy[i])
                    if cache_value_size + getsize(batch_data) <= lower_bound:
                        batch_data[current_pos] = cache_copy[i]
                        current_pos += 1
                        del cache[i]
                    elif lower_bound < cache_value_size + getsize(batch_data) < upper_bound:
                        batch_data[current_pos] = cache_copy[i]
                        save_path = pjoin(save_dir, 'pretrain_' + str(data_label) + '_bert.pt')
                        torch.save(batch_data, save_path)
                        save_count += 1
                        data_label += 1
                        batch_data = {}
                        current_pos = 0
                        del cache[i]
            else:
                data = torch.load(data_dir + file)
                for data_idx, item in enumerate(list(data.items())):
                    batch_data[data_idx + current_pos] = item[-1]
                current_pos += len(data)
        else:
            while getsize(batch_data) >= upper_bound:
                key = list(batch_data.keys())
                cache.update({key[-1]: batch_data[key[-1]]})
                del batch_data[key[-1]]

def getsize(data):
    """
    Get size of an object in bytes
    """
    size = 0
    if isinstance(data, dict):
        for i in list(data.values()):
            for j in i:
                sent_size = len(j.encode('utf8'))
                size += sent_size
    elif isinstance(data, list):
        for i in data:
            sent_size = len(i.encode('utf8'))
            size += sent_size
    elif isinstance(data, str):
        size = len(data.encode('utf8'))
    return size

def batch_data(data_dir, save_file):
    """
    Batch data and resets dict keys
    """
    files = [file for file in listdir(data_dir)]
    full_data = {}
    current_pos = len(full_data)
    for file in files:
        data = torch.load(data_dir + file)
        for data_idx, item in enumerate(list(data.items())):
            full_data[data_idx + current_pos] = item[-1]
        current_pos += len(data)
    torch.save(full_data, save_file)

def split_first_comment(text):
    """
    Split the first comment of the bug report
    """
    if len(text[0].split()) == 0:
        text.pop(0)
        return text
    else:
        split_chars = ['.', '?', '!']
        text[0] = text[0].replace('\n', ' ')
        split_text = custom_split(text[0], split_chars)
        if len(split_text) > 1:
            for i in range(len(split_text)):
                split_text[i] = ' '.join(split_text[i].split())
            text.pop(0)
            new_text = split_text + text
            return new_text
        else:
            text[0] = ' '.join(split_text[0].split())
            return text

def merge_dir(data_dir, new_dir):
    directories = listdir(data_dir)
    dest_dir = listdir(new_dir)
    for dir in directories:
        d = pjoin(data_dir, dir)
        files = listdir(d)
        for file in files:
            if file in dest_dir:
                dest = new_dir + '/' + dir + '_' + file
            else:
                dest = pjoin(new_dir, file)
            f = pjoin(d, file)
            shutil.copy(f, dest)

def write_to_text(data_dir, save_file, shard_size=1000000, bugzilla=False):
    """
    Write contents of object to a text file
    """
    files = listdir(data_dir)
    file_num = 0
    with open(save_file + str(file_num) + '.txt', 'a+') as f:
        for file in files:
            data = torch.load(data_dir + file)
            for bug in list(data.keys()):
                if bugzilla:
                    data[bug] = split_first_comment(data[bug])
                for sent in data[bug]:
                    f.write(sent + '\n')
                f.write(' \n')
                if os.stat(save_file + str(file_num) + '.txt').st_size > shard_size:
                    f.close()
                    file_num += 1
                    f = open(save_file + str(file_num) + '.txt', 'a+')

def shard_text(data_dir, save_file, shard_size):
    # Shard a text file into multiple smaller files
    file_num = 0
    r = open(data_dir, 'r')
    with open(save_file + str(file_num) + '.txt', 'a+') as f:
        for line in r:
            if line == ' \n':
                if os.stat(save_file + str(file_num) + '.txt').st_size > shard_size:
                    f.close()
                    file_num += 1
                    f = open(save_file + str(file_num) + '.txt', 'a+')
                f.write(line)
            else:
                f.write(line)

def write_to_full(data_dir, save_file):
    files = listdir(data_dir)
    text_list = []
    full_dict = {}
    for file in files:
        bug_dict = torch.load(data_dir + file)
        for text in bug_dict.values():
            text_list.append(text)
    for i, j in enumerate(text_list):
        full_dict[i] = j
    torch.save(full_dict, save_file)

def shard_by_report(data, save_dir):
    full_dict = torch.load(data)
    for bug_num in full_dict.keys():
        with open(save_dir + '/' + 'bug_' + str(bug_num) + '.txt', 'w+') as f:
            bug = full_dict[bug_num]
            f.write('Bug Report: ' + bug['summary'] + '\n')
            f.write('\n')
            comment_nums = list(bug.keys())
            for i in range(1, len(comment_nums)):
                creator = bug[i]['creator']
                creation_time = bug[i]['creation_time']
                f.write('Author: ' + creator + '\n')
                f.write('Date posted: ' + creation_time + '\n')
                for idx, sent in enumerate(bug[i]['text']):
                    f.write(str(i) + '.' + str(idx+1) + ' ' + sent + '\n')
                f.write('\n')
            f.close()

def convert_file_format(data_dir):
    files = [file for file in listdir(data_dir)]
    for file in files:
        os.rename(data_dir + file, data_dir + file + '.txt')

def get_distribution(data): #Not Used
    distribution_data = []
    for bug in data:
        bug_intent_stat = []
        for ext_sum in bug['ext_text_lst']:
            intent_lst = [bug['src_text'][sent[1]] for sent in ext_sum]
            intent_counter = Counter(intent_lst)
            bug_intent_stat.append((intent_counter, len(list(intent_counter.elements()))))
        distribution_data.append(bug_intent_stat)

def convert_state_dict(config_file, state_dict, save_file):
    # Convert NVIDIA AMP state dicts into regular state dicts
    config = BertConfig.from_json_file(config_file)
    model = BertModel(config)
    amp_checkpoint = torch.load(state_dict)
    amp_state_dict = amp_checkpoint["model"]
    dict_length = len(model.state_dict().keys())
    amp_state_dict_keys = list(amp_state_dict.keys())

    for key in amp_state_dict_keys:
        if 'intent' in key:
            amp_state_dict_keys.remove(key)
    amp_state_dict_keys = amp_state_dict_keys[:dict_length]

    converted_state_dict_keys = deepcopy(amp_state_dict_keys)

    for i in range(len(converted_state_dict_keys)):
        converted_state_dict_keys[i] = converted_state_dict_keys[i][5:]
        if 'bias' in converted_state_dict_keys[i]:
            if '_act' in converted_state_dict_keys[i]:
                converted_state_dict_keys[i] = converted_state_dict_keys[i][:-9] + converted_state_dict_keys[i][-5:]
        elif 'weight' in converted_state_dict_keys[i]:
            if '_act' in converted_state_dict_keys[i]:
                converted_state_dict_keys[i] = converted_state_dict_keys[i][:-11] + converted_state_dict_keys[i][-7:]
        else:
            continue

    state_dict = OrderedDict()

    for idx, key in enumerate(converted_state_dict_keys):
        state_dict[key] = amp_state_dict[amp_state_dict_keys[idx]]

    torch.save(state_dict, save_file)

def split_gold(data_file, save_file, gold_num):
    # Split the finetune datasets according to the number of gold summaries they have
    data = torch.load(data_file)
    for num in range(len(gold_num)):
        for i in range(len(data)):
            data[i]['tgt'] = data[i]['tgt'][gold_num]
            data[i]['src_sent_labels'] = data[i]['src_sent_labels'][gold_num]
            data[i]['ext_text_lst'] = data[i]['ext_text_lst'][gold_num]
            data[i]['tgt_text_lst'] = data[i]['tgt_text_lst'][gold_num]
        torch.save(data, save_file + 'bug.valid.' + str(num) + '.bert.pt')
