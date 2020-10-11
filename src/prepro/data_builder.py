import os
import gc
import re
import random
import xml.etree.ElementTree as ET
import torch
from copy import deepcopy
from os.path import join as pjoin
from others.logging import logger
from tokenization import BertTokenizer
from others.utils import split_gold

def load_xml_bug(data, annotated=False): #tested
    """
    # Extracting data from XML
    """
    tree = ET.parse(data)
    root = tree.getroot()

    if annotated:
        abs_datasets = []
        ext_datasets = []
        abs_count = 0
        ext_count = 0

        for child in root: # bug report level

            tgt_text1 = []
            tgt_text2 = []
            tgt_text3 = []
            sent_ext1 = []
            sent_ext2 = []
            sent_ext3 = []
            empty_keys = []

            for abs in child.iter('AbstractiveSummary'): # abs summary level
                sentences = abs.iter('Sentence')
                abs_count += 1
                for sent in sentences: # sentence level
                    if len(sent.text.split()) > 0:
                        if abs_count == 1:
                            tgt_text1.append(sent.text)
                        elif abs_count == 2:
                            tgt_text2.append(sent.text)
                        elif abs_count == 3:
                            tgt_text3.append(sent.text)
                    else:
                        pass
            data_dict = {'tgt_text1': tgt_text1, 'tgt_text2': tgt_text2, 'tgt_text3': tgt_text3}
            for key in data_dict.copy():
                if len(data_dict[key]) == 0:
                    del data_dict[key]
                    empty_keys.append(key)
            abs_datasets.append(data_dict)
            abs_count = 0

            for ext in child.iter('ExtractiveSummary'):
                sentences = ext.iter('Sentence')
                ext_count += 1
                for sent in sentences:
                    if ext_count == 1:
                        sent_ext1.append(sent.attrib)
                    elif ext_count == 2:
                        sent_ext2.append(sent.attrib)
                    elif ext_count == 3:
                        sent_ext3.append(sent.attrib)
            data_dict = {'sent_ext1': sent_ext1, 'sent_ext2': sent_ext2, 'sent_ext3': sent_ext3}
            for key in empty_keys:
                if key is 'tgt_text1':
                    del data_dict['sent_ext1']
                elif key is 'tgt_text2':
                    del data_dict['sent_ext2']
                elif key is 'tgt_text3':
                    del data_dict['sent_ext3']
            ext_datasets.append(data_dict)
            ext_count = 0
        return abs_datasets, ext_datasets

    else:
        datasets = []

        for child in root:

            src_text = []
            sent_id = []

            title = child.findtext('Title')
            for sent in child.iter('Sentence'):
                if len(sent.text.split()) > 0:
                    src_text.append(sent.text.strip())
                    sent_id.append(sent.attrib)
                else:
                    pass
            data_dict = {'src_text': src_text, 'sent_id': sent_id, 'title': title}
            datasets.append(data_dict)
        return datasets

def data_dict_combine(dict1, dict2): #tested
    # Combining two data dicts
    for i in range(len(dict1)):
        dict1[i].update(dict2[i])
    return dict1

def get_bug_ids(datasets, id_save_path): #tested
    # Get all the bug ids for all the bug reports in the data
    file = open(id_save_path, 'w+')
    for data_dict in datasets:
        title = data_dict['title']
        match = re.findall(r"[(][0-9]*[)]", title)
        bug_id = int(match[0][1:-1])
        file.write(str(bug_id)+'\n')
    file.close()

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer(args.vocab_file, do_lower_case=False)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '上'
        self.tgt_eos = '下'
        self.tgt_sent_split = '中'
        self.sep_vid = self.tokenizer.vocab[self.sep_token] # token ids
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, data_dict, tgt_keys, ext_keys, is_test=False):
        """

        Args:
            data_dict: Data of individual bug reports
            tgt_keys: a list of keys for accessing the target summaries in data_dict
            ext_keys: similar to tgt_keys but for extractive summaries
            use_bert_basic_tokenizer: boolean
            is_test: boolean

        Returns:
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt
        """
        if ((not is_test) and len(data_dict['src_text']) == 0):
            return None

        src = data_dict['src_text']
        sent_id = data_dict['sent_id']

        src = [src[i].split() for i in range(len(src))]
        idxs = [i for i, s in enumerate(src) if (len(s) > 2)]
        sent_id = [sent_id[i] for i in idxs]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        sent_labels = [] # list for extractive summary sentence labels and order
        ext_text_lst = []
        for key in ext_keys:
            sent_labels.append([])
            data_dict[key + '_labels'] = deepcopy(sent_id)
            data_dict[key + '_pos'] = deepcopy(data_dict[key])
            for j in range(len(data_dict[key])):
                data_dict[key][j]['ID'] = data_dict[key][j]['ID'].strip()

            # note that the order of sentence labels are not equal to summary, need to adjust later

            for k in range(len(sent_id)):
                if data_dict[key + '_labels'][k] in data_dict[key]:
                    data_dict[key + '_labels'][k] = 1
                else:
                    data_dict[key + '_labels'][k] = 0
                sent_labels[-1].append(data_dict[key + '_labels'][k])

            for ext_idx, ext_sent in enumerate(data_dict[key]):
                for idx, sent in enumerate(sent_id):
                    if ext_sent == sent:
                        # (sentence id, sentence label, sentence order)
                        data_dict[key + '_labels'][idx] = (data_dict[key + '_labels'][idx], 1, ext_idx)
                        # (sentence id, sentence position in src)
                        data_dict[key + '_pos'][ext_idx] = (data_dict[key + '_pos'][ext_idx], idx)
                    else:
                        data_dict[key + '_labels'][idx] = (data_dict[key + '_labels'][idx], 0, None)
            ext_text_lst.append(data_dict[key + '_pos'])


        src_text = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_text)
        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        for key in ext_keys:
            data_dict[key + '_labels'] = data_dict[key + '_labels'][:len(cls_ids)]

        tgt_subtokens_idxs_lst = []
        tgt_text_lst = []
        for key in tgt_keys:
            tgt = data_dict[key]
            tgt_tokens = [tgt[i].split() for i in range(len(tgt))]
            # add three tokens representing the start of sequence, sequence split and end of sequence tokens.
            tgt_subtokens_str = '上 ' + ' 中 '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in
                 tgt_tokens]) + ' 下'
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

            if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
                return None

            tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

            tgt_text = '<q>'.join([' '.join(tt) for tt in tgt_tokens])
            tgt_subtokens_idxs_lst.append(tgt_subtoken_idxs)
            tgt_text_lst.append(tgt_text)

        return src_subtoken_idxs, sent_labels, tgt_subtokens_idxs_lst, segments_ids, cls_ids, src_text, tgt_text_lst, ext_text_lst


def format_to_bert(args):
    """
    Preprocess data with train-valid-test split
    """
    bug_data = load_xml_bug(args.raw_path)
    get_bug_ids(bug_data, args.id_save_path)
    abs_, ext_ = load_xml_bug(args.raw_path_annotated, annotated=True)
    sum_data = data_dict_combine(abs_, ext_)
    full_data = data_dict_combine(bug_data, sum_data)
    random.Random(args.seed).shuffle(full_data)

    dataset_len = len(full_data)
    train_len = round(dataset_len * args.split_ratio)
    valid_len = round(dataset_len * ((1 - args.split_ratio)/2))

    train_data = full_data[:train_len]
    valid_data = full_data[train_len:train_len + valid_len]
    test_data = full_data[train_len + valid_len:]

    assert len(train_data) + len(valid_data) + len(test_data) == dataset_len

    datasets = [train_data, valid_data, test_data]
    datasets_type = ['train', 'valid', 'test']
    i = 0
    for corpus in datasets:
        a_lst = [corpus, datasets_type[i], args, pjoin(args.save_path, datasets_type[i]+'_bert.pt')]
        _format_to_bert(a_lst)
        i += 1

def _format_to_bert(params):
    """
    Preprocess data into BERT format
    """
    corpus, datasets_type, args, save_file = params
    is_test = datasets_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    logger.info('Processing %s' % datasets_type)
    datasets = []

    tgt_keys_lst = ['tgt_text1', 'tgt_text2', 'tgt_text3']
    ext_keys_lst = ['sent_ext1', 'sent_ext2', 'sent_ext3']

    for report in corpus:
        tgt_keys = []
        ext_keys = []
        for key in report.keys():
            if key in tgt_keys_lst:
                tgt_keys.append(key)
            if key in ext_keys_lst:
                ext_keys.append(key)
        bert_data = bert.preprocess(report, tgt_keys, ext_keys, is_test=is_test)
        src_subtoken_idxs, sent_labels, tgt_subtokens_idxs_lst, segments_ids, cls_ids, src_text, tgt_text_lst, ext_text_lst = bert_data

        bert_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtokens_idxs_lst,
                          "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                          'src_text': src_text, "tgt_text_lst": tgt_text_lst, "ext_text_lst": ext_text_lst}
        datasets.append(bert_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()