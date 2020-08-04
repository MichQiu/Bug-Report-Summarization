import os
import gc
import re
import random
import xml.etree.ElementTree as ET
import torch
from nltk.parse import stanford
from collections import defaultdict
from multiprocessing import Pool
from copy import deepcopy
from os.path import join as pjoin
from others.logging import logger
from others.tokenization import BertTokenizer

os.environ['STANFORD_PARSER'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'
os.environ['STANFORD_MODELS'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'

'''
data = "/home/mich_qiu/PycharmProjects/MSc_Thesis/data/IBRS-Corpus/bugreports.xml"
annotated_data = "/home/mich_qiu/PycharmProjects/MSc_Thesis/data/IBRS-Corpus/annotation.xml"
'''

def load_xml_bug(data, annotated=False):
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

            titles = []
            src_text = []
            sent_id = []

            titles.append(child.findtext('Title'))
            for sent in child.iter('Sentence'):
                if len(sent.text.split()) > 0:
                    src_text.append(sent.text.strip())
                    sent_id.append(sent.attrib)
                else:
                    pass
            data_dict = {'src_text': src_text, 'sent_id': sent_id, 'titles': titles}
            datasets.append(data_dict)
        return datasets

def data_dict_combine(dict1, dict2):
    for i in range(len(dict1)):
        dict1[i].update(dict2[i])
    return dict1

class Heuristics():
    def __init__(self, args, data_dict):
        self.args = args
        self.data_dict = data_dict

    def evaluate_sent(self, word_file):
        """Get indices for evaluative and duplicate sentences"""
        with open(word_file, 'r') as f:
            for line in f:
                all_sent_idxs = {}
                words = line.split()
                if words[-1] == 'first':
                    sent_idxs_f = self._get_special_sents(words[0], first=True, last=False)
                    eval_dup_dict = self._get_eval_dup_dict('first', sent_idxs_f)
                if words[-1] == 'last':
                    sent_idxs_l = self._get_special_sents(words[0], first=False, last=True)
                    eval_dup_dict = self._get_eval_dup_dict('last', sent_idxs_l)
                all_sent_idxs[words[0]] = eval_dup_dict
            return all_sent_idxs

    def _get_special_sents(self, word, first=False, last=False):
        """Get a list of indices of sentences that includes a specific word"""
        sent_idxs_lst = []
        for i in range(len(self.data_dict['src_text'])):
            sent = self.data_dict['src_text'][i].split()
            if first:
                if sent[0] == word:
                    sent_idxs_lst.append(i)
            elif last:
                if sent[-1] == word:
                    sent_idxs_lst.append(i)
            else:
                if word in sent:
                    sent_idxs_lst.append(i)
        return sent_idxs_lst

    def _get_eval_dup_dict(self, position, sent_idxs):
        """Get a dict containing the indices of evaluative and duplicate sentences"""
        eval_dup_dict = {}
        for idx in sent_idxs:  # index of sentences that contains evaluation of other sentences
            sent = self.data_dict['src_text'][idx]
            sent_tokens = sent.split()
            if position == 'first':
                sent_tokens.pop(0)
            elif position == 'last':
                sent_tokens.pop()
            eval_idx = []
            for i in range(len(self.data_dict['src_text'])):  # all other sentences
                other_tokens = self.data_dict['src_text'][i].split()
                if len(list(set(sent_tokens) - set(other_tokens))) == 0:
                    eval_idx.append(i)
                else:
                    diff = len(list(set(sent_tokens) - set(other_tokens)))
                    if len(sent_tokens) > len(other_tokens):
                        length = len(sent_tokens)
                    else:
                        length = len(other_tokens)
                    matching_rate = 1 - diff / length
                    if matching_rate > 0.9:
                        eval_idx.append(i)
            eval_dup_dict[idx] = eval_idx
        return eval_dup_dict

    def _get_comment_bounds(self):
        """Get the sentence index boundaries for each comment"""
        comment_bounds = []
        for i in range(1, len(self.data_dict['sent_id'])):
            id = self.data_dict['sent_id'][i]
            prev_id = self.data_dict['sent_id'][i - 1]
            if int(float(id['ID'])) > int(float(prev_id['ID'])):
                comment_bounds.append(i)
        return comment_bounds

    def _is_description(self, comment_bounds):
        """Check if sentences are bug descriptions"""
        description_sent_idxs = [i for i in range(comment_bounds[0])]
        return description_sent_idxs

    def _is_question(self, text, parse=False):
        """Use CFG parsing or RegEx to determine if a sentence is a question"""
        if parse:
            parser = stanford.StanfordParser(
                model_path="/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars/englishPCFG.ser.gz")
            sentences = parser.raw_parse_sents(text)
            cfg_tree_list = [list(dep_graphs) for dep_graphs in sentences]
            question_sents_idxs = []
            for i in range(len(text)):
                finish = False
                node_list = cfg_tree_list[i][0].productions()
                for j in range(len(node_list)):
                    if finish:
                        continue
                    node_l = node_list[j].lhs().symbol()
                    if node_l == 'SBARQ' or node_l == 'SQ':
                        question_sents_idxs.append(i)
                        continue
                    node_tup = node_list[j].rhs()
                    for k in range(len(node_tup)):
                        node_r = node_tup[k].symbol()
                        if node_r == 'SBARQ' or node_r == 'SQ':
                            question_sents_idxs.append(i)
                            finish = True
                            continue
            return question_sents_idxs
        else:
            for sent in text:
                question_what = re.search(r"^[Ww]hat.*?$", text)
                question_where = re.search(r"^[Ww]here.*?$", text)
                question_why = re.search(r"^[Ww]hy.*?$", text)
                question_who = re.search(r"^[Ww]ho.*?$", text)
                question_when = re.search(r"^[Ww]hen.*?$", text)
                question_how = re.search(r"^[Hh]ow.*?$", text)


















He = Heuristics(0, full_data[0])

all_sent_idxs = He._get_special_sents('&gt;', first=True, last=False)



a = deepcopy(full_data[0])
a['src_text'][1]

'''
bug_data = load_xml_bug(data)
abs_, ext_ = load_xml_bug(annotated_data, annotated=True)
sum_data = data_dict_combine(abs_, ext_)
full_data = data_dict_combine(bug_data, sum_data)
random.shuffle(full_data)

dataset_len = len(full_data)
train_len = round(dataset_len * 0.7)
valid_len = round(dataset_len * 0.15)
test_len = dataset_len - train_len - valid_len

train_data = full_data[:train_len]
valid_data = full_data[train_len:train_len + valid_len]
test_data = full_data[train_len + valid_len:]
'''

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token] # token ids
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, data_dict, tgt_keys, ext_keys, use_bert_basic_tokenizer=False, is_test=False):
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

        idxs = [i for i, s in enumerate(src) if (len(s) > 2)]
        sent_id = [sent_id[i] for i in idxs]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = [src[i].split() for i in range(len(src))]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        sent_labels = []
        for key in ext_keys:
            data_dict[key + '_labels'] = deepcopy(sent_id)
            for j in range(len(data_dict[key])):
                data_dict[key][j]['ID'] = data_dict[key][j]['ID'].strip()

            # note that the order of sentence labels are not equal to summary, need to adjust later
            for k in range(len(sent_id)):
                if data_dict[key + '_labels'][k] in data_dict[key]:
                    data_dict[key + '_labels'][k] = 1
                else:
                    data_dict[key + '_labels'][k] = 0
            sent_labels.append(data_dict[key + '_labels'])

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
            tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
                [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in
                 tgt_tokens]) + ' [unused1]'
            tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
            tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

            if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
                return None

            tgt_text = '<q>'.join([' '.join(tt) for tt in tgt_tokens])
            tgt_subtokens_idxs_lst.append(tgt_subtoken_idxs)
            tgt_text_lst.append(tgt_text)

        return src_subtoken_idxs, sent_labels, tgt_subtokens_idxs_lst, segments_ids, cls_ids, src_text, tgt_text_lst


def format_to_bert(args):
    bug_data = load_xml_bug(args.raw_path)
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
        bert_data = bert.preprocess(report, tgt_keys, ext_keys, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
        src_subtoken_idxs, sent_labels, tgt_subtokens_idxs_lst, segments_ids, cls_ids, src_text, tgt_text_lst = bert_data

        bert_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtokens_idxs_lst,
                          "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                          'src_text': src_text, "tgt_text_lst": tgt_text_lst}
        datasets.append(bert_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


'''
tgt_len = []
for i in range(len(datasets)):
    for j in range(len(datasets[i]['tgt'])):
        tgt_len.append(len(datasets[i]['tgt'][j]))


for i in range(len(full_data)):
    if len(full_data[i]['tgt_text1']) == 0:
        print(i, 'text1')
    if len(full_data[i]['tgt_text2']) == 0:
        print(i, 'text2')
    if len(full_data[i]['tgt_text3']) == 0:
        print(i, 'text3')

for i in range(len(full_data)):
    print(full_data[i].keys())

# data = torch.load('/home/mich_qiu/PycharmProjects/MSc Thesis/data/bert_data_cnndm_final/cnndm.train.0.bert.pt')

for i in range(len(data[2]['src_sent_labels'])):
    if data[2]['src_sent_labels'][i] == 1:
        print(i)

data[2]['src_txt'][24]
data[2]['tgt_txt']

'''

# data = torch.load('/home/mich_qiu/PycharmProjects/MSc_Thesis/data/IBRS-Corpus/preprocessed/base/train_bert.pt')
