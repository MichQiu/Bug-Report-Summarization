import os
import re
import torch
import stanza
import pickle
import time
from copy import deepcopy
from nltk.parse import stanford
from multiprocessing import Pool
from pretrain.data_source import custom_split

os.environ['STANFORD_PARSER'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'
os.environ['STANFORD_MODELS'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'

class Heuristics():
    def __init__(self, args, bug_comments=None, data_dict=None):
        """
        :param args: parsed arguments
        :param bug_comments: individual bug comments from pretrain data (bug_comments[bug_id], type: list)
        :param data_dict: individual bug reports from finetune data (type: dict)
        *Note that heuristics must be applied after the text has been tokenized regularly than joined back as
        strings to be tokenized again using the BERT tokenizer
        """
        self.args = args
        self.bug_comments = bug_comments
        self.data_dict = data_dict
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

    def evaluate_sent(self, word_file): #tested
        """Get indices for evaluative and duplicate sentences"""
        all_sent_idxs = {}
        with open(word_file, 'r') as f: # text file including all the special words and their positions
            for line in f:
                words = line.split()
                if words[-1] == 'first':
                    sent_idxs_f = self._get_special_sents(words[0], first=True, last=False)
                    eval_dup_dict = self._get_eval_dup_dict('first', sent_idxs_f)
                if words[-1] == 'last':
                    sent_idxs_l = self._get_special_sents(words[0], first=False, last=True)
                    eval_dup_dict = self._get_eval_dup_dict('last', sent_idxs_l)
                all_sent_idxs[words[0]] = eval_dup_dict
            return all_sent_idxs

    def _get_special_sents(self, word, first=False, last=False): #tested
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
            elif word in sent:
                    sent_idxs_lst.append(i)
        return sent_idxs_lst

    def _get_eval_dup_dict(self, position, sent_idxs): #tested
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
            for i in range(idx):  # all other sentences up to the duplicate sentence index
                other_tokens = self.data_dict['src_text'][i].split()
                if len(other_tokens) - len(sent_tokens) > 5:
                    pass
                else:
                    if len(list(set(sent_tokens) - set(other_tokens))) == 0:
                        eval_idx.append(i)
                    else:
                        diff = len(list(set(sent_tokens) - set(other_tokens)))
                        if len(sent_tokens) > len(other_tokens):
                            length = len(sent_tokens)
                        else:
                            length = len(other_tokens)
                        matching_rate = 1 - diff / length
                        if matching_rate > 0.8:
                            eval_idx.append(i)
            eval_dup_dict[idx] = eval_idx
        return eval_dup_dict

    def identify_intent_pr(self):
        """Assign intentions to sentences for pretraining data"""
        tagged_idx = []
        description_sent_idxs = self._is_description_pr()
        question_sent_idxs = self._is_question(self.bug_comments)
        code_sent_idxs = self._is_code(self.bug_comments)
        for idx in description_sent_idxs:
            tokens = self.bug_comments[idx].split()
            tagged_tokens = ['DS'] + tokens
            self.bug_comments[idx] = ''.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['QS'] + tokens
                self.bug_comments[idx] = ''.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in code_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['CO'] + tokens
                self.bug_comments[idx] = ''.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in range(len(self.bug_comments)):
            if idx not in tagged_idx:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['ST'] + tokens
                self.bug_comments[idx] = ''.join(tagged_tokens)

    def identify_intent_ft(self):
        """Assign intentions to sentences for finetune data"""
        tagged_idx = []
        eval_dup_sent_idxs = self.evaluate_sent(self.args.eval_words)
        comment_bounds = self._get_comment_bounds()
        description_sent_idxs = self._is_description(comment_bounds)
        question_sent_idxs = self._is_question(self.data_dict['src_text'])
        code_sent_idxs = self._is_code(self.data_dict['src_text'])
        for idx in description_sent_idxs:
            tokens = self.data_dict['src_text'][idx].split()
            tagged_tokens = ['DS'] + tokens
            self.data_dict['src_text'][idx] = ''.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['QS'] + tokens
                self.data_dict['src_text'][idx] = ''.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in code_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['CO'] + tokens
                self.data_dict['src_text'][idx] = ''.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in range(len(self.data_dict['src_text'])):
            if idx not in tagged_idx:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['ST'] + tokens
                self.data_dict['src_text'][idx] = ''.join(tagged_tokens)
        for words in eval_dup_sent_idxs:
            eval_dup_sents = eval_dup_sent_idxs[words]
            for dup in eval_dup_sents:
                tokens = self.data_dict['src_text'][dup].split()
                tagged_tokens = ['DUP'] + tokens
                self.data_dict['src_text'][dup] = ''.join(tagged_tokens)
                for eval in eval_dup_sents[dup]:
                    tokens = self.data_dict['src_text'][eval].split()
                    tagged_tokens = ['EVAL'] + tokens
                    self.data_dict['src_text'][eval] = ''.join(tagged_tokens)

    def _get_comment_bounds(self): #tested
        """Get the sentence index boundaries for each comment"""
        comment_bounds = []
        for i in range(1, len(self.data_dict['sent_id'])):
            id = self.data_dict['sent_id'][i]
            prev_id = self.data_dict['sent_id'][i - 1]
            if int(float(id['ID'])) > int(float(prev_id['ID'])):
                comment_bounds.append(i)
        return comment_bounds

    def _is_description(self, comment_bounds): #tested
        """Check if sentences are bug descriptions"""
        description_sent_idxs = [i for i in range(comment_bounds[0])]
        return description_sent_idxs

    def _is_question(self, text): #tested
        """Use CFG parsing to determine if a sentence is a question"""
        question_sent_idxs = []
        parser = stanford.StanfordParser(
            model_path=self.args.treebank_file)
        sentences = parser.raw_parse_sents(tuple(text))
        cfg_tree_list = [list(dep_graphs) for dep_graphs in sentences]
        for i in range(len(text)):
            finish = False
            node_list = cfg_tree_list[i][0].productions()
            for j in range(len(node_list)):
                if finish:
                    continue
                node_l = node_list[j].lhs().symbol()
                if node_l == 'SBARQ' or node_l == 'SQ':
                    question_sent_idxs.append(i)
                    continue
                node_tup = node_list[j].rhs()
                for k in range(len(node_tup)):
                    if not isinstance(node_tup[k], str):
                        node_r = node_tup[k].symbol()
                        if node_r == 'SBARQ' or node_r == 'SQ':
                            question_sent_idxs.append(i)
                            finish = True
                            continue
            if i not in question_sent_idxs:
                char_list = list(text[i])
                if char_list[-1] == '?':
                    question_sent_idxs.append(i)
        return question_sent_idxs

    def _is_description_pr(self): #tested
        """Check if sentences are descriptions in pretraining data"""
        description_sent_idxs = []
        split_chars = ['.', '?', '!']
        self.bug_comments[0] = self.bug_comments[0].replace('\n', ' ')
        split_text = custom_split(self.bug_comments[0], split_chars)
        if len(split_text) > 1:
            self.bug_comments.pop(0)
            self.bug_comments = split_text + self.bug_comments
            for idx, sent in enumerate(split_text):
                description_sent_idxs.append(idx)
        else:
            description_sent_idxs.append(0)
        return description_sent_idxs

    def _is_code(self, text):
        code_sent_idxs = []
        for i in range(len(text)): # add more regex patterns for different type of codes
            with open(self.args.code_regex, 'rb+') as f:
                regex = pickle.load(f)
            code_tokens = re.findall(r''+regex, text[i])
            all_tokens = text[i].split()
            code_percentage = len(code_tokens) / len(all_tokens)
            if code_percentage > 0.9:
                code_sent_idxs.append(i)
        return code_sent_idxs

def apply_all_data(args):
    with open(args.data_file, 'r') as f:
        for line in f:
            args.bug_comments = line
            apply_heuristics(args)

def apply_heuristics(args):
    pool = Pool(args.n_cpus)
    if args.bug_comments:
        comment_list = list(args.bug_comments.values())
        pool_list = [(args, comment) for comment in comment_list]
    elif args.datasets:
        comment_list = [bug for bug in args.datasets]
        pool_list = [(args, comment) for comment in comment_list]
    heuristics_dataset = []
    for d in pool.imap(_apply_heuristics, pool_list):
        h_bug_comments = d
        heuristics_dataset.append(h_bug_comments)
    pool.close()
    pool.join()
    torch.save(heuristics_dataset, args.save_file)

def _apply_heuristics(args, data):
    data_type = isinstance(data, list)
    if data_type is False:
        data_type = isinstance(data, dict)
        assert data_type is True
        heuristics = Heuristics(args, data_dict=data)
        heuristics.identify_intent_ft(heuristics.data_dict['src_text'])
        return heuristics.data_dict
    else:
        heuristics = Heuristics(args, bug_comments=data)
        heuristics.identify_intent_pr(heuristics.bug_comments)
        return heuristics.bug_comments