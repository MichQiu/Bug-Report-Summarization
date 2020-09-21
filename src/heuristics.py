import os
import re
import pickle
import torch
import spacy
import argparse
import warnings
from pathlib import Path
from spacy.matcher import PhraseMatcher
from multiprocessing import Pool
from others.utils import custom_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load("en_core_web_lg")

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
        solution_sent_idxs = self._is_statement(self.bug_comments, self.args.h_solution, "solution")
        info_sent_idxs = self._is_statement(self.bug_comments, self.args.h_info, "info")
        for idx in description_sent_idxs:
            tokens = self.bug_comments[idx].split()
            tagged_tokens = ['[DES]'] + tokens
            self.bug_comments[idx] = ' '.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['[QS]'] + tokens
                self.bug_comments[idx] = ' '.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in code_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['[CODE]'] + tokens
                self.bug_comments[idx] = ' '.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in solution_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['[SOLU]'] + tokens
                self.bug_comments[idx] = ' '.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in info_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['[INFO]'] + tokens
                self.bug_comments[idx] = ' '.join(tagged_tokens)
                tagged_idx.append(idx)
        for idx in range(len(self.bug_comments)):
            if idx not in tagged_idx:
                tokens = self.bug_comments[idx].split()
                tagged_tokens = ['[NON]'] + tokens
                self.bug_comments[idx] = ' '.join(tagged_tokens)

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
            tagged_tokens = ['[DES]'] + tokens
            self.data_dict['src_text'][idx] = ' '.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['[QS]'] + tokens
                self.data_dict['src_text'][idx] = ' '.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in code_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['[CODE]'] + tokens
                self.data_dict['src_text'][idx] = ' '.join(tagged_tokens)
            tagged_idx.append(idx)
        for idx in range(len(self.data_dict['src_text'])):
            if idx not in tagged_idx:
                tokens = self.data_dict['src_text'][idx].split()
                tagged_tokens = ['[ST]'] + tokens
                self.data_dict['src_text'][idx] = ' '.join(tagged_tokens)
        for words in eval_dup_sent_idxs:
            eval_dup_sents = eval_dup_sent_idxs[words]
            for dup in eval_dup_sents:
                tokens = self.data_dict['src_text'][dup].split()
                tagged_tokens = ['[DUP]'] + tokens
                self.data_dict['src_text'][dup] = ''.join(tagged_tokens)
                for eval in eval_dup_sents[dup]:
                    tokens = self.data_dict['src_text'][eval].split()
                    tagged_tokens = ['[EVAL]'] + tokens
                    self.data_dict['src_text'][eval] = ' '.join(tagged_tokens)

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
        for idx, sent in enumerate(text):
            try:
                if sent[-1] == "?":
                    question_sent_idxs.append(idx)
            except:
                continue
        return question_sent_idxs

    def _is_description_pr(self): #tested
        """Check if sentences are descriptions in pretraining data"""
        description_sent_idxs = []
        try:
            split_chars = ['.', '?', '!']
            if self.bug_comments[0] != '':
                self.bug_comments[0] = self.bug_comments[0].replace('\n', ' ')
                split_text = custom_split(self.bug_comments[0], split_chars)
                if len(split_text) > 1:
                    self.bug_comments.pop(0)
                    self.bug_comments = split_text + self.bug_comments
                    for idx, sent in enumerate(split_text):
                        try:
                            description_sent_idxs.append(idx)
                        except:
                            continue
                else:
                    description_sent_idxs.append(0)
            else:
                self.bug_comments.pop(0)
        except:
            pass
        return description_sent_idxs

    def _is_code(self, text): #tested
        """Check if sentences mainly consist of code and non-natural language texts"""
        code_sent_idxs = []
        try:
            for i in range(len(text)):  # add more regex patterns for different type of codes
                try:
                    code_tokens = re.findall(
                        r'([<a-zA-z0-9>(){}.,\-_/\\:;!@#$%^&*?\[\]|`~]+\.[<a-zA-z0-9>(){}.,\-_/\\:;!@#$%^&*?\[\]|`~]+)|([!@#$%^&*()\-_+=\[\]{}:;<>|\~`/\\]+[a-zA-z0-9]+)+|([!@#$%^&*()\-_+=\[\]{}:;<>|\~`/\\]+)+|([a-zA-Z0-9]+[!@#$%^&*()_+=\[\]{}:;<>|\~`/\\0-9]+[a-zA-Z0-9]+)+|([a-zA-Z0-9]+[!@#$%^&*()\-_+=\[\]{}:;<>|\~`/\\0-9]+)+|(\[.+\])|(\{.+\})|(\(.+\))',
                        text[i])
                    cleaned_token_list = []
                    for token in code_tokens:
                        cleaned_code_token = ''.join(token)
                        cleaned_token_list.append(cleaned_code_token)
                    joined_tokens = ' '.join(cleaned_token_list)
                    if len(joined_tokens) > len(text[i]):
                        continue
                    else:
                        code_percentage = len(joined_tokens) / len(text[i])
                        if code_percentage > 0.5:
                            code_sent_idxs.append(i)
                except:
                    continue
        except:
            pass
        return code_sent_idxs

    def _is_statement(self, text, heuristics_file, sent_type):
        statement_sent_idxs = []
        try:
            matcher = PhraseMatcher(nlp.vocab, attr="POS")
            with open(heuristics_file, 'r') as f:
                heur_list = [nlp(' '.join(line.split())) for line in f]
                matcher.add(sent_type, heur_list)
                docs = nlp.pipe(text, disable=["parser", "ner", "textcat"])
                matches = matcher.pipe(docs, return_matches=True)
                match_list = [match for match in matches]
                for idx, match in enumerate(match_list):
                    try:
                        if match[1]:
                            statement_sent_idxs.append(idx)
                    except:
                        continue
        except:
            pass
        return statement_sent_idxs

def apply_full(args):
    path = Path(args.data_dir)
    files = list(path.glob('**/*.pt'))
    for file in tqdm(files):
        apply_heuristics(file, args)

def apply_heuristics(src_data, args):
    """
    Make sure that args.datasets and args.bug_comments are set from their respective directories
    """
    pool = Pool(args.n_cpus)
    if args.finetune:
        data = torch.load(src_data)
        comment_list = [(args, bug) for bug in data['src_text']]
        if len(comment_list) < args.n_cpus:
            chunksize = 1
        else:
            chunksize = round((1 + len(comment_list))/args.n_cpus)
        for _ in pool.imap(_apply_heuristics_ft, comment_list, chunksize):
            pass
        pool.close()
        pool.join()
    else:
        data = torch.load(src_data)
        comment_list = [(args, bug) for bug in data.values()]
        if len(comment_list) < args.n_cpus:
            chunksize = 1
        else:
            chunksize = round((1 + len(comment_list))/args.n_cpus)
        for _ in pool.imap(_apply_heuristics_pr, comment_list, chunksize):
            pass
        pool.close()
        pool.join()

def _apply_heuristics_ft(params):
    args, data = params
    data = [sent for sent in data if len(sent.split()) > 2]
    heuristics = Heuristics(args, data_dict=data)
    heuristics.identify_intent_ft()
    with open(args.save_file, 'a+') as f:
        for sent in heuristics.data_dict:
            f.write(sent+'\n')
        f.write(' \n')
    return heuristics.data_dict

def _apply_heuristics_pr(params):
    args, data = params
    data = [sent for sent in data if len(sent.split()) > 2]
    heuristics = Heuristics(args, bug_comments=data)
    heuristics.identify_intent_pr()
    with open(args.save_file, 'a+') as f:
        for sent in heuristics.bug_comments:
            f.write(sent+'\n')
        f.write(' \n')
    return heuristics.bug_comments

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default='', type=str)
    arg_parser.add_argument("--finetune", default=False, type=bool)
    arg_parser.add_argument("--save_file", default='', type=str)
    arg_parser.add_argument("--n_cpus", default=10, type=int)
    arg_parser.add_argument("--h_solution", default='', type=str)
    arg_parser.add_argument("--h_info", default='', type=str)

    args = arg_parser.parse_args()
    apply_full(args)