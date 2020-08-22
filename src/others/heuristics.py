import os
import re
import stanza
from copy import deepcopy
from nltk.parse import stanford

os.environ['STANFORD_PARSER'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'
os.environ['STANFORD_MODELS'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'

class Heuristics():
    def __init__(self, args, bug_comments=None, data_dict=None):
        """
        :param args: parsed arguments
        :param bug_comments: individual bug comments from pretrain data (bug_comments[bug_id], type: list)
        :param data_dict: individual bug reports from finetune data (type: dict)
        *Note that heuristics must be applied after the text has been tokenized regularly than joined back as
        strings to be tokenized again using BERT tokenizer
        """
        self.args = args
        self.bug_comments = bug_comments
        self.data_dict = data_dict
        self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

    def evaluate_sent(self, word_file):
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

    def identify_intent_pr(self, text):
        """Assign intentions to sentences for pretraining data"""
        tagged_idx = []
        description_sent_idxs = self._is_description_pr(text)
        question_sent_idxs = self._is_question(text)
        statement_sent_idxs = self._is_statement(text)
        for idx in description_sent_idxs:
            text[idx] = ['DS'] + text[idx]
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                text[idx] = ['QS'] + text[idx]
                tagged_idx.append(idx)
        for idx in statement_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                text[idx] = ['ST'] + text[idx]
        for idx in range(len(text)):
            if idx not in tagged_idx:
                text[idx] = ["NO"] + text[idx]

    def identify_intent_ft(self, text):
        """Assign intentions to sentences for finetune data"""
        tagged_idx = []
        comment_bounds = self._get_comment_bounds()
        description_sent_idxs = self._is_description(comment_bounds)
        question_sent_idxs = self._is_question(text)
        statement_sent_idxs = self._is_statement(text)
        for idx in description_sent_idxs:
            text[idx] = ['DS'] + text[idx]
            tagged_idx.append(idx)
        for idx in question_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                text[idx] = ['QS'] + text[idx]
                tagged_idx.append(idx)
        for idx in statement_sent_idxs:
            if idx in tagged_idx:
                pass
            else:
                text[idx] = ['ST'] + text[idx]
        for idx in range(len(text)):
            if idx not in tagged_idx:
                text[idx] = ["NO"] + text[idx]

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

    def _is_question(self, text):
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
                    text[i] = ['[QT]'] + text[i]
                    continue
                node_tup = node_list[j].rhs()
                for k in range(len(node_tup)):
                    node_r = node_tup[k].symbol()
                    if node_r == 'SBARQ' or node_r == 'SQ':
                        question_sent_idxs.append(i)
                        text[i] = ['[QT]'] + text[i]
                        finish = True
                        continue
        return question_sent_idxs

    def _is_description_pr(self, text):
        """Check if sentences are descriptions in pretraining data"""
        description_sent_idxs = []
        description_sent = []
        text[0] = text[0].replace('\n', ' ')
        for k in range(len(text[0])):
            if text[0][k] == '.' and k != (len(text[0]) - 1):
                if text[0][k + 1] is not ' ':
                    description_sent_idxs.append(0)
                    break
            elif k == (len(text) - 1):
                split_text = text.split('.')
                if split_text[-1] == '':
                    split_text.pop(-1)
                for sent in split_text:
                    if '?' or '!' in sent:
                        pass
                    else:
                        sent = sent + '.'
                    description_sent.append(sent)
        if len(description_sent) > 1:
            text.pop(0)
            text = description_sent + text
            for idx, sent in enumerate(text):
                description_sent_idxs.append(idx)
        return description_sent_idxs

    def _is_statement(self, text):
        """Check if sentences are statements"""
        statement_sent_idxs = []
        with open(self.args.heuristics, 'r') as f:
            for line in f:
                sent = line.split()
                for i in range(len(text)):
                    check = self.compare(sent, text[i])
                    if check:
                        statement_sent_idxs.append(i)
        return statement_sent_idxs

    def compare(self, heur_text, text_sent):
        """Comparing sentences between heuristics and target and counting the number of matches"""
        tags = {"[something]": ('NOUN', 'PRON', 'PROPN'), "[someone]": ('NOUN', 'PRON', 'PROPN'),
                "[verb]": ('VERB'), "[link]": ('SYM'), "[date]": ('NUM')}
        no_of_match = 0
        for i in range(len(heur_text)):
            if no_of_match != i:
                break
            _word = heur_text[i]
            if _word in tags:
                match = False
                for tag in tags.keys():
                    if match is True:
                        continue
                    elif _word == tag:
                        _pos_tags = tags[tag]
                        self._compare(text_sent, no_of_match, match, _pos_tags, 'tag')
            else:
                self._compare(text_sent, no_of_match, _word)
        if no_of_match == len(heur_text):
            return True
        else:
            return False

    def _compare(self, text_sent, no_of_match, match=False, _word=None, _pos_tags=None, type='word'):
        """Comparing a word from a heuristics sentence to words in a target sentence"""
        for j in range(len(text_sent)):
            if type == 'word':
                word = text_sent[j]
                if word == _word or word == _word.lower():
                    no_of_match += 1
                    break
                else:
                    pass
            elif type == 'tag':
                pos_tag = self._POS(text_sent[j])
                if pos_tag in _pos_tags:
                    no_of_match += 1
                    match = True
                    break
                else:
                    pass

    def _POS(self, word):
        """Obtain the POS tag of a word"""
        word_POS = self.pipeline(word)
        word_POS_dict = word_POS.to_dict()
        pos_tag = word_POS_dict[0][0]['upos']
        return pos_tag


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