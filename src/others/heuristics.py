import os
from nltk.parse import stanford

os.environ['STANFORD_PARSER'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'
os.environ['STANFORD_MODELS'] = '/home/mich_qiu/Standford_parser/stanford-parser-4.0.0/jars'

class Heuristics():
    def __init__(self, args, data_dict):
        self.args = args
        self.data_dict = data_dict

    def evaluate_sent(self, word_file):
        """Get indices for evaluative and duplicate sentences"""
        all_sent_idxs = {}
        with open(word_file, 'r') as f:
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
            for i in range(idx):  # all other sentences
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
        """Use CFG parsing or RegEx to determine if a sentence is a question"""
        question_sents_idxs = []
        parser = stanford.StanfordParser(
            model_path=self.args.treebank_file)
        sentences = parser.raw_parse_sents(text)
        cfg_tree_list = [list(dep_graphs) for dep_graphs in sentences]
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
