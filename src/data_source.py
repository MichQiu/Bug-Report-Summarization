from __future__ import print_function

import re
from os.path import join as pjoin
from multiprocessing import Pool
from html.parser import HTMLParser
from others.logging import logger
import urllib.request as urllib2
import torch
import requests
import argparse
from tqdm import tqdm
from others.utils import _clean_text, custom_split

class MyHTMLParser(HTMLParser):
    ls_data = []
    append_data = False

    def handle_starttag(self, startTag, attrs):
        if startTag == 'select':
            for attr in attrs:
                if attr[-1] == 'product':
                    self.append_data = True
                    break

    def handle_endtag(self, endTag):
        if endTag == 'select' and self.append_data:
            self.append_data = False

    def handle_data(self, data):
        if self.append_data:
            non_space = re.search(r'[\n]', data)
            if non_space is None:
                self.ls_data.append(data)

def get_product_lists(url): #tested
    parser = MyHTMLParser()
    parser.ls_data = []
    html_page = urllib2.urlopen(url)
    logger.info("Parsing data from %s" % url)
    parser.feed(str(html_page.read()))
    logger.info("Parsing complete")
    parser.ls_data = parser.ls_data[1::2]
    return parser.ls_data

class DataExtract():

    def __init__(self, args, product_list):
        self.api = args.api
        self.n_cpu = args.n_cpu
        self.product_list = product_list
        self.save_path = args.save_path
        try:
            self.finetune_ids_file = open(args.finetune_ids_file, 'r')
            self.finetune_ids = {}
            self.no_finetune_data = False
            for line in self.finetune_ids_file:
                bug_id = line[:-1]
                self.finetune_ids[bug_id] = False
            self.finetune_ids_file.close()
        except:
            self.no_finetune_data = True

    def source_data(self, include_extra_info=False): # tested
        bugreport_len = self.calculate_bugs_len()
        char_list = ["/", "."]
        for product in tqdm(list(bugreport_len.keys())):
            product_name, comments = self._source(product, bugreport_len[product], include_extra_info=include_extra_info)
            if len(comments) > 0:
                product_name = ['_' if char in char_list else char for char in product_name]
                product_name = ''.join(product_name)
                save_file = pjoin(self.save_path, product_name + '_bert.pt')
                torch.save(comments, save_file)
            del product_name, comments

    def _source(self, product, bug_ids, include_extra_info=False):  # tested
        """Get the bug ids from the product and extract their comments"""
        bug_comments = {}
        if len(bug_ids) < self.n_cpu:
            pool = Pool(len(bug_ids))
            divisor = len(bug_ids)
        else:
            pool = Pool(self.n_cpu)
            divisor = self.n_cpu
        if include_extra_info:
            source_method = self._get_bug_info
        else:
            source_method = self._get_text
        for d in pool.imap(source_method, bug_ids, round((1 + len(bug_ids)) / divisor)):
            bug_id, bug_info = d
            if bug_id is False and bug_info is False:
                continue
            elif len(bug_info) >= 8:
                bug_comments[bug_id] = bug_info
        pool.close()
        pool.join()
        return product, bug_comments

    def _get_text(self, bug_id): # tested
        """Get the comments of an individual bug via its bug id"""
        try:
            session = requests.Session()
            response = session.get(self.api + '/' + bug_id + '/comment')
            bug = response.json()
            comments = bug['bugs'][bug_id]['comments']
            sent_list = []
            if len(comments) > 0:
                for comment in comments:
                    sent_list.append(comment['text'])
                src_text = []
                split_chars = ['.', '?', '!']
                src_text.append(sent_list[0])
                for i in range(1, len(sent_list)):
                    text = sent_list[i]
                    text = _clean_text(text)
                    clear_whitespace = text.split()
                    text = ' '.join(clear_whitespace)
                    split_text = custom_split(text, split_chars)
                    for sent in split_text:
                        src_text.append(sent)
                return bug_id, src_text
            else:
                return False, False
        except:
            return False, False

    def _get_bug_info(self, bug_id):
        try:
            bug_info = {}
            session = requests.Session()
            response = session.get(self.api + '/' + bug_id)
            _bug = response.json()
            bug_summary = _bug['bugs'][0]['summary']
            bug_info['summary'] = bug_summary
            response = session.get(self.api + '/' + bug_id + '/comment')
            bug = response.json()
            author_comments = bug['bugs'][bug_id]['comments']
            if len(author_comments) > 0:
                for num, comment in enumerate(author_comments):
                    sent_list = custom_split(_clean_text(comment['text']), ['.', '?', '!'])
                    sent_list = [' '.join(sent.split()) for sent in sent_list]
                    bug_info[num + 1] = {'creator': comment['creator'], 'creation_time': comment['creation_time'],
                                         'text': sent_list}
                return bug_id, bug_info
            else:
                return False, False
        except:
            return False, False

    def calculate_bugs_len(self): # tested
        """
        Calculate the number of bug reports in each product category
        :return {'product1': bug_length1, ...}
        """
        bugreport_len = {}
        if len(self.product_list) < self.n_cpu:
            pool = Pool(len(self.product_list))
            divisor = len(self.product_list)
        else:
            pool = Pool(self.n_cpu)
            divisor = self.n_cpu
        for d in pool.imap(self._get_bug_ids, self.product_list, round((1 + len(self.product_list)) / divisor)):
            product, bug_ids_list = d
            if len(bug_ids_list) > 0:
                bugreport_len[product] = bug_ids_list
        pool.close()
        pool.join()
        return bugreport_len

    def _get_bug_ids(self, product): # tested
        """
        Conduct query into the bug repository based on the product category and retrieve the bug ids
        :return [bug_id1, bug_id2, ...]
        """
        session = requests.Session()
        response = session.get(self.api + '?product=' + product + '&include_fields=id')
        bug_ids = response.json()
        bug_ids_list = [str(id['id']) for id in bug_ids['bugs']]
        if self.no_finetune_data is False:
            _finetune_ids = list(self.finetune_ids.items())
            ids_notfound = [pair for pair in _finetune_ids if pair[-1] is False]
            for id_pair in ids_notfound:
                id = id_pair[0]
                if id in bug_ids_list:
                    bug_ids_list.remove(id)
                    self.finetune_ids[id] = True
        return product, bug_ids_list

def main():
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--platform_url",
                        default=None,
                        type=str,
                        required=True,
                        help="The advanced search url of the bugzilla platform for extracting the product list.")
    parser.add_argument("--no_url",
                        default=False,
                        type=str2bool,
                        nargs='?',
                        const=True,
                        help="Statement is true when the url of the bugzilla platform is incompatible with the html parser.")
    parser.add_argument("--products_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The file that consists of the product list for the bugzilla platform with incompatible url.")
    parser.add_argument("--api",
                        default=None,
                        type=str,
                        required=True,
                        help="The api address for the specific bugzilla platform.")
    parser.add_argument("--n_cpu",
                        default=None,
                        type=int,
                        required=True,
                        help="The number of cpus to use for multithreading.")
    parser.add_argument("--save_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file path for storing the extracted data.")
    parser.add_argument("--finetune_ids_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The file path where the dataset for finetuning is stored.")
    parser.add_argument("--include_extra_info",
                        default=False,
                        type=str2bool,
                        nargs='?',
                        const=True,
                        help="Option to choose if data sourcing includes additional information such as author etc.")

    args = parser.parse_args()
    if args.no_url:
        product_list = []
        with open(args.products_file, 'r') as f:
            for line in f:
                product_list.append(line[:-1])
    else:
        product_list = get_product_lists(args.platform_url)
    bugzilla = DataExtract(args, product_list)

    bugzilla.source_data(include_extra_info=args.include_extra_info)

if __name__ == '__main__':
    main()