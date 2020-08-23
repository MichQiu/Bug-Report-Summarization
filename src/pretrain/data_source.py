from __future__ import print_function

import re
import time
from os.path import join as pjoin
from multiprocessing import Pool
from copy import deepcopy
from html.parser import HTMLParser
from others.logging import logger
import urllib.request as urllib2
import pprint
import bugzilla
import torch

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
            non_space = re.search(r'[\\n]', data)
            if non_space is None:
                self.ls_data.append(data)

def get_product_lists(url):
    parser = MyHTMLParser()
    parser.ls_data = []
    html_page = urllib2.urlopen(url)
    logger.info("Parsing data from %s" % url)
    parser.feed(str(html_page.read()))
    logger.info("Parsing complete")
    return parser.ls_data


class BugSource():

    def __init__(self, args, product_list):
        self.args = args
        self.url = args.url
        self.product_list = product_list
        self.finetune_ids = open(args.finetune_ids, 'r')
        self.bzapi = bugzilla.Bugzilla(self.url)

    def source(self):
        """
        Obtain a dictionary that contains every product in the platform as its keys
        Within each product, there is a dictionary of bug ids and their associated bug comments
        :return {'product_A': {bug_id1:[src_text1], ...}, ...}
        """
        logger.info('Processing products...')
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(self._source, self.product_list):
            product, bug_comments = d
            save_file = pjoin(self.args.save_path, product+'_bert.pt')
            logger.info('Saving to %s' % save_file)
            torch.save(bug_comments, save_file)
        pool.close()
        pool.join()
        logger.info('Processed products')

    def _source(self, product):
        """Get the bug ids from the product and extract their comments"""
        _, bug_ids = self._get_bug_id(product)
        bug_comments = {}
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(self._get_text, bug_ids):
            bug_id, src_text = d
            bug_comments[bug_id] = src_text
        pool.close()
        pool.join()
        return product, bug_comments

    def _get_text(self, bug_id):
        """Get the comments of an individual bug via its bug id"""
        bug = self.bzapi.getbug(bug_id)
        comments = bug.getcomments()
        src_text = []
        split_chars = ['.', '?', '!']
        src_text.append(comments[0]['raw_text'])
        for j in range(1, len(comments)):
            text = comments[j]['raw_text']
            text = text.replace('\n', ' ')
            split_text = custom_split(text, split_chars)
            for sent in split_text:
                src_text.append(sent)
        return bug_id, src_text

    def remove_empty_products(self):
        """Remove all product categories with 0 bug reports"""
        logger.info("Calculating number of bugs in platform products...")
        full_bugreport_len = self._calculate_bugs_len()
        logger.info("Calculation complete")
        bugreport_len = deepcopy(full_bugreport_len)
        for product in bugreport_len:
            if bugreport_len[product] == 0:
                del bugreport_len[product]
        logger.info("Removed all product categories with 0 bugs")
        return bugreport_len

    def _calculate_bugs_len(self):
        """
        Calculate the number of bug reports in each product category
        :return {'product1': bug_length1, ...}
        """
        bugreport_len = {}
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(self._get_bug_id, self.product_list):
            product, bug_ids = d
            bugreport_len[product] = len(bug_ids)
        pool.close()
        pool.join()
        return bugreport_len

    def _get_bug_id(self, product):
        """
        Conduct query into the bug repository based on the product category and retrieve the bug ids
        :return [bug_id1, bug_id2, ...]
        """
        id_query = self.bzapi.build_query(product=product, include_fields=['id'])
        bug_ids = self.bzapi.query(id_query)
        _finetune_ids = list(self.finetune_ids.items())
        ids_notfound = [pair for pair in _finetune_ids if pair[-1] is False]
        for id_pair in ids_notfound:
            id = id_pair[0]
            if id in bug_ids:
                bug_ids.remove(id)
                self.finetune_ids[id] = True
        return product, bug_ids

def custom_split(input_string, split_chars):
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
        elif char in split_chars and char_list[idx+1] == ' ':
            end_idx = idx
            char_list_selection = char_list[start_idx:end_idx+1]
            split_string = ''.join(char_list_selection)
            split_string_list.append(split_string)
            start_idx = idx + 2
    return split_string_list

def source_data(args, mozilla=False):
    if mozilla:
        product_list = []
        with open(args.mozilla_products, 'r') as f:
            for line in f:
                product_list.append(line)
    else:
        product_list = get_product_lists(args.url_platform)
    bug_source = BugSource(args, product_list)
    bugreport_len = bug_source.remove_empty_products()
    bug_source.product_list = list(bugreport_len.keys())
    bug_source.source()



URL = 'bugzilla.mozilla.org'
bzapi = bugzilla.Bugzilla(URL)

query = bzapi.build_query(
    product="Firefox",
    include_fields=["id"])

bugreport_len = {}

def cal_len(product):
    t1 = time.time()
    query = bzapi.build_query(
        product=product,
        include_fields=["id"])
    bugs = bzapi.query(query)
    t2 = time.time()
    print("Query processing time: %s" % (t2 - t1))
    return product, len(bugs)

def func(x):
    return x + 1

pool = Pool(4)
for d in pool.imap(func, [1, 2, 3, 4]):
    print(d)
pool.close()
pool.join()

def temp():
    bugreport_len = {}
    pool = Pool(8)
    for d in pool.imap(cal_len, prods[:8]):
        product, length = d
        bugreport_len[product] = length
    pool.close()
    pool.join()
    return bugreport_len

a = temp()

t1 = time.time()
bugs = bzapi.query(query)
t2 = time.time()
print("Query processing time: %s" % (t2 - t1))


def get_text(idx):
    bug_id = bugs[idx].id
    bug = bzapi.getbug(bug_id)
    comments = bug.getcomments()
    src_text = []
    split_chars = ['.', '?', '!']
    src_text.append(comments[0]['raw_text'])
    for j in range(1, len(comments)):
        text = comments[j]['raw_text']
        text = text.replace('\n', ' ')
        split_text = custom_split(text, split_chars)
        for sent in split_text:
            src_text.append(sent)
    return bug_id, src_text

def get_text_old(idx):
    bug_id = bugs[idx].id
    bug = bzapi.getbug(bug_id)
    comments = bug.getcomments()
    src_text = []
    for j in range(1, len(comments)):
        text = comments[j]['raw_text']
        src_text.append(text)
    return bug_id, src_text

t1 = time.time()
bug_comment = {}
pool = Pool(10)
for d in pool.imap(get_text, list(range(10))):
    bug_id, src_text = d
    bug_comment[bug_id] = src_text
pool.close()
pool.join()
t2 = time.time()
print("Comment fetching processing time: %s" % (t2 - t1))





bug = bzapi.getbug(440812)
print("Fetched bug #%s:" % bug.id)
print("  Product   = %s" % bug.product)
print("  Component = %s" % bug.component)
print("  Status    = %s" % bug.status)
print("  Resolution= %s" % bug.resolution)
print("  Summary   = %s" % bug.summary)

comments = bug.getcomments()
src_text = []
for i in range(len(comments)):
    src_text.append(comments[i]['raw_text'])


print("\nComments:\n%s" % pprint.pformat(comments[:]))

