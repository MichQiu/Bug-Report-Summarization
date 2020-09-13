from __future__ import print_function

import re
import gc
import time
import pickle
from os.path import join as pjoin
from multiprocessing import Pool
from copy import deepcopy
from html.parser import HTMLParser
from others.logging import logger
import urllib.request as urllib2
import pprint
import bugzilla
from pretrain import bugsource, args_info
import torch
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

def bugsource_init(args):
    global bug_source
    if args.mozilla:
        product_list = []
        with open(args.mozilla_products, 'r') as f:
            for line in f:
                product_list.append(line[:-1])
    else:
        product_list = get_product_lists(args.url_platform)
    bug_source = bugsource.BugSource(args, product_list)

args_list = args_info.Args()

flag = True
while flag is True:
    for i, j in enumerate(args_list.args):
        print(i, list(args_list.files.keys())[i])
    platform = input("Please select the bug platform to source from by choosing its number: ")
    print("Loading arguments...")
    if int(platform) not in range(len(args_list.args)):
        flag = True
    else:
        flag = False
bugsource_init(args_list.args[int(platform)])
print("Finished loading")

def calculate_bugs_len(): #tested
    """
    Calculate the number of bug reports in each product category
    :return {'product1': bug_length1, ...}
    """
    bugreport_len = {}
    pool = Pool(10)
    for d in pool.imap(_get_bug_id, bug_source.product_list, round((1 + len(bug_source.product_list)) / 10)):
        product, bug_ids = d
        if len(bug_ids) > 0:
            bugreport_len[product] = bug_ids
    pool.close()
    pool.join()
    return bugreport_len

def _get_bug_id(product):  #tested
    """
    Conduct query into the bug repository based on the product category and retrieve the bug ids
    :return [bug_id1, bug_id2, ...]
    """
    id_query = bug_source.bzapi.build_query(product=product, include_fields=['id'])
    bug_ids = bug_source.bzapi.query(id_query)
    for i in range(len(bug_ids)):
        bug_ids[i] = bug_ids[i].id
    if bug_source.no_finetune_data is False:
        _finetune_ids = list(bug_source.finetune_ids.items())
        ids_notfound = [pair for pair in _finetune_ids if pair[-1] is False]
        for id_pair in ids_notfound:
            id = id_pair[0]
            if id in bug_ids:
                bug_ids.remove(id)
                bug_source.finetune_ids[id] = True
    return product, bug_ids

def _source(product, bug_ids): #tested
    """Get the bug ids from the product and extract their comments"""
    bug_comments = {}
    if len(bug_ids) < 10 - 1:
        pool = Pool(len(bug_ids))

        for d in pool.imap(_get_text, bug_ids, round((1 + len(bug_ids)) / len(bug_ids))):
            bug_id, src_text = d
            if bug_id is False and src_text is False:
                continue
            elif len(src_text) >= 8:
                bug_comments[bug_id] = src_text
        pool.close()
        pool.join()
        return product, bug_comments
    else:
        pool = Pool(10)
        for d in pool.imap(_get_text, bug_ids, round((1 + len(bug_ids)) / 10)):
            bug_id, src_text = d
            if bug_id is False and src_text is False:
                continue
            elif len(src_text) >= 8:
                bug_comments[bug_id] = src_text
        pool.close()
        pool.join()
        return product, bug_comments

def _get_text(bug_id): #tested
    """Get the comments of an individual bug via its bug id"""
    try:
        bug = bug_source.bzapi.getbug(bug_id)
        comments = bug.getcomments()
        if len(comments) > 0:
            src_text = []
            split_chars = ['.', '?', '!']
            src_text.append(comments[0]['text'])
            for j in range(1, len(comments)):
                text = comments[j]['text']
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

'''
prod_list = get_product_lists(args.url_platform)
prod_list = prod_list[1:3]
bug_source = bugsource.BugSource(args, prod_list)
'''

def source_data():
    bugreport_len = calculate_bugs_len()
    char_list = ["/", "."]
    for product in list(bugreport_len.keys()):
        product, comments = _source(product, bugreport_len[product])
        if len(comments) > 0:
            product = ['_' if char in char_list else char for char in product]
            product = ''.join(product)
            save_file = pjoin(args_list.args[int(platform)].save_path, product + '_bert.pt')
            torch.save(comments, save_file)
        del product, comments

blen = calculate_bugs_len()

from os import listdir
onlyfiles = [f for f in listdir('/home/mich_qiu/PycharmProjects/MSc_Thesis/data/Pretraining/kernel/')]
files = [file[:-8] for file in onlyfiles]

new_list = []
for i in list(blen.keys()):
    if i not in files:
        new_list.append(i)

#bug_source.product_list = new_list[2:]




'''
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
product, bug_comments = _source(product, bug_ids)
t2 = time.time()
print("Query processing time: %s" % (t2 - t1))
bugs = bzapi.query(query)


def get_text(idx):
    bug_id = bugs[idx].id
    bug = bzapi.getbug(bug_id)
    comments = bug.getcomments()
    src_text = []
    split_chars = ['.', '?', '!']
    src_text.append(comments[0]['raw_text'])
    for j in range(1, len(comments)):
        text = comments[j]['raw_text']
        text = _clean_text(text)
        clear_whitespace = text.split()
        text = ' '.join(clear_whitespace)
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
pool = Pool(8)
for d in pool.imap(get_text, list(range(20)), 2):
    bug_id, src_text = d
    bug_comment[bug_id] = src_text
pool.close()
pool.join()
t2 = time.time()
print("Comment fetching processing time: %s" % (t2 - t1))

t1 = time.time()
bug_comment = {}
pool = Pool(8)
for d in pool.imap(_get_text, bug_ids[:20], 2):
    bug_id, src_text = d
    if len(src_text) >= 15:
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
'''
