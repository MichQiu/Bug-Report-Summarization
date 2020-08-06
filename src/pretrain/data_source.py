from __future__ import print_function

import re
import time
from multiprocessing import Pool
from copy import deepcopy
from html.parser import HTMLParser
import urllib.request as urllib2
import pprint
import bugzilla

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


def _get_product_lists(url):
    parser = MyHTMLParser()
    parser.ls_data = []
    html_page = urllib2.urlopen(url)
    parser.feed(str(html_page.read()))
    return parser.ls_data


class BugSource():

    def __init__(self, args):
        self.args = args
        self.url = args.url
        self.product_list = args.product_list
        self.bzapi = bugzilla.Bugzilla(self.url)

    def source(self):
        product_dict = {}
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(self._source, self.product_list):
            product, bug_comments = d
            product_dict[product] = bug_comments
        pool.close()
        pool.join()
        return product_dict

    def _source(self, product):
        _, bug_ids = self._get_bug_id(product)
        bug_comments = self._get_text(bug_ids)
        return product, bug_comments

    def _get_text(self, bug_ids):
        bug_comments = {}
        for i in range(len(bug_ids)):
            bug_id = bug_ids[i].id
            bug = self.bzapi.getbug(bug_id)
            comments = bug.getcomments()
            src_text = []
            for j in range(len(comments)):
                src_text.append(comments[j]['raw_text'])
            bug_comments[bug_id] = src_text
        return bug_comments

    def _remove_empty_products(self):
        full_bugreport_len = self._calculate_bugs_len()
        bugreport_len = deepcopy(full_bugreport_len)
        for product in bugreport_len:
            if bugreport_len[product] == 0:
                del bugreport_len[product]
        return bugreport_len

    def _calculate_bugs_len(self):
        bugreport_len = {}
        pool = Pool(self.args.n_cpus)
        for d in pool.imap(self._get_bug_id, self.product_list):
            product, bug_ids = d
            bugreport_len[product] = len(bug_ids)
        pool.close()
        pool.join()
        return bugreport_len

    def _get_bug_id(self, product):
        id_query = self.bzapi.build_query(product=product, include_fields=['id'])
        bug_ids = self.bzapi.query(id_query)
        return product, bug_ids


URL = 'bugzilla.mozilla.org'
bzapi = bugzilla.Bugzilla(URL)

query = bzapi.build_query(
    product="addons.mozilla.org",
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

t1 = time.time()
bug_comment = {}
for i in range(len(bugs[:5])):
    bug_id = bugs[i].id
    bug = bzapi.getbug(bug_id)
    comments = bug.getcomments()
    src_text = []
    for j in range(len(comments)):
        src_text.append(comments[j]['raw_text'])
    bug_comment[bug_id] = src_text
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

prods_0 = ['Calendar', 'Cloud Services', 'Data Platform and Tools', 'Emerging Markets', 'Fenix', 'Firefox',
            'Firefox for Android', 'Firefox for Echo Show', 'Firefox for FireTV', 'Firefox for iOS',
            'Firefox Private Network', 'Focus', 'Focus-iOS', 'Instantbird', 'Lockwise', 'Mozilla Localizations',
            'Mozilla VPN', 'Other Applications', 'Pocket', 'SeaMonkey', 'Thunderbird', 'Web Compatibility']
prods_1 = ['bugzilla.mozilla.org', 'Conduit', 'Developer Infrastructure', 'Firefox Build System', 'Tree Management'];
prods_2 = ['Chat Core', 'Core', 'DevTools', 'Directory', 'External Software Affecting Firefox', 'GeckoView', 'JSS',
            'MailNews Core', 'NSPR', 'NSS', 'Remote Protocol', 'Testing', 'Toolkit', 'WebExtensions']
prods_3 = ['addons.mozilla.org', 'Bugzilla', 'Socorro', 'Tecken', 'Testopia', 'Webtools']
prods_4 = ['Air Mozilla', 'Community Building', 'Data & BI Services Team', 'Data Compliance', 'Data Science',
            'Developer Ecosystem', 'Developer Engagement', 'Developer Services', 'developer.mozilla.org',
            'Enterprise Information Security', 'Firefox Friends', 'Infrastructure & Operations', 'Instantbird Servers',
            'Internet Public Policy', 'L20n', 'Localization Infrastructure and Tools', 'Location', 'Marketing',
            'Marketing Graveyard', 'Mozilla China', 'Mozilla Foundation', 'Mozilla Foundation Communications',
            'Mozilla Grants', 'Mozilla Messaging', 'Mozilla Metrics', 'Mozilla QA', 'Mozilla Reps', 'mozilla.org',
            'Participation Infrastructure', 'Plugin Check', 'Product Innovation', 'quality.mozilla.org',
            'Release Engineering', 'Shield', 'Snippets', 'support.mozilla.org', 'support.mozilla.org - Lithium',
            'Taskcluster', 'Tracking', 'Untriaged Bugs', 'User Experience Design', 'User Research', 'UX Systems',
            'Web Apps', 'Websites', 'www.mozilla.org']

product_list_mozilla = prods_0 + prods_1 + prods_2 + prods_3 + prods_4

