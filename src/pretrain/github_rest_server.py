import requests
from requests.auth import HTTPBasicAuth
from others.utils import _clean_text, custom_split
from tokenization import BasicTokenizer
import pickle

def extract_issues(api_address, page, issues_extracted, username, token):
    s = requests.Session()
    per_page = 100
    issues_dict = {}

    while per_page == 100:
        print("Current page number:", page)
        r = s.get(api_address + '?state=all&page=' + str(page) + '&per_page=100',
                  auth=HTTPBasicAuth(username, token))
        json_data = r.json()
        for i in range(len(json_data)):
            if json_data[i]['body'] is None:
                continue
            else:
                issue_no = json_data[i]['number']
                text = _clean_text(' '.join(json_data[i]['body'].split('\n')))
                issues_dict[issue_no] = text
        page += 1
        per_page = len(json_data)
        issues_extracted += per_page
        print("Number of issues extracted:", issues_extracted)

    issues_sent_dicts = {}
    issues_txt_dicts = {}

    for issue_no in issues_dict.keys():
        split_chars = ['!', '?', '.']
        text = issues_dict[issue_no]
        sent_list = custom_split(text, split_chars)
        new_sent_list = []
        if len(sent_list) < 8:
            continue
        else:
            for sent in sent_list:
                split_sent = sent.split()
                if len(split_sent) < 3:
                    continue
                else:
                    new_sent_list.append(' '.join(split_sent))
        if len(new_sent_list) < 8:
            continue
        else:
            issues_txt_dicts[issue_no] = text
            issues_sent_dicts[issue_no] = new_sent_list

    return issues_txt_dicts, issues_sent_dicts

def save_data(issues_txt_dicts, issues_sent_dicts, file_path_txt, file_path_bin, file_name):
    with open(file_path_txt + file_name + '.txt', 'w') as f:
        print("Saving plain text file...")
        for issue_no in issues_txt_dicts.keys():
            f.write(issues_txt_dicts[issue_no]+'\n')
            f.write(' \n')
    print("Text file saved!")

    with open(file_path_bin + file_name + '.pkl', 'wb') as f:
        print("Saving binary file...")
        pickle.dump(issues_sent_dicts, f)
    print("Binary file saved!")


if __name__ == '__main__':
    issues_txt_dicts, issues_sent_dicts = extract_issues('https://api.github.com/repos/flutter/flutter/issues',
                                                         1, 0, 'MichQiu',
                                                         '7b9b770b9371c1d069a2372f4dd16c4f03976582')
    save_data(issues_txt_dicts, issues_sent_dicts,
              '/home/mich_qiu/PycharmProjects/Bug-Report-Summarization/src/pretrain/github_repos_text/',
              '/home/mich_qiu/PycharmProjects/Bug-Report-Summarization/src/pretrain/github_repos_binary/', 'flutter')
