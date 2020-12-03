import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from others.utils import _clean_text, custom_split
import pickle
import argparse
import time

def extract_issues(api_address, page, issues_extracted, username, token, total_requests, current_time):
    s = requests.Session()
    per_page = 100
    issues_dict = {}

    while per_page == 100:
        if time.time() - current_time > 3600:
            total_requests = 0
            current_time = time.time()
        elif total_requests >= 4900:
            print("API access rate limit reached. Now idling for an hour...")
            for sec in tqdm(range(3601)):
                time.sleep(1)
            total_requests = 0
            print("API access now available!")
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
        total_requests += 1
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

    return issues_txt_dicts, issues_sent_dicts, total_requests, current_time

def save_data(issues_txt_dicts, issues_sent_dicts, file_path_txt, file_path_bin, file_name):
    with open(file_path_txt + '/' + file_name + '.txt', 'w') as f:
        print("Saving plain text file...")
        for issue_no in issues_txt_dicts.keys():
            f.write(issues_txt_dicts[issue_no]+'\n')
            f.write(' \n')
    print("Text file saved!")

    with open(file_path_bin + '/' + file_name + '.pkl', 'wb') as f:
        print("Saving binary file...")
        pickle.dump(issues_sent_dicts, f)
    print("Binary file saved!")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_address_list",
                        default=None,
                        type=str,
                        required=True,
                        help="The file that contains a list of api address to extract issues from.")
    parser.add_argument("--text_file_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file path for storing the data in plain text format (.txt).")
    parser.add_argument("--binary_file_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The file path for storing the data in binary format (.pkl).")
    parser.add_argument("--github_username",
                        default=None,
                        type=str,
                        required=True,
                        help="The github username used for basic HTTP authentication to access the github api")
    parser.add_argument("--personal_access_token",
                        default=None,
                        type=str,
                        required=True,
                        help="The personal access token required for basic HTTP authentication to access the github api")
    parser.add_argument("--page_number",
                        default=1,
                        type=int,
                        required=False,
                        help="Set the page number of the api address to extract the issues from.")
    parser.add_argument("--number_of_issues_extracted",
                        default=0,
                        type=int,
                        required=False,
                        help="Set the number of issues that are already extracted. for printing purposes only.")

    args = parser.parse_args()
    total_requests = 0
    current_time = time.time()

    with open(args.api_address_list, 'r') as f:
        for line in f:
            project_name = []
            slash_count = 0
            for i in range(29, len(line)):
                if line[i] == '/':
                    slash_count += 1
                    if slash_count == 2:
                        break
                    else:
                        project_name.append('_')
                else:
                    project_name.append(line[i])
            project_name = ''.join(project_name)
            print("### Extracting issues from: " + project_name + " ###")
            issues_txt_dicts, issues_sent_dicts, total_requests, current_time = extract_issues(line[:-1],
                                                                                 args.page_number,
                                                                                 args.number_of_issues_extracted,
                                                                                 args.github_username,
                                                                                 args.personal_access_token,
                                                                                 total_requests, current_time)
            save_data(issues_txt_dicts, issues_sent_dicts, args.text_file_path, args.binary_file_path,
                      project_name)

if __name__ == '__main__':
    main()