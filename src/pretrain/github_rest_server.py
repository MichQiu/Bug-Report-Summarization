import requests
import time
from others.utils import _clean_text

'''
app = Flask(__name__)
api = Api(app)

class Users(Resource):

    def get_issues(self):
        pass

class Locations(Resource):
    pass
'''

s = requests.Session()
r = s.get('https://api.github.com/repos/vuejs/vue/issues?state=all&page=80&per_page=100')
json_data = r.json()
text = [sent for sent in json_data[3]['body'].split('\n') if len(sent) > 0]
text[-2]

t1 = time.time()

page = 1
per_page = 100
issues_extracted = 0
issues_dict = {}

while per_page == 100:
    r = s.get('https://api.github.com/repos/vuejs/vue/issues?state=all&page='+str(page)+'&per_page='+str(per_page))
    json_data = r.json()
    for i in range(len(json_data)):
        issue_no = json_data[i]['number']
        text = _clean_text(' '.join(json_data[i]['body'].split('\n')))
        issues_dict[issue_no] = text
    page += 1
    per_page = len(json_data)
    issues_extracted += per_page
    print("Number of issues extracted:", issues_extracted)



t2 = time.time()

t2 - t1

'''
api.add_resource(Users, '/users')
api.add_resource(Locations, '/locations')


if __name__ == '__main__':
    app.run()
'''
