class Args():

    def __init__(self, url, url_platform, finetune_ids_file, log_file, mozilla=False, mozilla_products=False, n_cpus=4,
                 save_path=None):
        self.url = url
        self.url_platform = url_platform
        self.finetune_ids_file = finetune_ids_file
        self.log_file = log_file
        self.mozilla = mozilla
        self.mozilla_products = mozilla_products
        self.n_cpus = n_cpus
        self.save_path = save_path

args = Args(
    'bugs.eclipse.org/bugs/xmlrpc.cgi',
    'https://bugs.eclipse.org/bugs/query.cgi?format=advanced',
    '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/prepro/bug_ids.txt',
    '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/logs/pretrain.log',
    n_cpus=8,
    save_path='/home/mich_qiu/PycharmProjects/MSc_Thesis/data/Pretraining/'
)