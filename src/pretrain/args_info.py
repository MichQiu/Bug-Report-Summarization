import pickle

class Args():

    def __init__(self):
        self.files = {'mozilla': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/mozilla.pkl',
                 'eclipse': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/eclipse.pkl',
                 'kde': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/kde.pkl',
                 'gnome': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/gnome.pkl',
                 'kernel': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/kernel.pkl',
                 'apache': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/apache.pkl',
                 'redhat': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/redhat.pkl',
                 'gentoo': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/gentoo.pkl',
                 'novell': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/novell.pkl',
                 'suse': '/home/mich_qiu/PycharmProjects/MSc_Thesis/PreSumm_Bug/src/pretrain/suse.pkl'
                 }

        self.args = []
        for platform in self.files.keys():
            with open(self.files[platform], 'rb') as f:
                arg = pickle.load(f)
            self.args.append(arg)