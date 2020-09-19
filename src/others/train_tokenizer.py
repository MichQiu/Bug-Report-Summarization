from tokenizers import BertWordPieceTokenizer
from os import listdir

data_dir = '/home/mich_qiu/PycharmProjects/MSc_Thesis/data/Pretraining/shards/'
paths = [data_dir + file for file in listdir(data_dir)]

tokenizer = BertWordPieceTokenizer(lowercase=False)

tokenizer.train(files=paths, vocab_size=50000, special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[DES]",
            "[QS]",
            "[CODE]",
            "[INFO]",
            "[SOLU]"])
tokenizer.save_model('/home/mich_qiu/PycharmProjects/MSc_Thesis/data/Pretraining/', "Bugzilla_tokenizer")
