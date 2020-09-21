from tokenizers import BertWordPieceTokenizer
from os import listdir
import argparse

def train_tokenizer(args):

    paths = [args.data_dir + file for file in listdir(args.data_dir)]
    tokenizer = BertWordPieceTokenizer(lowercase=False)
    tokenizer.train(files=paths, vocab_size=args.vocab_size, special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[DES]",
                "[QS]",
                "[CODE]",
                "[SOLU]",
                "[INFO]",
                "[NON]"])
    tokenizer.save_model(args.save_dir, "Bugzilla_tokenizer")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default='', type=str)
    arg_parser.add_argument("--save_dir", default='', type=str)
    arg_parser.add_argument("--vocab_size", default=50000, type=int)

    args = arg_parser.parse_args()
    train_tokenizer(args)
