import pickle
import spacy
from copy import deepcopy

nlp = spacy.load("en_core_web_lg")

with open("/home/mich_qiu/PycharmProjects/MSc_Thesis/Bug-Report-Summarization/src/others/h_solution.txt", 'r') as f:
    sent_idx = 1
    heur_list = []
    pos_lst = ["NOUN", "PRON", "PROPN"]
    for line in f:
        doc = nlp(line)
        heur_sent = []
        heur_sent2 = []
        heur_sent3 = []
        #print("Sentence {}".format(sent_idx))
        for token in doc:
            if token.pos_ == "PUNCT" or token.text == '\n':
                continue
            for idx, pos in enumerate(pos_lst):
                if token.pos_ == pos:
                    pos_other = deepcopy(pos_lst)
                    pos_other.pop(idx)
                    heur_sent.append({"POS": token.pos_})
                    heur_sent2.append({"POS": pos_other[0]})
                    heur_sent3.append({"POS": pos_other[-1]})
            if heur_sent2:
                if token.text == "verb":
                    heur_sent.append({"POS": "VERB"})
                    heur_sent2.append({"POS": "VERB"})
                    heur_sent3.append({"POS": "VERB"})
                elif token.text == "modal":
                    heur_sent.append({"POS": "AUX"})
                    heur_sent2.append({"POS": "AUX"})
                    heur_sent3.append({"POS": "AUX"})
                elif token.text == "link":
                    heur_sent.append({"POS": "X"})
                    heur_sent2.append({"POS": "X"})
                    heur_sent3.append({"POS": "X"})
                elif token.text == "date":
                    heur_sent.append({"POS": "NUM"})
                    heur_sent2.append({"POS": "NUM"})
                    heur_sent3.append({"POS": "NUM"})
                elif token.pos_ not in pos_lst:
                    heur_sent.append({"LEMMA": token.lemma_, "POS": token.pos_})
                    heur_sent2.append({"LEMMA": token.lemma_, "POS": token.pos_})
                    heur_sent3.append({"LEMMA": token.lemma_, "POS": token.pos_})
            else:
                if token.text == "verb":
                    heur_sent.append({"POS": "VERB"})
                elif token.text == "modal":
                    heur_sent.append({"POS": "AUX"})
                elif token.text == "link":
                    heur_sent.append({"POS": "X"})
                elif token.text == "date":
                    heur_sent.append({"POS": "NUM"})
                elif token.pos_ not in pos_lst:
                    heur_sent.append({"LEMMA": token.lemma_, "POS": token.pos_})
        if heur_sent2:
            heur_list.extend([heur_sent, heur_sent2, heur_sent3])
        else:
            heur_list.append(heur_sent)

with open('/home/mich_qiu/PycharmProjects/MSc_Thesis/h_solution.pkl', 'wb+') as w:
    pickle.dump(heur_list, w)

