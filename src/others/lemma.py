import spacy

nlp = spacy.load("en_core_web_lg")

with open("/home/mich_qiu/PycharmProjects/MSc_Thesis/Bug-Report-Summarization/src/others/h_solution.txt", 'r') as f:
    new_sents = []
    for line in f:
        doc = nlp(line)
        pos_noun_idx = []
        pos_noun_idx2 = []
        count = 0
        for token in doc:
            if token.pos_ == "NOUN":
                pos_noun_idx.append("someone")
                pos_noun_idx2.append("john")
                count += 1
            elif token.pos_ == "PRON":
                pos_noun_idx.append("cat")
                pos_noun_idx2.append("john")
                count += 1
            elif token.pos_ == "PROPN":
                pos_noun_idx.append("someone")
                pos_noun_idx2.append("cat")
                count += 1
            elif token.text == "verb":
                pos_noun_idx.append("run")
                pos_noun_idx2.append("run")
            elif token.text == "link":
                pos_noun_idx.append("http://www.github.com")
                pos_noun_idx2.append("http://www.github.com")
            elif token.text == "date":
                pos_noun_idx.append("21/09/2020")
                pos_noun_idx2.append("21/09/2020")
            elif token.text == "modal":
                pos_noun_idx.append("should")
                pos_noun_idx2.append("should")
            else:
                pos_noun_idx.append(token.text)
                pos_noun_idx2.append(token.text)
        if count < 1:
            continue
        sent = ' '.join(pos_noun_idx)
        sent2 = ' '.join(pos_noun_idx2)
        new_sents.extend([sent, sent2])
    f.close()
with open("/home/mich_qiu/PycharmProjects/MSc_Thesis/Bug-Report-Summarization/src/others/h_solution.txt", 'a') as f:
    f.write('\n')
    for sent in new_sents:
        f.write(sent)
    f.close()



