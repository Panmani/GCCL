import os, json
import numpy as np
from tqdm import tqdm
from download import read_wiki_doc
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
import math
import sklearn

from config import *

def extract_text_from_wiki_url(wiki_url):
    # Specify url of the web page
    source = urlopen(wiki_url).read()

    # Make a soup
    soup = BeautifulSoup(source,'lxml')

    # Extract the plain text content from paragraphs
    paras = []
    for paragraph in soup.find_all('p'):
            paras.append(str(paragraph.text))

    # Extract text from paragraph headers
    heads = []
    for head in soup.find_all('span', attrs={'mw-headline'}):
            heads.append(str(head.text))

    # Interleave paragraphs & headers
    text = [val for pair in zip(paras, heads) for val in pair]
    text = ' '.join(text)

    # Drop footnote superscripts in brackets
    text = re.sub(r"\[.*?\]+", '', text)

    # Replace '\n' (a new line) with '' and end the string at $1000.
    text = text.replace('\n', '')
    return text

def load_word_dataset(meta_data_list, split):
    data_list = []
    for index, cur_pair in enumerate(tqdm(meta_data_list)):
        wiki_doc1 = read_wiki_doc(cur_pair["doc_pair"][0])
        wiki_doc2 = read_wiki_doc(cur_pair["doc_pair"][1])
        cur_pair["doc_pair"] = [wiki_doc1, wiki_doc2]
        data_list.append(cur_pair)
    return data_list


if __name__ == '__main__':

    """ Test on WORD dataset """

    rawContentDict = []

    with open("fixed_meta_split.json", "r") as meta_file:
        train_meta, val_meta, test_meta = json.load(meta_file)

    # with open(os.path.join(RAW_DATA_DIR, META_DATA_NAME), "r") as meta_file:
    #     meta_data_list = json.load(meta_file)
    #
    # train_meta, test_meta = train_test_split(meta_data_list,
    #                                             test_size=0.2,
    #                                             random_state=42)

    test_set = load_word_dataset(test_meta, "test")

    for i in range(len(test_set)):
        rawContentDict.append(test_set[i]["doc_pair"][0])
        rawContentDict.append(test_set[i]["doc_pair"][1])

    tokenized_corpus = [doc.split(" ") for doc in rawContentDict]
    bm25 = BM25Okapi(tokenized_corpus)

    SIM_THRESHOLD = 0.4
    correct_pred = 0
    pos_label_count = 0
    test_size = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    pred_score = []
    true_label_list = []
    for index, cur_pair in enumerate(tqdm(test_set)):
        wiki_doc1, wiki_doc2 = cur_pair["doc_pair"]
        query = wiki_doc1.split(" ")
        scores = bm25.get_scores(query)
        scores_ = []

        for i in range(len(scores)):
            scores_.append((scores[i] - min(scores))/(max(scores)-min(scores)))
        cur_sim = scores_[2*index+1]

        binary_pred = int(cur_sim > SIM_THRESHOLD)
        assert cur_sim >= 0 and cur_sim <= 1
        pred_score.append(cur_sim)
        true_label = (cur_pair["score"] > 0) * 1
        true_label_list.append(true_label)

        pos_label_count += true_label
        if binary_pred == true_label:
            correct_pred += 1
            if true_label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if binary_pred == 1:
                FP += 1
            else:
                FN += 1
        test_size += 1
        # print("Index: {} --- Cur Acc: {}; Cur positive percentage: {}".format(test_size, correct_pred / test_size, pos_label_count / test_size))


    print("TP, FP, TN, FN", TP, FP, TN, FN)
    if TP + FN > 0 and TP + FP > 0 and TN + FP > 0:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1_score = 2 / (1/recall + 1/precision)
        specificity = TN / (TN+FP)
        print("Accuracy: {}; Recall: {}; Precision: {}; Specificity: {}; F1: {}".format(correct_pred / test_size, recall, precision, specificity, f1_score))

    auc = sklearn.metrics.roc_auc_score(true_label_list, pred_score)
    print("AUC", auc)
