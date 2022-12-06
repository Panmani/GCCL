# ---------- Basic dependencies ----------
from tqdm import tqdm
import os
import pickle
import numpy as np
import json
from collections import defaultdict
import random
import traceback

# ---------- Model dependencies ----------
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split

# ---------- local ----------
# from download import read_wiki_doc
from config import *


def read_wiki_doc(doc_name):
    # encoding="utf8" is need for running on Windows
    with open(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name + ".txt"), "r", encoding="utf8") as doc_file:
    # with open(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name + ".txt"), "r") as doc_file:
        doc = doc_file.read()
    return doc

def train_val_test_split(meta_data_list, split_sizes=[0.7, 0.2, 0.1], random_seed = None):

    assert (sum(split_sizes) - 1.0) < 1e-5
    val_test_size = split_sizes[1] + split_sizes[2]
    test_size = split_sizes[2] / val_test_size

    train_meta, val_test_meta = train_test_split(meta_data_list,
                                                test_size=val_test_size,
                                                random_state=random_seed)
    val_meta, test_meta = train_test_split(val_test_meta,
                                                test_size=test_size,
                                                random_state=random_seed)

    return train_meta, val_meta, test_meta

def official_train_test_split(meta_data_list):
    train_meta = []
    test_meta = []
    for meta in meta_data_list:
        if meta['split'] == "Train":
            train_meta.append(meta)
        elif meta['split'] == "Test":
            test_meta.append(meta)
        else:
            print("In neither split")
    return train_meta, test_meta

def load_word_for_bert(meta_data_list,
                        debug_size = None,
                        truncate_at = None):

    if truncate_at is not None:
        print("-------------- Truncation at {} --------------".format(truncate_at))
    doc_s_list, doc_t_list, scores = [], [], []
    for index, cur_pair in enumerate(tqdm(meta_data_list)):
        try:
            wiki_doc1 = read_wiki_doc(cur_pair["doc_pair"][0])
            wiki_doc2 = read_wiki_doc(cur_pair["doc_pair"][1])

            if len(wiki_doc1) > 0 and len(wiki_doc2) > 0:
                if truncate_at is not None:
                    # Data augmentation by truncating long documents
                    doc1_split = wiki_doc1.split()
                    doc2_split = wiki_doc2.split()
                    for trunc_len in truncate_at:
                        if len(doc1_split) > trunc_len or len(doc2_split) > trunc_len:
                            doc_s_list.append( " ".join(doc1_split[:trunc_len]) )
                            doc_t_list.append( " ".join(doc2_split[:trunc_len]) )
                            scores.append(cur_pair['score'])
                else:
                    doc_s_list.append(wiki_doc1)
                    doc_t_list.append(wiki_doc2)
                    scores.append(cur_pair['score'])

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(repr(e))
            print(traceback.format_exc())
            continue

    if truncate_at is not None:
        print("Truncation data augmentation: from {} raw doc pairs, generated {} truncated doc pairs".format( len(meta_data_list), len(scores) ))

    if debug_size is None:
        return doc_s_list, doc_t_list, scores
    else:
        assert debug_size > 0
        return doc_s_list[:debug_size], doc_t_list[:debug_size], scores[:debug_size]

def augment_dataset(meta_list, cluster_threshold,
                                use_commutativity = True,
                                use_transitivity = True):
    """
    Augment the dataset according to the following two properties of relatedness:

        1. Relatedness is transitive (a ~ b & b ~ c ==> a ~ c)
        2. Relatedness is commutative (a ~ b ==> b ~ a)

    The method is to cluster (strongly) related docs, and
    sample from the clusters related and unrelated doc pairs
    """
    def sort_by_length(lst):
        """Sort the inner lists in a nested list by their lengths"""
        lst2 = sorted(lst, key=len, reverse=True)
        return lst2

    def sorted_length(lst):
        """Sort the length of the inner lists in a nested list"""
        lst2 = sorted(lst, key=len, reverse=True)
        length_sorted = map(len, lst2)
        return list(length_sorted)

    if not use_commutativity and not use_transitivity:
        return meta_list

    print("-------------- Clustering data augmentation --------------")
    # assert cluster_threshold >= 0.1
    comm_aug_rel_count = 0
    comm_aug_unrel_count = 0
    trans_aug_rel_count = 0
    trans_aug_unrel_count = 0

    # Step 1: Build clusters
    doc2cluster = {}
    next_cluster_index = 0
    related_pairs = []
    unrelated_pairs = []
    for meta in meta_list:
        doc_s, doc_t = meta['doc_pair'][0], meta['doc_pair'][1]
        if meta['score'] > SCORE_THRESHOLD:
            # doc_s and doc_t are related
            related_pairs.append((doc_s, doc_t))
            if use_commutativity:
                related_pairs.append((doc_t, doc_s))
                comm_aug_rel_count += 1
            if meta['score'] >= cluster_threshold:
                # The two docs are strongly related (if cluster_threshold > SCORE_THRESHOLD)
                # if cluster_threshold = 0, then it is vanilla clustering
                if doc_s in doc2cluster and doc_t not in doc2cluster:
                    doc2cluster[doc_t] = doc2cluster[doc_s]
                elif doc_s not in doc2cluster and doc_t in doc2cluster:
                    doc2cluster[doc_s] = doc2cluster[doc_t]
                elif doc_s not in doc2cluster and doc_t not in doc2cluster:
                    # The related docs are a new cluster
                    doc2cluster[doc_s] = next_cluster_index
                    doc2cluster[doc_t] = next_cluster_index
                    next_cluster_index += 1
                # If both are in doc2cluster, not action needed
        else:
            # doc_s and doc_t unrelated
            unrelated_pairs.append((doc_s, doc_t))
            if use_commutativity:
                unrelated_pairs.append((doc_t, doc_s))
                comm_aug_unrel_count += 1
            if doc_s not in doc2cluster and doc_t in doc2cluster:
                doc2cluster[doc_s] = next_cluster_index
                next_cluster_index += 1
            elif doc_s in doc2cluster and doc_t not in doc2cluster:
                doc2cluster[doc_t] = next_cluster_index
                next_cluster_index += 1
            elif doc_s not in doc2cluster and doc_t not in doc2cluster:
                doc2cluster[doc_s] = next_cluster_index
                next_cluster_index += 1
                doc2cluster[doc_t] = next_cluster_index
                next_cluster_index += 1
            # If both are in doc2cluster, not action needed

    if use_commutativity:
        print("Augmented pairs using commutativity:", comm_aug_rel_count + comm_aug_unrel_count)
        print("\t related pairs   :", comm_aug_rel_count)
        print("\t unrelated pairs :", comm_aug_unrel_count)

    if use_transitivity:
        cluster2doc = defaultdict(lambda:[])
        for doc in doc2cluster:
            cluster_idx = doc2cluster[doc]
            cluster2doc[cluster_idx].append(doc)

        # print(sort_by_length(cluster2doc.values()))
        # print(sorted_length(cluster2doc.values()))
        # print(sum(sorted_length(cluster2doc.values())))

        # Step 2: Sample from clusters
        for cluster_idx in cluster2doc:
            docs_in_cluster = cluster2doc[cluster_idx]
            if len(docs_in_cluster) > 1:
                for i in range(len(docs_in_cluster) - 1):
                    for j in range(i + 1, len(docs_in_cluster)):
                        if (docs_in_cluster[i], docs_in_cluster[j]) not in related_pairs:
                            related_pairs.append((docs_in_cluster[i], docs_in_cluster[j]))
                            trans_aug_rel_count += 1
                        if (docs_in_cluster[j], docs_in_cluster[i]) not in related_pairs:
                            related_pairs.append((docs_in_cluster[j], docs_in_cluster[i]))
                            trans_aug_rel_count += 1

        while len(related_pairs) - len(unrelated_pairs) > 0:
            i, j = random.sample(range(next_cluster_index), 2)
            if i == j:
                continue

            doc_s = random.choice(cluster2doc[i])
            doc_t = random.choice(cluster2doc[j])
            # i.e., we need to check for conflicting labels
            if (doc_s, doc_t) not in unrelated_pairs and \
                (doc_s, doc_t) not in related_pairs and \
                (doc_t, doc_s) not in related_pairs:
                # transitive augmentation is automatic
                unrelated_pairs.append((doc_s, doc_t))
                trans_aug_unrel_count += 1

        print("Augmented pairs using transitivity:", trans_aug_rel_count + trans_aug_unrel_count)
        print("\t Number of clusters found:", next_cluster_index)
        print("\t Cluster threshold:", cluster_threshold)
        print("\t related pairs   :", trans_aug_rel_count)
        print("\t unrelated pairs :", trans_aug_unrel_count)

    # Step 3: Merge related and unrelated doc pairs
    augmented_meta_list = []
    for doc_pair in related_pairs:
        cur_pair = {
            "doc_pair" : doc_pair,
            "score"    : 1.      ,
        }
        augmented_meta_list.append(cur_pair)

    for doc_pair in unrelated_pairs:
        cur_pair = {
            "doc_pair" : doc_pair,
            "score"    : 0.      ,
        }
        augmented_meta_list.append(cur_pair)
    print("Merged into augmented_meta_list, from old size to new size: {} --[aug]-> {}".format( len(meta_list), len(augmented_meta_list) ))

    return augmented_meta_list

def collate_doc_pair(batch):
    longest_len = 0
    if MAX_SENT_NUM is None:
        for doc_s, doc_t, _ in batch:
            if longest_len < max(doc_s.shape[0], doc_t.shape[0]):
                longest_len = max(doc_s.shape[0], doc_t.shape[0])
    else:
        longest_len = MAX_SENT_NUM

    padded_batch_s = torch.zeros((len(batch), longest_len, MAX_SEQ_LEN), dtype=torch.int64)
    padded_batch_t = torch.zeros((len(batch), longest_len, MAX_SEQ_LEN), dtype=torch.int64)
    sent_len_s_t = torch.zeros((len(batch), 2), dtype=torch.int64)
    batch_labels = torch.zeros((len(batch),), dtype=torch.int64)
    for i, (doc_s, doc_t, label) in enumerate(batch):
        len_s, len_t = doc_s.shape[0], doc_t.shape[0]
        if len_s > longest_len:
            len_s = longest_len
        if len_t > longest_len:
            len_t = longest_len
        padded_batch_s[i, :len_s, :] = doc_s[:len_s, :]
        padded_batch_t[i, :len_t, :] = doc_t[:len_t, :]
        sent_len_s_t[i, 0], sent_len_s_t[i, 1] = (len_s, len_t)
        batch_labels[i] = label
    return padded_batch_s, padded_batch_t, sent_len_s_t, batch_labels

# class DocPairDataset(Dataset):
#     def __init__(self, doc_s_list, doc_t_list, scores, max_len = None, score_threshold = 0.0):
#         self.doc_s_list = doc_s_list
#         self.doc_t_list = doc_t_list
#         self.scores = scores
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#         # max_len is where we truncate the document
#         # MAX_SEQ_LEN is the maximal length that the BERT can take for CRE
#         # MAX_SEQ_LEN = max sequence length of transformer / 2 - 1
#         if max_len is None:
#             self.max_len = MAX_SEQ_LEN
#         else:
#             assert max_len <= MAX_SEQ_LEN and max_len > 0
#             self.max_len = max_len
#
#         self.score_threshold = score_threshold
#         self.id_cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
#         self.id_sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
#
#     def __getitem__(self, index):
#
#         tokenized_s = self.tokenizer.tokenize(self.doc_s_list[index])
#         tokenized_t = self.tokenizer.tokenize(self.doc_t_list[index])
#
#         bert_max_len = 512
#         target_len_s = MAX_SEQ_LEN
#         target_len_t = MAX_SEQ_LEN
#         if len(tokenized_s) <= self.max_len and len(tokenized_t) > self.max_len:
#             target_len_s = len(tokenized_s)
#             target_len_t = bert_max_len - len(tokenized_s) - 2
#             tokenized_t = tokenized_t[:target_len_t]
#         if len(tokenized_s) > self.max_len and len(tokenized_t) <= self.max_len:
#             target_len_s = bert_max_len - len(tokenized_t) - 2
#             target_len_t = len(tokenized_t)
#             tokenized_s = tokenized_s[:target_len_s]
#         if len(tokenized_s) > self.max_len and len(tokenized_t) > self.max_len:
#             tokenized_s = tokenized_s[:target_len_s]
#             tokenized_t = tokenized_t[:target_len_t]
#
#         ids_s  = self.tokenizer.convert_tokens_to_ids(tokenized_s)
#         ids_s = self.pad_sequence(ids_s, target_len_s)
#         ids_t  = self.tokenizer.convert_tokens_to_ids(tokenized_t)
#         ids_t = self.pad_sequence(ids_t, target_len_t)
#         ids_cls_s_sep_t = self.id_cls + ids_s + self.id_sep + ids_t
#         ids_cls_s_sep_t_tensor = torch.tensor(ids_cls_s_sep_t)
#         assert len(ids_cls_s_sep_t) == bert_max_len
#
#         score = self.scores[index]
#         score = (score > self.score_threshold) * 1
#         label = torch.from_numpy(np.array(score))
#
#         return ids_cls_s_sep_t_tensor, label
#
#     def pad_sequence(self, sequence, target_len = MAX_SEQ_LEN):
#         padding = [0] * (target_len - len(sequence))
#         sequence += padding
#         return sequence
#
#     def __len__(self):
#         return len(self.scores)


class WholeDocPairDataset(Dataset):
    def __init__(self, doc_s_list, doc_t_list, scores, max_len = None, score_threshold = 0.0):
        self.doc_s_list = doc_s_list
        self.doc_t_list = doc_t_list
        self.scores = scores
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # max_len is where we truncate the document
        # MAX_SEQ_LEN is the maximal length that the BERT can take for CRE
        # MAX_SEQ_LEN = max sequence length of transformer / 2 - 1
        if max_len is None:
            self.max_len = MAX_SEQ_LEN
        else:
            assert max_len <= MAX_SEQ_LEN and max_len > 0
            self.max_len = max_len

        self.score_threshold = score_threshold
        self.id_cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        self.id_sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])

    def __getitem__(self, index):

        doc_s_ids_mtx = self.tokenize_whole_doc_sents(self.doc_s_list[index])
        doc_t_ids_mtx = self.tokenize_whole_doc_sents(self.doc_t_list[index])
        # doc_s_t_stack = torch.cat((doc_s_ids_mtx, doc_t_ids_mtx), 0)

        score = (self.scores[index] > self.score_threshold) * 1
        label = torch.from_numpy(np.array(score))

        return doc_s_ids_mtx, doc_t_ids_mtx, label
        # return doc_s_t_stack, label

    def __len__(self):
        return len(self.scores)

    def pad_sequence(self, sequence, target_len = MAX_SEQ_LEN):
        padding = [0] * (target_len - len(sequence))
        sequence += padding
        return sequence

    def tokenize_whole_doc_sents(self, doc):
        doc_sents = doc.split(".")
        doc_sent_ids_list = []
        for sent in doc_sents:
            sent_strip = sent.strip()
            sent_tokens = self.tokenizer.tokenize(sent_strip)
            if len(sent_tokens) > self.max_len - 2:
                # print("A doc sent exceeds max seq len: doc_len={} > {}=max_seq_len, doc content: {}".format(len(sent_tokens), self.max_len, sent_strip))
                sent_tokens = sent_tokens[:self.max_len - 2]
            sent_ids  = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_ids_with_cls = self.id_cls + sent_ids + self.id_sep
            sent_ids_with_cls_padded = self.pad_sequence(sent_ids_with_cls, self.max_len)
            assert len(sent_ids_with_cls_padded) == self.max_len
            doc_sent_ids_list.append(sent_ids_with_cls_padded)

        doc_sent_ids_mtx = torch.tensor(doc_sent_ids_list)
        return doc_sent_ids_mtx

if __name__ == '__main__':

    with open(os.path.join(RAW_DATA_DIR, META_DATA_NAME), "r") as meta_file:
        meta_data_list = json.load(meta_file)

    pos_count = 0
    for meta in meta_data_list:
        if meta['score'] > SCORE_THRESHOLD:
            pos_count += 1
    print("Positive ratio: {} / {}".format(pos_count, len(meta_data_list)))

    train_meta, test_meta = official_train_test_split(meta_data_list)
    train_meta, val_meta = train_test_split(train_meta,
                                            test_size=0.1,
                                            random_state=42)

    print(len(train_meta))
    print(len(val_meta))
    print(len(test_meta))

    # train_meta, val_meta, test_meta = train_val_test_split(meta_data_list,
    #                                                     split_sizes=[0.7, 0.2, 0.1],
    #                                                     random_seed = 42)

    # Create training set: Step 1 & 2
    # Step 1: Data augmentation (Clustering & Truncation)
    augmented_train_meta = augment_dataset(train_meta, cluster_threshold = 0.8, use_commutativity = True, use_transitivity = True)

    # Step 2: Create final training set by merging original set with the augmentation sets
    # print("-------------- Load original training & test set --------------")
    # doc_s_list_cluster, doc_t_list_cluster, scores_cluster = load_word_for_bert(augmented_train_meta, "train", debug_size = None)
    # doc_s_list = doc_s_list_trunc + doc_s_list_cluster
    # doc_t_list = doc_t_list_trunc + doc_t_list_cluster
    # scores = scores_trunc + scores_cluster
    doc_s_list, doc_t_list, scores = load_word_for_bert(augmented_train_meta)
    print("Original training set size: {}; Final training set size: {}".format( len(train_meta), len(scores) ))
    train_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # Create val set
    doc_s_list, doc_t_list, scores = load_word_for_bert(val_meta, debug_size = None)
    val_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # Create test set
    doc_s_list, doc_t_list, scores = load_word_for_bert(test_meta, debug_size = None)
    test_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    dataloaders_dict = {'train' : torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_doc_pair),
                        'val'   : torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_doc_pair),
                        'test'  : torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_doc_pair)
                        }
    dataset_sizes = {   'train' : len(train_set),
                        'val'   : len(val_set),
                        'test'  : len(test_set)
                    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(dataloaders_dict)
    print(dataset_sizes)
    print(device)

    for batch in dataloaders_dict["train"]:
        padded_batch_s, padded_batch_t, sent_len_s_t, batch_labels = batch
        # print(padded_batch_s)
        # print(padded_batch_t)
        print(padded_batch_s.shape)
        print(padded_batch_t.shape)
        print(sent_len_s_t)
        print(batch_labels)
        break

    # for batch in dataloaders_dict["train"]:
    #     doc_s_t_mtx, label = batch
    #     print(doc_s_t_mtx)
    #     print(doc_s_t_mtx.shape)
    #     print(label)
    #     # break
