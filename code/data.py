# ---------- Basic dependencies ----------
from tqdm import tqdm
import os
import pickle
import numpy as np
import json
import copy
import argparse
import pandas as pd
from collections import defaultdict
import random
from statistics import mean
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Model dependencies ----------
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx


class Concept(object):
    def __init__(self, id, name, doc_name):
        self.id = id
        self.name = name
        self.doc_name = doc_name

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        truncate = 80
        s = "ID: " + str(self.id) + " |||| " + self.name + "\n"
        s += f"File: {self.doc_name}.txt\n"
        s += "Content: " + self.doc[:truncate] + " ..." if len(self.doc) > truncate else ""
        return s

    def preprocess(self, yaml_config, tokenizer, cns_doc_list = None):
        if cns_doc_list is None:
            self.doc = read_wiki_doc(self.doc_name, yaml_config)
            dataset_type = 'WORD'
        else:
            # For CNS, self.doc_name is the document id given by cns raw dataset file
            cns_doc = cns_doc_list[self.doc_name]
            self.doc = cns_doc['title'] + "\n" + cns_doc['content']
            dataset_type = "CNS"

        self.doc_seg_ids_list, self.doc_seg_mask, self.doc_att_mask_list = \
                split_doc(  self.doc,
                            tokenizer,
                            max_seg_num = yaml_config['DATASET']['MAX_SEG_NUM'], 
                            max_seg_len = yaml_config['DATASET']['MAX_SEG_LEN'],
                            max_par_len = yaml_config['DATASET']['MAX_PART_NUM'],
                            period_char = yaml_config['DATASET'][dataset_type]['PERIOD_CHAR']
                        )

        # tokenized_doc = tokenizer.tokenize(self.doc)
        # self.doc_ids = tokenizer.convert_tokens_to_ids(tokenized_doc)

    # def preprocess(self, yaml_config, tokenizer):
    #     self.doc = read_wiki_doc(self.doc_name, yaml_config)
    #     # try:
    #     # except KeyboardInterrupt:
    #     #     exit()
    #     # except Exception as e:
    #     #     print(repr(e))

    #     self.doc_seg_ids_list, self.doc_seg_mask, self.doc_att_mask_list = \
    #             split_doc(  self.doc,
    #                         tokenizer,
    #                         max_seg_num = yaml_config['DATASET']['MAX_SEG_NUM'], 
    #                         max_seg_len = yaml_config['DATASET']['MAX_SEG_LEN'],
    #                         max_par_len = yaml_config['DATASET']['MAX_PART_NUM'],
    #                     )

def split_doc(doc, tokenizer, max_seg_num = 64, max_seg_len = 32, max_par_len = None, period_char = '.'):
    """
        doc_seg_ids_list : (doc_parts, max_seg_num, max_seg_len)
        doc_seg_mask : (doc_parts, max_seg_num)
        doc_att_mask_list : (doc_parts, max_seg_num, max_seg_len)
    """

    def find_elements(input, target):
        return [i for i,x in enumerate(input) if x==target]

    tokenized_doc = tokenizer.tokenize(doc)

    cumul_sents = []
    start_ptr = 0
    end_ptr = 0
    while end_ptr < len(tokenized_doc):
        end_ptr += max_seg_len - 2
        period_indices = find_elements(tokenized_doc[start_ptr : end_ptr], period_char)

        if len(period_indices) != 0:
            last_period_index = period_indices[-1]
            end_ptr = start_ptr + last_period_index + 1

        cumul_sents.append(tokenized_doc[start_ptr : end_ptr])
        start_ptr = end_ptr

    seg_ids_list = []
    seg_mask = []
    att_mask_list = []
    sent_lens = []
    for cur_concat in cumul_sents:
        sent_lens.append(len(cur_concat))
        assert len(cur_concat) <= max_seg_len - 2
        pad_len = max_seg_len - 2 - len(cur_concat)
        # new_seg =   [tokenizer.cls_token] + \
        #             cur_concat + \
        #             [tokenizer.sep_token]
        # new_att_mask = [1] * (max_seg_len - pad_len)
        new_seg =   [tokenizer.cls_token] + \
                    cur_concat + \
                    [tokenizer.sep_token] + \
                    [tokenizer.pad_token] * pad_len
        new_att_mask = [1] * (max_seg_len - pad_len) + [0] * pad_len

        new_seg_ids = tokenizer.convert_tokens_to_ids(new_seg)

        seg_ids_list.append(new_seg_ids)
        seg_mask.append(1)
        att_mask_list.append(new_att_mask)

    doc_seg_ids_list = []
    doc_seg_mask = []
    doc_att_mask_list = []
    while True:
        cur_seg_ids_list = seg_ids_list[:max_seg_num]
        cur_seg_mask = seg_mask[:max_seg_num]
        cur_att_mask_list = att_mask_list[:max_seg_num]

        seg_pad_len = (max_seg_num - len(cur_seg_ids_list))
        cur_seg_ids_list += [[tokenizer.pad_token_id] * max_seg_len] * seg_pad_len
        cur_seg_mask += [0] * seg_pad_len
        cur_att_mask_list += [[0] * max_seg_len] * seg_pad_len

        doc_seg_ids_list.append(cur_seg_ids_list)
        doc_seg_mask.append(cur_seg_mask)
        doc_att_mask_list.append(cur_att_mask_list)

        seg_ids_list = seg_ids_list[max_seg_num:]
        seg_mask = seg_mask[max_seg_num:]
        att_mask_list = att_mask_list[max_seg_num:]

        if len(seg_ids_list) == 0:
            break

    # print(np.array(doc_seg_ids_list).shape)
    # print(np.array(doc_att_mask_list).sum(axis = -1))
    # if np.sum(np.array(doc_att_mask_list).sum(axis = -1) == 0) > 0:
    #     print("---------------------------")
    #     print(len(doc), "doc:", doc[:30])
    #     # print("cumul_sents:", cumul_sents)

    # print(doc_seg_ids_list, doc_seg_mask, doc_att_mask_list)

    if max_par_len is not None:
        doc_seg_ids_list, doc_seg_mask, doc_att_mask_list = doc_seg_ids_list[:max_par_len], doc_seg_mask[:max_par_len], doc_att_mask_list[:max_par_len]

    return doc_seg_ids_list, doc_seg_mask, doc_att_mask_list

def read_wiki_doc(doc_name, yaml_config):
    # [WORD-dataset]
    # encoding="utf8" is need for running on Windows
    file_path = os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'],
                             yaml_config['DATASET']['WORD']['WIKI_DOC_DIR'], 
                            doc_name + ".txt")
    with open(file_path, "r", encoding="utf8") as doc_file:
    # with open(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, doc_name + ".txt"), "r") as doc_file:
        doc = doc_file.read()
    return doc


# ---------------------- WORD ----------------------
def word_add_concept_ids(meta_data_list, yaml_config, tokenizer):
    concept_list = set()
    for meta in meta_data_list:
        concept_list.add(meta['doc_pair'][0])
        concept_list.add(meta['doc_pair'][1])
    concept_list = list(concept_list)
    concept_list.sort()

    concept2id = {}
    for id, concept in enumerate(concept_list):
        concept2id[concept] = id

    meta_data_list_w_ids = []
    id2concept = [None] * len(concept_list)
    for meta in meta_data_list:
        s_id = concept2id[meta['doc_pair'][0]]
        t_id = concept2id[meta['doc_pair'][1]]
        meta['id_pair'] = (s_id, t_id)
        meta_data_list_w_ids.append(meta)

        for i in range(2):
            id = meta['id_pair'][i]
            if id2concept[id] is None:
                id2concept[id] = Concept(id, meta['concept_pair'][i], meta['doc_pair'][i])

    for id in tqdm(range(len(id2concept)), desc = "Preprocess Concept"):
        assert id2concept[id] is not None
        id2concept[id].preprocess(yaml_config, tokenizer)

    return meta_data_list_w_ids, id2concept

def read_word_data(args, yaml_config, tokenizer, val_size = 0):
    """
    Meta data is a list of dict:
        cur_pair = {
            "id"            : index,
            "concept_pair"  : (row["concept 1"], row["concept 2"]),
            "doc_pair"      : (doc_name1, doc_name2),
            "id_pair"       : (concept_id1, concept_id2), # Added by add_concept_ids
            "score"         : row['score'],
            "split"         : row['Train/Test'],
        }
    """
    if not args.overwrite_dataset_cache and os.path.isfile(args.dataset_cache_path):
        with open(args.dataset_cache_path, "rb") as ds_cache:
            meta_data_list, id2concept = pickle.load(ds_cache)
        logger.info("Loaded cached dataset from %s", args.dataset_cache_path)

    else:
        if args.overwrite_dataset_cache:
            logger.info("Overwrite dataset cache")
        if not os.path.isfile(args.dataset_cache_path):
            logger.info("Cannot find cached dataset file, generate cache from scratch")

        with open(os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['META_DATA_NAME']), "r") as meta_file:
            meta_data_list = json.load(meta_file)

        meta_data_list, id2concept = word_add_concept_ids(meta_data_list, yaml_config, tokenizer)

        with open(args.dataset_cache_path, "wb") as ds_cache:
            pickle.dump([meta_data_list, id2concept], ds_cache)

        logger.info("Cached dataset to %s", args.dataset_cache_path)

    if args.new_split:

        if args.random_seed is not None:
            logger.info("New split generated with random seed = %d", args.random_seed)
        else:
            logger.info("New split generated randomly")
        train_meta, test_meta = official_train_test_split(meta_data_list)
        if val_size > 0:
            train_meta, val_meta = train_test_split(train_meta,
                                                    test_size=val_size,
                                                    random_state=args.random_seed)
        else:
            val_meta = None

        with open(args.split_file, "w") as split_file:
            json.dump([train_meta, val_meta, test_meta], split_file)
        logger.info("Stored new split in file: %s", args.split_file)
    else:
        with open(args.split_file, "r") as split_file:
            train_meta, val_meta, test_meta = json.load(split_file)
        logger.info("Loaded split from file: %s", args.split_file)
    return train_meta, val_meta, test_meta, id2concept

# ---------------------- CNS ----------------------
def get_table_meta_data(table):
    # [CNS-dataset]
    # label|doc_id1|doc_id2|title1|title2|content1|content2|keywords1|keywords2|main_keywords1|main_keywords2|ner_keywords1|ner_keywords2|ner1|ner2|category1|category2|time1|time2

    meta_data_list = []
    all_doc_list = {}
    all_doc_count = {}
    for index, row in table.iterrows():
        # print("row.index", row.index)
        cur_pair = {
            "id"            : index,
            "concept_pair"  : (row["title1"], row["title2"]),
            "doc_pair"      : (row["doc_id1"], row["doc_id2"]),
            "score"         : row['label'],
            # "split"         : row['Train/Test'],
        }

        if type(row["content1"]) is str and type(row["content2"]) is str:
            meta_data_list.append(cur_pair)
        else:
            print(">>> row with index", index, "cannot be loaded")

        if type(row["content1"]) is str:
            if row["doc_id1"] not in all_doc_list:
                all_doc_list[row["doc_id1"]] = {
                                                    "title" : row["title1"],
                                                    "content" : row["content1"].replace(" ", ""),
                                                }
                all_doc_count[row["doc_id1"]] = 0
            else:
                all_doc_count[row["doc_id1"]] += 1

        if type(row["content2"]) is str:
            if row["doc_id2"] not in all_doc_list:
                all_doc_list[row["doc_id2"]] = {
                                                    "title" : row["title2"],
                                                    "content" : row["content2"].replace(" ", ""),
                                                }
                all_doc_count[row["doc_id2"]] = 0
            else:
                all_doc_count[row["doc_id2"]] += 1

    sorted_count = list(all_doc_count.values())
    sorted_count.sort()
    # print(sorted_count)

    return meta_data_list, all_doc_list

def cns_add_concept_ids(meta_data_list, all_doc_list, yaml_config, tokenizer):
    concept_list = set()
    for meta in meta_data_list:
        concept_list.add(meta['doc_pair'][0])
        concept_list.add(meta['doc_pair'][1])
    concept_list = list(concept_list)
    concept_list.sort()

    concept2id = {}
    for id, concept in enumerate(concept_list):
        concept2id[concept] = id

    meta_data_list_w_ids = []
    id2concept = [None] * len(concept_list)
    for meta in meta_data_list:
        s_id = concept2id[meta['doc_pair'][0]]
        t_id = concept2id[meta['doc_pair'][1]]
        meta['id_pair'] = (s_id, t_id)
        meta_data_list_w_ids.append(meta)

        for i in range(2):
            id = meta['id_pair'][i]
            if id2concept[id] is None:
                id2concept[id] = Concept(id, meta['concept_pair'][i], meta['doc_pair'][i])

    for id in tqdm(range(len(id2concept)), desc = "Preprocess Concept"):
        assert id2concept[id] is not None
        id2concept[id].preprocess(yaml_config, tokenizer, cns_doc_list = all_doc_list)

    return meta_data_list_w_ids, id2concept

def match_split(args, yaml_config, tokenizer):
    def transport_meta(index2meta, indices):
        result_meta = []
        for id in indices:
            if id in index2meta:
                result_meta.append(index2meta[id])
            else:
                print("Not in index2meta:", id)
        return result_meta

    with open(args.split_file, "r") as split_file:
        train_indices, val_indices, test_indices = json.load(split_file)

    # print(train_meta, val_meta, test_meta)

    cns_path = os.path.join(yaml_config['DATASET']['CNS']['CNS_ROOT_DIR'], yaml_config['DATASET']['CNS']['FILE_NAME'][args.dataset])
    cns_table = pd.read_csv(cns_path, delimiter = "|")

    meta_data_list, all_doc_list = get_table_meta_data(cns_table)
    meta_data_list, id2concept = cns_add_concept_ids(meta_data_list, all_doc_list, yaml_config, tokenizer)

    index2meta = {}
    for meta in meta_data_list:
        index2meta[meta['id']] = meta

    train_meta = transport_meta(index2meta, train_indices)
    val_meta = transport_meta(index2meta, val_indices)
    test_meta = transport_meta(index2meta, test_indices)

    with open(args.split_file, "w") as split_file:
        json.dump([train_meta, val_meta, test_meta], split_file)

def read_cns_data(args, yaml_config, tokenizer):
    """
        cur_pair = {
            "id"            : index,
            "concept_pair"  : (row["title1"], row["title2"]),
            "doc_pair"      : (row["doc_id1"], row["doc_id2"]),
            "score"         : row['label'],
            # "split"         : row['Train/Test'],
        }
    """

    if not args.overwrite_dataset_cache and os.path.isfile(args.dataset_cache_path):
        with open(args.dataset_cache_path, "rb") as ds_cache:
            meta_data_list, id2concept = pickle.load(ds_cache)
        logger.info("Loaded cached dataset from %s", args.dataset_cache_path)

    else:
        if args.overwrite_dataset_cache:
            logger.info("Overwrite dataset cache")
        else:
            if not os.path.isfile(args.dataset_cache_path):
                logger.info("Cannot find cached dataset file " + args.dataset_cache_path + ", generate cache from scratch")

        cns_path = os.path.join(yaml_config['DATASET']['CNS']['CNS_ROOT_DIR'], yaml_config['DATASET']['CNS']['FILE_NAME'][args.dataset])
        cns_table = pd.read_csv(cns_path, delimiter = "|")
        meta_data_list, all_doc_list = get_table_meta_data(cns_table)
        meta_data_list, id2concept = cns_add_concept_ids(meta_data_list, all_doc_list, yaml_config, tokenizer)

        with open(args.dataset_cache_path, "wb") as ds_cache:
            pickle.dump([meta_data_list, id2concept], ds_cache)

        logger.info("Cached dataset to %s", args.dataset_cache_path)


    if args.new_split:

        if args.random_seed is not None:
            logger.info("New split generated with random seed = %d", args.random_seed)
        else:
            logger.info("New split generated randomly")
        train_meta, val_meta, test_meta = train_val_test_split(meta_data_list,
                                                            split_sizes=[0.7, 0.2, 0.1],
                                                            random_seed = args.random_seed)
        with open(args.split_file, "w") as split_file:
            json.dump([train_meta, val_meta, test_meta], split_file)
        logger.info("Stored new split in file: %s", args.split_file)
    else:
        with open(args.split_file, "r") as split_file:
            train_meta, val_meta, test_meta = json.load(split_file)
        logger.info("Loaded split from file: %s", args.split_file)

    return train_meta, val_meta, test_meta, id2concept

def pad_sequence(sequence, target_len, pad_id):
    padding = [pad_id,] * (target_len - len(sequence))
    sequence += padding
    return sequence

# def create_dataset(doc_s_list, doc_t_list, scores, tokenizer, 
#                                             max_doc_len = None, 
#                                             score_threshold = None):
#     assert len(doc_s_list) == len(doc_t_list)
#     input_ids = []
#     labels = []
#     seg_num = []
#     for i in tqdm(range(len(doc_s_list)), desc = "Create Dataset"):
#         doc_s = doc_s_list[i]
#         doc_t = doc_t_list[i]

#         seg_list_s = split_doc(doc_s, tokenizer)
#         seg_list_t = split_doc(doc_s, tokenizer)
#         seg_num.append(len(seg_list_s))
#         seg_num.append(len(seg_list_t))

#         tokenized_s = tokenizer.tokenize(doc_s_list[i])
#         tokenized_t = tokenizer.tokenize(doc_t_list[i])

#         bert_max_len = 512
#         if len(tokenized_s) <= max_doc_len and len(tokenized_t) > max_doc_len:
#             target_len_s = len(tokenized_s)
#             target_len_t = bert_max_len - len(tokenized_s) - 2
#             tokenized_t = tokenized_t[:target_len_t]
#         if len(tokenized_s) > max_doc_len and len(tokenized_t) <= max_doc_len:
#             target_len_s = bert_max_len - len(tokenized_t) - 2
#             target_len_t = len(tokenized_t)
#             tokenized_s = tokenized_s[:target_len_s]
#         if len(tokenized_s) > max_doc_len and len(tokenized_t) > max_doc_len:
#             target_len_s = target_len_t = max_doc_len
#             tokenized_s = tokenized_s[:target_len_s]
#             tokenized_t = tokenized_t[:target_len_t]

#         cls_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
#         sep_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
#         pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])

#         ids_s  = tokenizer.convert_tokens_to_ids(tokenized_s)
#         ids_s = pad_sequence(ids_s, target_len_s, pad_id)
#         ids_t  = tokenizer.convert_tokens_to_ids(tokenized_t)
#         ids_t = pad_sequence(ids_t, target_len_t, pad_id)
#         ids_cls_s_sep_t = cls_id + ids_s + sep_id + ids_t
#         input_ids.append(ids_cls_s_sep_t)

#         label = (scores[i] > score_threshold) * 1
#         labels.append(label)

#     for per in range(0, 100, 5):
#         print(str(per) + "th percentile of trans_score : ", np.percentile(seg_num, per))


#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)
#     dataset = torch.utils.data.TensorDataset(input_ids, labels)
#     return dataset


def load_dataset(meta_data_list, id2concept, 
                        tokenizer,
                        debug_size = None,
                        # truncate_at = None,
                        yaml_config = None):
    # [WORD-dataset]

    doc_s_list, doc_t_list, scores = [], [], []
    for index, cur_pair in enumerate(meta_data_list):
        # wiki_doc1 = read_wiki_doc(cur_pair["doc_pair"][0], yaml_config)
        # wiki_doc2 = read_wiki_doc(cur_pair["doc_pair"][1], yaml_config)
        srs_concept = id2concept[cur_pair["id_pair"][0]]
        tgt_concept = id2concept[cur_pair["id_pair"][1]]

        if len(srs_concept.doc) > 0 and len(tgt_concept.doc) > 0:
            doc_s_list.append(srs_concept)
            doc_t_list.append(tgt_concept)
            scores.append(cur_pair['score'])

    if debug_size is not None:
        assert debug_size > 0
        doc_s_list, doc_t_list, scores = doc_s_list[:debug_size], doc_t_list[:debug_size], scores[:debug_size]

    dataset = create_dataset(doc_s_list, doc_t_list, scores, 
                            # tokenizer = tokenizer, 
                            # max_seg_num = yaml_config['DATASET']['MAX_SEG_NUM'], 
                            # max_seg_len = yaml_config['DATASET']['MAX_SEG_LEN'],
                            score_threshold = yaml_config['DATASET']['SCORE_THRESHOLD'],
                            # mode = yaml_config['DATASET']['MODE']
                            )
    # dataset = create_dataset(doc_s_list, doc_t_list, scores, tokenizer, 
    #                                         max_doc_len = 255, 
    #                                         score_threshold = 0.0)

    return dataset

def create_dataset(doc_s_list, doc_t_list, scores, 
                                            # tokenizer, 
                                            # max_seg_num = 64, 
                                            # max_seg_len = 32,
                                            score_threshold = None,
                                            # mode = "segmentation"
                                            ):
    assert len(doc_s_list) == len(doc_t_list)
    srs_input_ids, srs_seg_masks, srs_att_masks = [], [], []
    tgt_input_ids, tgt_seg_masks, tgt_att_masks = [], [], []
    labels = []
    srs_ids, tgt_ids = [], []
    for i in range(len(doc_s_list)):
        concept_s = doc_s_list[i]
        concept_t = doc_t_list[i]

        if len(concept_s.doc) == 0 or len(concept_t.doc) == 0:
            continue

        # seg_ids_list_s, seg_mask_s, att_mask_s = split_doc(doc_s, tokenizer, max_seg_num = max_seg_num, max_seg_len = max_seg_len, mode = mode)

        # seg_ids_list_t, seg_mask_t, att_mask_t = split_doc(doc_t, tokenizer, max_seg_num = max_seg_num, max_seg_len = max_seg_len, mode = mode)
        seg_ids_list_s, seg_mask_s, att_mask_s = concept_s.doc_seg_ids_list, concept_s.doc_seg_mask, concept_s.doc_att_mask_list
        seg_ids_list_t, seg_mask_t, att_mask_t = concept_t.doc_seg_ids_list, concept_t.doc_seg_mask, concept_t.doc_att_mask_list

        if len(seg_ids_list_s) < len(seg_ids_list_t):
            len_diff = len(seg_ids_list_t) - len(seg_ids_list_s)
            seg_ids_list_s += seg_ids_list_s[:1] * len_diff
            seg_mask_s += seg_mask_s[:1] * len_diff
            att_mask_s += att_mask_s[:1] * len_diff
        elif len(seg_ids_list_s) > len(seg_ids_list_t):
            len_diff = len(seg_ids_list_s) - len(seg_ids_list_t)
            seg_ids_list_t += seg_ids_list_t[:1] * len_diff
            seg_mask_t += seg_mask_t[:1] * len_diff
            att_mask_t += att_mask_t[:1] * len_diff

        label = (scores[i] > score_threshold) * 1 # for classification
        # label = (scores[i] > score_threshold) * 2. - 1 # for cos sim

        assert len(seg_ids_list_s) == len(seg_ids_list_t), str(len(seg_ids_list_s)) + ", " + str(len(seg_ids_list_t))

        srs_input_ids += seg_ids_list_s
        srs_seg_masks += seg_mask_s
        srs_att_masks += att_mask_s
        tgt_input_ids += seg_ids_list_t
        tgt_seg_masks += seg_mask_t
        tgt_att_masks += att_mask_t
        labels += [label] * len(seg_ids_list_s)
        srs_ids += [concept_s.id] * len(seg_ids_list_s)
        tgt_ids += [concept_t.id] * len(seg_ids_list_t)

        # srs_input_ids += seg_ids_list_s[:1]
        # srs_seg_masks += seg_mask_s[:1]
        # srs_att_masks += att_mask_s[:1]
        # tgt_input_ids += seg_ids_list_t[:1]
        # tgt_seg_masks += seg_mask_t[:1]
        # tgt_att_masks += att_mask_t[:1]
        # label = (scores[i] > score_threshold) * 1
        # labels += [label]
        # srs_ids += [concept_s.id]
        # tgt_ids += [concept_t.id]


    srs_input_ids = torch.tensor(srs_input_ids, dtype = torch.long)
    srs_seg_masks = torch.tensor(srs_seg_masks, dtype = torch.long)
    srs_att_masks = torch.tensor(srs_att_masks, dtype = torch.long)
    tgt_input_ids = torch.tensor(tgt_input_ids, dtype = torch.long)
    tgt_seg_masks = torch.tensor(tgt_seg_masks, dtype = torch.long)
    tgt_att_masks = torch.tensor(tgt_att_masks, dtype = torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    srs_ids = torch.tensor(srs_ids, dtype = torch.long)
    tgt_ids = torch.tensor(tgt_ids, dtype = torch.long)
    dataset = torch.utils.data.TensorDataset(srs_input_ids, srs_seg_masks, srs_att_masks, srs_ids, tgt_input_ids, tgt_seg_masks, tgt_att_masks, tgt_ids, labels)

    # ds_size, seg_num, seg_len = srs_input_ids.shape
    # input_ids = torch.zeros(ds_size, seg_num, seg_len * 2)
    # attention_masks = torch.zeros(ds_size, seg_num, seg_len * 2)
    # seg_ids = torch.zeros(ds_size, seg_num, seg_len * 2)
    # for i in range(ds_size):
    #     for j in range(seg_num):
    #         srs_input_len = srs_att_masks[i, j, :].sum()
    #         tgt_input_len = tgt_att_masks[i, j, :].sum()
    #         input_ids[i, j, :srs_input_len] = srs_input_ids[i, j, :srs_input_len]
    #         input_ids[i, j, srs_input_len : srs_input_len+tgt_input_len] = tgt_input_ids[i, j, :tgt_input_len]
    #         attention_masks[i, j, : srs_input_len+tgt_input_len] = 1
    #         seg_ids[i, j, srs_input_len : srs_input_len+tgt_input_len] = 1
    # input_ids = torch.tensor(input_ids, dtype = torch.long)
    # attention_masks = torch.tensor(attention_masks, dtype = torch.long)
    # seg_ids = torch.tensor(seg_ids, dtype = torch.long)
    # dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, seg_ids, srs_seg_masks, srs_ids, tgt_seg_masks, tgt_ids, labels)

    return dataset

# def create_dataset(doc_s_list, doc_t_list, scores, tokenizer, 
#                                             max_doc_len = None, 
#                                             score_threshold = None):
#     assert len(doc_s_list) == len(doc_t_list)
#     input_ids = []
#     att_masks = []
#     labels = []
#     for i in tqdm(range(len(doc_s_list)), desc = "Create Dataset"):
#         srs_doc_ids = doc_s_list[i].doc_seg_ids_list[0][0]
#         tgt_doc_ids = doc_t_list[i].doc_seg_ids_list[0][0]

#         print(len(srs_doc_ids), len(tgt_doc_ids))

#         bert_max_len = 512
#         if len(srs_doc_ids) <= max_doc_len and len(tgt_doc_ids) > max_doc_len:
#             target_len_s = len(srs_doc_ids)
#             target_len_t = bert_max_len - len(srs_doc_ids) - 2
#             tgt_doc_ids = tgt_doc_ids[:target_len_t]
#         if len(srs_doc_ids) > max_doc_len and len(tgt_doc_ids) <= max_doc_len:
#             target_len_s = bert_max_len - len(tgt_doc_ids) - 2
#             target_len_t = len(tgt_doc_ids)
#             srs_doc_ids = srs_doc_ids[:target_len_s]
#         if len(srs_doc_ids) > max_doc_len and len(tgt_doc_ids) > max_doc_len:
#             target_len_s = target_len_t = max_doc_len
#             srs_doc_ids = srs_doc_ids[:target_len_s]
#             tgt_doc_ids = tgt_doc_ids[:target_len_t]

#         ids_cls_s_sep_t = [tokenizer.cls_token_id] + srs_doc_ids + [tokenizer.sep_token_id] + tgt_doc_ids
#         cur_input_ids = pad_sequence(ids_cls_s_sep_t, bert_max_len, tokenizer.pad_token_id)
#         attention_mask = [1] * len(ids_cls_s_sep_t) + [0] * (bert_max_len - len(ids_cls_s_sep_t))

#         # max_doc_len = 256
#         # srs_doc_ids = pad_sequence(srs_doc_ids[:max_doc_len], max_doc_len, tokenizer.pad_token_id)
#         # tgt_doc_ids = pad_sequence(tgt_doc_ids[:max_doc_len], max_doc_len, tokenizer.pad_token_id)
#         # cur_input_ids = srs_doc_ids + tgt_doc_ids
#         # attention_mask = [1] * len(cur_input_ids) + [0] * (bert_max_len - len(cur_input_ids))

#         label = (scores[i] > score_threshold) * 1

#         # if len(ids_cls_s_sep_t) < bert_max_len:
#         # print('------------------------')
#         # print(ids_cls_s_sep_t)
#         # print(cur_input_ids)
#         # print(tokenizer.convert_tokens_to_ids(ids_cls_s_sep_t))

#         input_ids.append(cur_input_ids)
#         att_masks.append(attention_mask)
#         labels.append(label)

#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     att_masks = torch.tensor(att_masks, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)
#     dataset = torch.utils.data.TensorDataset(input_ids, att_masks, labels)

#     return dataset


def get_document_statistics(doc, tokenizer, max_seg_num = 128, max_seg_len = 32):
    sent_list = doc.split('.')
    sent_lens = []
    for sent in sent_list:
        tokenized_sent = tokenizer.tokenize(sent)
        sent_lens.append(len(tokenized_sent))

    return len(sent_list), sent_lens

def get_dataset_statistics(doc_s_list, doc_t_list, scores, tokenizer, 
                                            max_seg_num = 128, 
                                            max_seg_len = 32,
                                            score_threshold = None):
    assert len(doc_s_list) == len(doc_t_list)
    seg_num_list = []
    sent_len_list = []
    for i in range(len(doc_s_list)):
        doc_s = doc_s_list[i]
        doc_t = doc_t_list[i]

        sent_num, sent_lens = get_document_statistics(doc_s, tokenizer, max_seg_num = max_seg_num, max_seg_len = max_seg_len)
        seg_num_list.append(sent_num)
        sent_len_list += sent_lens

        sent_num, sent_lens = get_document_statistics(doc_t, tokenizer, max_seg_num = max_seg_num, max_seg_len = max_seg_len)
        seg_num_list.append(sent_num)
        sent_len_list += sent_lens

    sent_len_list = np.array(sent_len_list)
    print("=========== sentence lengths ===========")
    print("Mean", np.mean(sent_len_list))
    for per in range(0, 101, 5):
        print(str(per) + "th percentile : ", np.percentile(sent_len_list, per))


    print("=========== the number of sentences in a document ===========")
    print("Mean", np.mean(seg_num_list))
    for per in range(0, 101, 5):
        print(str(per) + "th percentile : ", np.percentile(seg_num_list, per))


# def load_cns_doc_pairs(meta_data_list, doc_list, 
#                         debug_size = None,
#                         # truncate_at = None
#                         ):
#     # [CNS-dataset]

#     # if truncate_at is not None:
#     #     print("-------------- Truncation at --------------")
#     doc_s_list, doc_t_list, scores = [], [], []
#     for index, cur_pair in enumerate(tqdm(meta_data_list)):
#         try:
#             cns_doc1 = doc_list[cur_pair["doc_pair"][0]]
#             cns_doc2 = doc_list[cur_pair["doc_pair"][0]]

#             doc_s_list.append(cns_doc1['title'] + "\n" + cns_doc1['content'])
#             doc_t_list.append(cns_doc2['title'] + "\n" + cns_doc2['content'])
#             scores.append(cur_pair['score'])

#         except KeyboardInterrupt:
#             exit()
#         except Exception as e:
#             print(repr(e))
#             continue

#     # if truncate_at is not None:
#     #     print("Truncation data augmentation: from {} raw doc pairs, generated {} truncated doc pairs".format( len(meta_data_list), len(scores) ))

#     if debug_size is None:
#         return doc_s_list, doc_t_list, scores
#     else:
#         assert debug_size > 0
#         return doc_s_list[:debug_size], doc_t_list[:debug_size], scores[:debug_size]

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
    # [WORD-dataset]
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

# ----------------------- Data Augmentation -----------------------
def print_path(G, path):
    """
    Unix ---[0.90]--> Open-source_software ---[0.70]--> Distributed_computing ---[1.00]--> Internet ---[0.60]--> Pornography ---[1.00]--> Pornographic_film_actor ---[1.00]--> Actor ---[0.78]--> Reality_television ---[0.50]--> Competition ---[0.40]--> Miss_America_protest ---[0.40]--> Social_movement ---[0.60]--> Socialism ---[0.60]--> Left-libertarianism
    """
    path_string = ""
    for i in range(len(path) - 1):
        node_i = path[i]
        node_j = path[i+1]
        path_string += f"{node_i} ---[{G[node_i][node_j]['weight']:.2f}]--> "
    path_string += path[-1]
    return path_string

def get_edge_scores_for_path(G, path):
    edge_scores = []
    for i in range(len(path) - 1):
        node_i = path[i]
        node_j = path[i+1]
        score = G[node_i][node_j]['score']
        edge_scores.append(score)
    return edge_scores

def all_pair_indices(size, shuffle = False):
    pair_indices = []
    for i in range(size-1):
        for j in range(i+1, size):
            pair_indices.append([i,j])
    if shuffle:
        random.shuffle(pair_indices)
    return pair_indices

def get_concrete_graph(meta_list, score_threshold):
    concrete_graph = nx.Graph()
    for meta in meta_list:
        srs_id, tgt_id = meta['id_pair'][0], meta['id_pair'][1]
        # if meta['score'] >= score_threshold:
        if meta['score'] > score_threshold:
            # edge_weight = edge_weight_dict[edge_weight_mapping]
            concrete_graph.add_edge(srs_id, tgt_id, weight=1., score=meta['score'])
        else:
            concrete_graph.add_node(srs_id)
            concrete_graph.add_node(tgt_id)

    return concrete_graph

def get_connected_components(concrete_graph, id2concept):
    all_connected_components = sorted(nx.connected_components(concrete_graph), key=len, reverse=False)
    # print(all_connected_components)
    cc_sizes = [len(cc) for cc in all_connected_components]
    # print("Graph component sizes", cc_sizes)
    cc_sizes = np.array(cc_sizes)

    concept2cluster = [-1] * len(id2concept)
    for cc_id, cc in enumerate(all_connected_components):
        for concept_id in cc:
            concept2cluster[concept_id] = cc_id
    return all_connected_components, concept2cluster

def graph_augment_dataset(meta_list, concrete_graph, 
                            score_threshold = 0.0,
                            args = None, 
                            ):
    """
    Augment the dataset according to the following two properties of relatedness:

        1. Relatedness is transitive (a ~ b & b ~ c ==> a ~ c)
        2. Relatedness is commutative (a ~ b ==> b ~ a)

    The method is to cluster (strongly) related docs, and
    sample from the clusters related and unrelated doc pairs
    """
    print("-------------- Clustering data augmentation --------------")

    related_pairs, related_scores = [], []
    unrelated_pairs, unrelated_scores = [], []
    original_related_count = original_unrelated_count = 0
    # comm_aug_rel_count = comm_aug_unrel_count = 0
    for meta in meta_list:
        doc_s, doc_t = meta['id_pair'][0], meta['id_pair'][1]
        if meta['score'] > score_threshold:
            related_pairs.append((doc_s, doc_t))
            related_scores.append(meta['score'])
            original_related_count += 1
            # if use_commutativity:
            #     related_pairs.append((doc_t, doc_s))
            #     related_scores.append(meta['score'])
            #     comm_aug_rel_count += 1
        else:
            unrelated_pairs.append((doc_s, doc_t))
            unrelated_scores.append(meta['score'])
            original_unrelated_count += 1
            # if use_commutativity:
            #     unrelated_pairs.append((doc_t, doc_s))
            #     unrelated_scores.append(meta['score'])
            #     comm_aug_unrel_count += 1

    # if use_transitivity:

    # concrete_graph = nx.Graph()
    # for meta in meta_list:
    #     doc_s, doc_t = meta['id_pair'][0], meta['id_pair'][1]
    #     # if meta['score'] >= score_threshold:
    #     if meta['score'] > score_threshold:
    #         edge_weight_dict = {
    #             "linear" : 1. - meta['score'],
    #             "recip"  : 1. / meta['score'] - 1,
    #             "poly"   : (1. - meta['score']) ** 2,
    #         }
    #         # edge_weight = 1. - meta['score']
    #         # edge_weight = 1. / meta['score'] - 1
    #         # edge_weight = (1. - meta['score']) ** 2
    #         edge_weight = edge_weight_dict[edge_weight_mapping]
    #         concrete_graph.add_edge(doc_s, doc_t, weight=edge_weight, score=meta['score'])
    #     else:
    #         concrete_graph.add_node(doc_s)
    #         concrete_graph.add_node(doc_t)
    # concrete_graph = get_concrete_graph(meta_list, score_threshold)

    if args.aug_ratio is not None and args.aug_ratio > 1.0:
        original_size = original_related_count + original_unrelated_count
        target_size = int(original_size * args.aug_ratio)
        aug_related_num = max(int(target_size / 2) - original_related_count, 0)
        # aug_unrelated_num = max(int(target_size / 2) - original_unrelated_count, 0)
        # aug_size = aug_related_num + aug_unrelated_num
    else:
        aug_related_num = None

    cluster_paths = dict(nx.all_pairs_dijkstra_path(concrete_graph))

    all_neighbor_node_pairs = set()
    for node in concrete_graph.nodes:
        # filter_length_min, filter_length_max = filter_length_range
        # if args > 1:
        #     K = filter_length_max
        # else:
        #     K = None
        k_hop_neighbors = nx.single_source_shortest_path_length(concrete_graph, node, cutoff=args.k_hops)
        for neighbor in k_hop_neighbors:
            if neighbor != node:
                new_pair = (node, neighbor)
                all_neighbor_node_pairs.add(new_pair)
    all_neighbor_node_pairs = list(all_neighbor_node_pairs)
    random.shuffle(all_neighbor_node_pairs)

    trans_scores_list = []
    trans_aug_rel_count = trans_aug_unrel_count = 0
    neighbor_pair_num = len(all_neighbor_node_pairs)
    overall_progress_bar = tqdm(all_neighbor_node_pairs, desc = "Filter")
    for node_i, node_j in overall_progress_bar:
        if args.score_threshold is None and args.k_hops is None:
            trans_score = 1.0
            trans_scores_list.append(trans_score)
        else:
            sp = cluster_paths[node_i][node_j]
            edge_scores = get_edge_scores_for_path(concrete_graph, sp)

            if args.score_estimator == 'mean':
                trans_score = np.mean(edge_scores)
            elif args.score_estimator == 'min':
                trans_score = np.min(edge_scores)
            elif args.score_estimator == 'prod':
                trans_score = np.prod(edge_scores)
            else:
                raise NotImplementedError

            trans_scores_list.append(trans_score)

        if args.score_threshold is not None and trans_score < args.score_threshold:
            continue

        if (node_i, node_j) not in related_pairs:
            related_pairs.append((node_i, node_j))
            related_scores.append(trans_score)
            trans_aug_rel_count += 1
            # sample_attempt_history[-1] = 1

            if args.aug_ratio is not None and trans_aug_rel_count >= aug_related_num:
                break

    N = concrete_graph.number_of_nodes()
    nodes = list(concrete_graph.nodes)
    sample_attempt_history = []
    aug_unrelated_num = int((len(related_pairs) - len(unrelated_pairs)))

    print('>>> Unrelated Trans Pairs Sampling, target number:', aug_unrelated_num)
    overall_progress_bar = tqdm(range(aug_unrelated_num))
    while len(unrelated_pairs) < len(related_pairs):
        i, j = random.sample(range(N), 2)
        node_i, node_j = nodes[i], nodes[j]
        if len(sample_attempt_history) > 0:
            sample_success_rate = np.sum(sample_attempt_history) / len(sample_attempt_history)
            overall_progress_bar.set_description(
                (
                    f'Attempts: {len(sample_attempt_history)}; '
                    f'Success: {trans_aug_unrel_count}; '
                    f'Rate: {sample_success_rate*100:.3f}% '
                )
            )
        sample_attempt_history.append(0)

        if node_i in cluster_paths and node_j not in cluster_paths[node_i]:
        #     if trans_aug_unrel_count < aug_unrelated_num:
            if (node_i, node_j) not in unrelated_pairs: # and (node_j, node_i) not in unrelated_pairs:
                trans_score = 0.0
                unrelated_pairs.append((node_i, node_j))
                unrelated_scores.append(trans_score)
                trans_aug_unrel_count += 1
                sample_attempt_history[-1] = 1
                overall_progress_bar.update(1)


    for per in range(0, 100, 5):
        print(str(per) + "th percentile of trans_score : ", np.percentile(trans_scores_list, per))

    augmented_meta_list = []
    for idx, id_pair in enumerate(related_pairs):
        cur_pair = {
            "id_pair" : id_pair,
            "score"    : related_scores[idx],
        }
        augmented_meta_list.append(cur_pair)

    for idx, id_pair in enumerate(unrelated_pairs):
        cur_pair = {
            "id_pair" : id_pair,
            "score"    : unrelated_scores[idx],
        }
        augmented_meta_list.append(cur_pair)

    # if use_commutativity:
    #     print("Augmented pairs using commutativity:", comm_aug_rel_count + comm_aug_unrel_count)
    #     print("\t related pairs   :", comm_aug_rel_count)
    #     print("\t unrelated pairs :", comm_aug_unrel_count)

    # if use_transitivity:
    print("Augmented pairs using transitivity:", trans_aug_rel_count + trans_aug_unrel_count)
    # print("\t Number of clusters found:", cluster_count)
    print("\t Augmentation Ratio:", args.aug_ratio)
    print("\t Filtering threshold:", args.score_threshold)
    print("\t related pairs   : {} / {}".format(trans_aug_rel_count, neighbor_pair_num))
    print("\t unrelated pairs : {} / {}".format(trans_aug_unrel_count, neighbor_pair_num))
    print("\t Realized aug ratio : {:.2f}%".format( (trans_aug_rel_count + trans_aug_unrel_count)*100./ (original_related_count + original_unrelated_count)) )
    # print("\t Sampling success rate: {:.2f}%".format(sample_success_rate * 100.))

    print("Merged into augmented_meta_list, from old size to new size: {} --[aug]-> {}".format( len(meta_list), len(augmented_meta_list) ))
    print("\t Related:   {} --[aug]-> {}".format( original_related_count, len(related_pairs) ))
    print("\t Unrelated: {} --[aug]-> {}".format( original_unrelated_count, len(unrelated_pairs) ))

    return augmented_meta_list

