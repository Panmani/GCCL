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
import inspect
from statistics import mean
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Model dependencies ----------
import torch
# from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx

from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import DataLoader, Dataset, Data
# from torch_geometric.nn import Sequential, GCNConv, SAGPooling, global_max_pool
# from torch.nn import Module, Linear, ReLU, Sigmoid, Dropout


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.root_n_id_s = torch.tensor(0)
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.root_n_id_t = torch.tensor(0)

    def __inc__(self, key, *args):
        if key == 'edge_index_s':
            return self.x_s.shape[0]
        if key == 'edge_index_t':
            return self.x_t.shape[0]
        else:
            return super().__inc__(key, *args)

class ConceptPairDataset(Dataset):
    def __init__(self, meta_list, id2concept, yaml_config):
        super(ConceptPairDataset, self).__init__()

        self.dataset = []
        for idx, cur_pair in enumerate(meta_list):
            srs_concept = id2concept[cur_pair["id_pair"][0]]
            tgt_concept = id2concept[cur_pair["id_pair"][1]]

            if len(srs_concept.doc) > 0 and len(tgt_concept.doc) > 0:
                g_text1 = srs_concept.pyg_graph
                g_text2 = tgt_concept.pyg_graph
                if g_text1.x is not None and g_text2.x is not None:
                    pair = PairData(g_text1.edge_index, g_text1.x, g_text2.edge_index, g_text2.x)

                    label = (cur_pair['score'] > yaml_config['DATASET']['SCORE_THRESHOLD']) * 1
                    self.dataset.append([pair, label])

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

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

    def preprocess(self, yaml_config, tokenizer, cns_doc_list = None, encoder_model = None):
        self.doc = ""
        if cns_doc_list is None:
            try:
                self.doc = read_wiki_doc(self.doc_name, yaml_config)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(repr(e))

            dataset_type = 'WORD'
        else:
            # For CNS, self.doc_name is the document id given by cns raw dataset file
            cns_doc = cns_doc_list[self.doc_name]
            self.doc = cns_doc['title'] + "\n" + cns_doc['content']
            dataset_type = "CNS"

        self.pyg_graph = build_graph(  self.doc,
                            tokenizer,
                            max_seg_len = yaml_config['DATASET']['MAX_SEG_LEN'],
                            encoder_model = encoder_model,
                            dataset_type = dataset_type,
                        )

def build_graph(doc, tokenizer, max_seg_len = 32, encoder_model = None, dataset_type = None):
    tokenized_doc = tokenizer.tokenize(doc)[:max_seg_len]
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_doc), dtype = torch.long)

    if dataset_type == "WORD":
        input_embeds = encoder_model.embeddings.word_embeddings(input_ids)
    elif dataset_type == "CNS":
        input_embeds = encoder_model.get_input_embeddings()(input_ids)
    else:
        raise NotImplementedError

    num_token = input_embeds.shape[0]
    adj = np.ones((num_token, num_token))

    G = nx.from_numpy_matrix(adj)
    pyg_graph = from_networkx(G)
    pyg_graph.x = torch.tensor(input_embeds.clone().detach())
    
    return pyg_graph

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
def word_add_concept_ids(meta_data_list, yaml_config, tokenizer, encoder_model = None):
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
        id2concept[id].preprocess(yaml_config, tokenizer, encoder_model = encoder_model)

    return meta_data_list_w_ids, id2concept

def read_word_data(args, yaml_config, tokenizer, val_size = 0, debug_size = None, encoder_model = None):
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

        if debug_size is not None:
            meta_data_list = meta_data_list[:debug_size]

        meta_data_list, id2concept = word_add_concept_ids(meta_data_list, yaml_config, tokenizer, encoder_model = encoder_model)

        with open(args.dataset_cache_path, "wb") as ds_cache:
            pickle.dump([meta_data_list, id2concept], ds_cache)

        logger.info("Cached dataset to %s", args.dataset_cache_path)

    if args.new_split:

        if args.random_seed is not None:
            logger.info("New split generated with random seed = %d", args.random_seed)
        else:
            logger.info("New split generated randomly")
        train_meta, test_meta = official_train_test_split(meta_data_list)
        # train_meta, test_meta = train_test_split(meta_data_list,
        #                                         test_size=0.1,
        #                                         random_state=args.random_seed)
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

def cns_add_concept_ids(meta_data_list, all_doc_list, yaml_config, tokenizer, encoder_model = None):
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
        id2concept[id].preprocess(yaml_config, tokenizer, cns_doc_list = all_doc_list, encoder_model = encoder_model)

    return meta_data_list_w_ids, id2concept

def read_cns_data(args, yaml_config, tokenizer, encoder_model = None, debug_size = None):
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

        if debug_size is not None:
            meta_data_list = meta_data_list[:debug_size]

        meta_data_list, id2concept = cns_add_concept_ids(meta_data_list, all_doc_list, yaml_config, tokenizer, encoder_model = encoder_model)

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
