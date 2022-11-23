from __future__ import print_function, division

import argparse
from distutils.debug import DEBUG
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from utils import loadWord2Vec, clean_str
from math import log
# from sklearn import svm
# from nltk.corpus import wordnet as wn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.spatial.distance import cosine
from tqdm import tqdm
from collections import Counter
import itertools

# ---------- Basic dependencies ----------
import os
import argparse
import json
import yaml
import pickle
import copy
import builtins
from tqdm import tqdm
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Model dependencies ----------
import numpy as np
import torch
import torch.distributed as dist
# from transformers import AutoTokenizer
import torch
import torch.nn as nn
import logging
from transformers import (BertTokenizer, BertModel, BertConfig,
                            RobertaConfig, RobertaModel, RobertaTokenizer,
                            XLNetConfig, XLNetModel, XLNetTokenizer)
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import numpy as np

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


# ---------- Config ----------
# from configs.config import MAX_SEQ_LEN, SCORE_THRESHOLD
# from configs.word_config import *
# from models.doc_ctrs_transformer import CtrsTransformer, WORD_MODEL_CLASSES, ConcreteTransformer
from utils import SoftmaxLoss, train_model, load_checkpoint, evaluate_model, fix_random_seed
from data import ConceptPairDataset, read_word_data, read_cns_data
from models.Jacobi import buildModel
from models.GCNII import GCNII
from models.ShaDow import ShaDow

DEBUG_SIZE = None


WORD_MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
}

class CreGNN(torch.nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        if args.model == "shadow":
            self.gnn = ShaDow(config.hidden_size, config.hidden_size, config.hidden_size)
        elif args.model == "jacobi":
            self.gnn = buildModel(conv_layer = 10, aggr = "gcn", alpha = 1.0)
        elif args.model == "gcnii":
            self.gnn = GCNII(nfeat=config.hidden_size,
                            nlayers=64,
                            nhidden=config.hidden_size,
                            # nclass=int(labels.max()) + 1,
                            dropout=0.1, # Dropout rate (1 - keep probability).
                            lamda = 0.5, 
                            alpha=0.1,
                            variant=False)
        else:
            raise NotImplementedError

        self.cre_loss = SoftmaxLoss(
                                sentence_embedding_dimension = config.hidden_size,
                                num_labels = 2,
                                concatenation_sent_rep = True,
                                concatenation_sent_difference = True,
                                concatenation_sent_multiplication = False)

    def forward(self, x_s, edge_index_s, batch_s, root_n_id_s, x_t, edge_index_t, batch_t, root_n_id_t, labels):
        srs_out = self.gnn(x_s, edge_index_s, batch_s, root_n_id_s)
        tgt_out = self.gnn(x_t, edge_index_t, batch_t, root_n_id_t)
        loss_cre, cre_logits = self.cre_loss([srs_out, tgt_out], labels.unsqueeze(-1) * 1.)

        loss_details = {}
        loss_details["loss"] = loss_cre

        return loss_cre, loss_details, cre_logits

    def predict_labels(self, logits):
        pred_labels = (logits[:, 0] > 0) * 1.
        sigma = nn.Sigmoid()
        pred_probs = sigma(logits)

        return pred_labels, pred_probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Train model')
    parser.add_argument('--load_ckpt',
                    default = None,
                    type = str,
                    help = 'Filename of the model checkpoint that is to be restored')
    parser.add_argument('--ckpt_dir',
                    default = None,
                    type = str,
                    help = 'Checkpoint directory')
    parser.add_argument('--no_cuda', dest = 'use_gpu', action = 'store_false',
                    help = 'Use GPU or not')
    parser.add_argument('--model',
                    # required = True,
                    type = str,
                    default = 'jacobi',
                    choices=["jacobi", "shadow", "gcnii"],
                    help = 'Choose the model among: jacobi, shadow, gcnii')
    parser.add_argument('--dataset',
                    required = True,
                    choices = ['word', 'cnse', 'cnss'],
                    help = 'Choose the dataset among: word, cnse, cnss')
    parser.add_argument('--config',
                    default = "configs/config.yaml",
                    type = str,
                    help = 'Yaml Config file path')
    parser.add_argument('--split_file',
                    default = None,
                    type = str,
                    help = 'Split json file')
    parser.add_argument('--new_split',
                    action = 'store_true',
                    help = 'Generate a new split')
    parser.add_argument('--dataset_cache_path',
                    default = "dataset_cache.pkl",
                    type = str,
                    help = 'dataset cache path')
    parser.add_argument('--overwrite_dataset_cache',
                    action = "store_true",
                    help = 'Overwrite dataset cache')

    parser.add_argument('--batch_size',
                    type = int, default = 4,
                    help = 'Batch size')
    parser.add_argument('--num_epochs',
                    type = int, default = 30,
                    help = 'Epoch number')
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # Model training settings
    parser.add_argument('--random_seed',
                    type = int, default = 42,
                    help = 'Random seed')
    parser.add_argument('--eval',
                    action = 'store_true',
                    help = 'Whether do evaluation during training')
    parser.add_argument('--logging_steps', type=int, default=125,
                        help="Log every X updates steps.")

    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join("ckpts", args.dataset + "_" + args.model)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    args.dataset_cache_path = os.path.join(args.ckpt_dir, args.dataset_cache_path)

    with open(args.config) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.dataset == 'word':
        config_class, model_class, _ = WORD_MODEL_CLASSES["bert"]
        model_config = config_class.from_pretrained(yaml_config['MODEL']['WORD']["bert"])
        encoder_model = model_class.from_pretrained(yaml_config['MODEL']['WORD']["bert"])

        _, _, tokenizer_class = WORD_MODEL_CLASSES["bert"]
        tokenizer = tokenizer_class.from_pretrained(yaml_config['MODEL']['WORD']["bert"])
        # model = CtrsTransformer(args = args, yaml_config = yaml_config)

        train_meta, val_meta, test_meta, id2concept = read_word_data(args, yaml_config, tokenizer, val_size = 0, encoder_model = encoder_model, debug_size = DEBUG_SIZE)

    elif args.dataset in ['cnse', 'cnss']:
        model_config = AutoConfig.from_pretrained(yaml_config['MODEL']['CNS']["bert"], output_hidden_states=True)

        encoder_model = AutoModelForMaskedLM.from_pretrained(yaml_config['MODEL']['CNS']["bert"], config = model_config)

        tokenizer = AutoTokenizer.from_pretrained(yaml_config['MODEL']['CNS']["bert"])
        # model = CtrsTransformer(args = args, yaml_config = yaml_config)
        # model, ckpt_path, train_history = load_checkpoint(model, args, yaml_config)
        # model.to(device)
        train_meta, val_meta, test_meta, id2concept = read_cns_data(args, yaml_config, tokenizer, encoder_model = encoder_model, debug_size = DEBUG_SIZE)

    else:
        raise NotImplementedError

    # ------------------ Data ------------------
    train_set = ConceptPairDataset(train_meta, id2concept, yaml_config)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'],
                          num_workers = 0,
                          pin_memory = True)

    if val_meta is None:
        logger.warning("Val meta data is None, using test meta as val meta")
        val_meta = test_meta

    val_set = ConceptPairDataset(val_meta, id2concept, yaml_config)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                          follow_batch=['x_s', 'x_t'],
                          num_workers = 0,
                          pin_memory = True)

    test_set = ConceptPairDataset(test_meta, id2concept, yaml_config)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                          follow_batch=['x_s', 'x_t'],
                          num_workers = 0,
                          pin_memory = True)

    dataloaders_dict = {'train' : train_loader,
                        'val'   : val_loader,
                        'test'  : test_loader}
    dataset_sizes = {'train' : len(train_set),
                     'val'   : len(val_set),
                     'test'  : len(test_set)}

    print(dataset_sizes)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CreGNN(model_config, args)
    model, ckpt_path, train_history = load_checkpoint(model, args, yaml_config, args.device)
    model = model.to(args.device)

    if not args.eval:
        model, last_model, train_history = \
                        train_model(model, dataloaders_dict, dataset_sizes,
                                    ckpt_dir = ckpt_path,
                                    train_history = None, 
                                    args = args,
                                    yaml_config = yaml_config)

        logger.info("Training finished! Training history: %s", train_history)

    # ------------------ Test ------------------
    print("-------- last model --------")
    evaluate_model(model, test_set, args.device, args)
