from __future__ import print_function, division

# ---------- Basic dependencies ----------
from tqdm import tqdm
import os
import pickle
import numpy as np
import argparse
import json
import copy
import time

# ---------- Model dependencies ----------
import sklearn
from sklearn.model_selection import train_test_split
# from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# from PIL import Image
# from random import randrange

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
from config import *
from model import SimBertGNN
from data import WholeDocPairDataset, load_word_for_bert, collate_doc_pair
# from eval import evaluate_accuracy

def model_predict(model, split_dataloader, device):

    model.eval()   # Set model to evaluate mode
    pred_logits = []
    for padded_batch_s, padded_batch_t, sent_len_s_t, _ in tqdm(split_dataloader):

        padded_batch_s = padded_batch_s.to(device)
        padded_batch_t = padded_batch_t.to(device)
        sent_len_s_t = sent_len_s_t.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(padded_batch_s, padded_batch_t, sent_len_s_t, device)

        pred_logits += outputs.detach()[:, 0].tolist()

    return np.array(pred_logits)

def specificity_score(test_label, pred_label):
    recall = sklearn.metrics.recall_score(test_label, pred_label)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(test_label, pred_label)
    specificity = balanced_acc * 2 - recall
    return specificity

def evaluate_model(model, split, device):
    split_dataloader = torch.utils.data.DataLoader(split, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_doc_pair)
    pred_logits = model_predict(model, split_dataloader, device)
    pred_classes = (pred_logits > 0) * 1
    true_classes = (np.array(split.scores) > split.score_threshold) * 1

    acc = sklearn.metrics.accuracy_score(true_classes, pred_classes)
    precision = sklearn.metrics.precision_score(true_classes, pred_classes)
    recall = sklearn.metrics.recall_score(true_classes, pred_classes)
    f1_score = sklearn.metrics.f1_score(true_classes, pred_classes)
    specificity = specificity_score(true_classes, pred_classes)
    auc = sklearn.metrics.roc_auc_score(true_classes, pred_logits)
    # # Expit & logit give the same auc
    # print("logit", pred_logits)
    # print("expit", expit(pred_logits))
    # print("auc logit:", sklearn.metrics.roc_auc_score(true_classes, pred_logits))
    # print("auc expit:", sklearn.metrics.roc_auc_score(true_classes, expit(pred_logits)))

    print("-------------- Test model --------------")
    # print(true_classes)
    # print(pred_classes)
    print('Accuracy = {} ;\nPrecision = {} ;\nRecall/Sensitivity = {} ;\nF1 score = {} ;\nSpecificity = {} ;\nAUC = {}'.format(acc, precision, recall, f1_score, specificity, auc))
    print('Latex table row: {}\% & {}\% & {}\% & {}\% & {}\% & {}\%'.format(round(acc * 100., 2), round(precision * 100., 2), round(recall * 100., 2), round(specificity * 100., 2), round(f1_score * 100., 2), round(auc * 100., 2)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description ='Evaluate saved model')
    parser.add_argument('-m', '--model', dest ='ckpt_name', action ='store',
                    default = None,
                    required = True,
                    help = 'Filename of the model checkpoint that is to be evaluated')
    parser.add_argument('--no_cuda', dest ='use_gpu', action ='store_false',
                    help = 'Use GPU or not')
    args = parser.parse_args()

    if args.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is used")
        else:
            device = "cpu"
            print("GPU not available, CPU is used")
    else:
        device = "cpu"
        print("CPU is used")
    device = torch.device(device)

    # ------------------ Model ------------------
    config = BertConfig(vocab_size_or_config_json_file=32000,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072)
    model = SimBertGNN(config)

    ckpt_path = args.ckpt_name
    if os.path.isfile(ckpt_path):
        print("Load model saved at checkpoint:", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("Checkpoint file does not exist:", ckpt_path)
        exit(1)

    model.to(device)

    # ------------------ Data ------------------
    with open(FIXED_SPLIT_FILE, "r") as split_file:
        train_meta, val_meta, test_meta = json.load(split_file)
    print("Loaded split from file:", FIXED_SPLIT_FILE)

    # # Create train set
    # doc_s_list, doc_t_list, scores = load_word_for_bert(train_meta, debug_size = None)
    # train_set = DocPairDataset(doc_s_list, doc_t_list, scores)
    #
    # # Create val set
    # doc_s_list, doc_t_list, scores = load_word_for_bert(val_meta, debug_size = None)
    # val_set = DocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # Create test set
    doc_s_list, doc_t_list, scores = load_word_for_bert(test_meta, debug_size = None)
    test_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # dataloaders_dict = {'train' : torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
    #                     'val'   : torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False),
    #                     'test'  : torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    #                     }
    # dataset_sizes = {'train' : len(train_set),
    #                  'val'   : len(val_set),
    #                  'test'  : len(test_set)
    #                 }

    # # ------------------ Evaluate training set ------------------
    # pred_logits = model_predict(model, dataloaders_dict, dataset_sizes, 'train', device)
    # pred_classes = (pred_logits > 0) * 1
    # true_classes = (np.array(train_set.scores) > train_set.score_threshold) * 1
    #
    # acc = accuracy_score(true_classes, pred_classes)
    # precision = precision_score(true_classes, pred_classes)
    # recall = recall_score(true_classes, pred_classes)
    # f1 = f1_score(true_classes, pred_classes)
    # specificity = specificity_score(true_classes, pred_classes)
    # auc = sklearn.metrics.roc_auc_score(true_classes, pred_logits)
    #
    #
    # print("-------- train --------")
    # # print(true_classes)
    # # print(pred_classes)
    # print('Precision = {}  ;  Recall/Sensitivity = {}  ;  F1 score = {}  ;  Specificity = {}  ;  AUC = {}'.format(precision, recall, f1, specificity, auc))
    # print('Latex table row: {}\% & {}\% & {}\% & {}\% & {}\% & {}\%'.format(round(test_acc * 100., 2), round(precision * 100., 2), round(recall * 100., 2), round(specificity * 100., 2), round(f1_score * 100., 2), round(auc * 100., 2)))

    # # ------------------ Evaluate testing set ------------------
    # pred_logits = model_predict(model, dataloaders_dict, dataset_sizes, 'val', device)
    # pred_classes = (pred_logits > 0) * 1
    # true_classes = (np.array(test_set.scores) > test_set.score_threshold) * 1
    #
    # acc = accuracy_score(true_classes, pred_classes)
    # precision = precision_score(true_classes, pred_classes)
    # recall = recall_score(true_classes, pred_classes)
    # f1 = f1_score(true_classes, pred_classes)
    #
    # print("-------- val --------")
    # # print(true_classes)
    # # print(pred_classes)
    # print("acc, precision, recall, f1:", acc, precision, recall, f1)

    # ------------------ Evaluate test set ------------------
    evaluate_model(model, test_set, device)
