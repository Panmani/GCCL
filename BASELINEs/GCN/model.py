# https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig

# ------------------------- GCN -------------------------
from torch_geometric.data import DataLoader, Dataset, Data
from torch_geometric.nn import Sequential, GCNConv, SAGPooling
from torch.nn import Module, Linear, ReLU, Sigmoid, Dropout


import json
from tqdm import tqdm
import pickle
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

from config import *


# class SimBERT(nn.Module):
#     def __init__(self, config):
#         super(SimBERT, self).__init__()
#         # self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained(MODEL_NAME)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         nn.init.xavier_normal_(self.classifier.weight)
#
#     def forward(self, input_ids_s_sep_t, token_type_ids=None, attention_mask=None, labels=None):
#         last_hidden_state_s, pooled_output_s = self.bert(input_ids_s_sep_t, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         sep_embeddings = self.dropout(last_hidden_state_s[:, 0, :])
#
#         # last_hidden_state_s, pooled_output_s = self.bert(input_ids_s, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         # pooled_output_s = self.dropout(pooled_output_s)
#
#         # last_hidden_state_t, pooled_output_t = self.bert(input_ids_t, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         # pooled_output_t = self.dropout(pooled_output_t)
#
#         # s_t_concat = torch.cat([pooled_output_s, pooled_output_t], 1)
#         logits = self.classifier(sep_embeddings)
#         return logits
#
#     def get_embedding(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         """For testing loaded self.bert"""
#         last_hidden_state, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         return last_hidden_state.detach().numpy()
#
#     def freeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def unfreeze_bert_encoder(self):
#         for param in self.bert.parameters():
#             param.requires_grad = True

class SimBertGNN(nn.Module):
    def __init__(self, config):
        super(SimBertGNN, self).__init__()
        # self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # nn.init.xavier_normal_(self.classifier.weight)

        self.GCN_backbone = Sequential('x, edge_index, null, batch', [
            (Dropout(p=0.1), 'x -> x'),
            (GCNConv(768, 64), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (SAGPooling(64, 0.5), 'x1, edge_index, null, batch, null -> x2, edge_index_2, _, batch, _, _'),
            (GCNConv(64, 16), 'x2, edge_index_2 -> x3'),
            ReLU(inplace=True),
            (SAGPooling(16, 0.5), 'x3, edge_index_2, null, batch, null -> x4, edge_index_4, _, batch, _, _'),
            # (GCNConv(64, 16), 'x4, edge_index_4 -> x5'),
            # ReLU(inplace=True),
            # (SAGPooling(16, 0.5), 'x5, edge_index_4, null, batch, null -> x6, edge_index_6, _, batch, _, _'),
        ])

        self.fc1 = Linear(16 * 2, 1)

        # nn.init.xavier_normal_(self.GCN_backbone.weight)
        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, batch_s_mtx, batch_t_mtx, sent_lens, device, token_type_ids=None, attention_mask=None, labels=None):
        # print(batch_s_mtx.shape)
        # print(batch_t_mtx.shape)

        batch_size = sent_lens.shape[0]

        s_mtx_collapse = batch_s_mtx.reshape((-1,) + batch_s_mtx.shape[2:])
        last_hidden_state_s, pooled_output_s = self.bert(s_mtx_collapse, token_type_ids, attention_mask, output_all_encoded_layers=False)
        s_node_embed = self.dropout(last_hidden_state_s[:, 0, :])

        t_mtx_collapse = batch_t_mtx.reshape((-1,) + batch_t_mtx.shape[2:])
        last_hidden_state_t, pooled_output_t = self.bert(t_mtx_collapse, token_type_ids, attention_mask, output_all_encoded_layers=False)
        t_node_embed = self.dropout(last_hidden_state_t[:, 0, :])

        edge_index_s = []
        x_s_batch = []
        edge_index_t = []
        x_t_batch = []
        for doc_pair_idx in range(batch_size):
            x_s_batch += [doc_pair_idx] * MAX_SENT_NUM
            for s_sent_idx in range(sent_lens[doc_pair_idx, 0] - 1):
                from_node = doc_pair_idx * MAX_SENT_NUM + s_sent_idx
                edge_index_s.append([from_node, from_node + 1])

            x_t_batch += [doc_pair_idx] * MAX_SENT_NUM
            for t_sent_idx in range(sent_lens[doc_pair_idx, 1] - 1):
                from_node = doc_pair_idx * MAX_SENT_NUM + t_sent_idx
                edge_index_t.append([from_node, from_node + 1])

        edge_index_s = torch.tensor(edge_index_s, dtype = torch.int64).t().to(device)
        edge_index_t = torch.tensor(edge_index_t, dtype = torch.int64).t().to(device)
        # print(edge_index_s.shape)
        # print(edge_index_s.shape)
        x_s_batch = torch.tensor(x_s_batch, dtype = torch.int64).to(device)
        x_t_batch = torch.tensor(x_t_batch, dtype = torch.int64).to(device)
        # print(s_node_embed.shape)
        # print(t_node_embed.shape)
        # print(edge_index_s.shape)
        # print(edge_index_t.shape)

        x_s_out, edge_index_s_out, _, x_s_batch_out, _, _ = self.GCN_backbone(s_node_embed, edge_index_s, None, x_s_batch)
        embed_s = self.get_graph_embedding(batch_size, x_s_out, x_s_batch_out)

        x_t_out, edge_index_t_out, _, x_t_batch_out, _, _ = self.GCN_backbone(s_node_embed, edge_index_s, None, x_s_batch)
        embed_t = self.get_graph_embedding(batch_size, x_t_out, x_t_batch_out)

        embedding_concat = torch.cat([embed_s, embed_t], 1)
        logits = self.fc1(embedding_concat)
        return logits

    # def forward(self, x_s, edge_index_s, x_s_batch, x_t, edge_index_t, x_t_batch):
    #     batch_size = x_s_batch.max() + 1
    #     x_s_out, edge_index_s_out, _, x_s_batch_out, _, _ = self.GCN_backbone(x_s, edge_index_s, None, x_s_batch)
    #     embed_s = self.get_graph_embedding(batch_size, x_s_out, x_s_batch_out)
    #
    #     x_t_out, edge_index_t_out, _, x_t_batch_out, _, _ = self.GCN_backbone(x_t, edge_index_t, None, x_t_batch)
    #     embed_t = self.get_graph_embedding(batch_size, x_t_out, x_t_batch_out)
    #
    #     embedding_concat = torch.cat([embed_s, embed_t], 1)
    #     outputs = self.fc1(embedding_concat)
    #     return outputs

    def get_graph_embedding(self, batch_size, x_out, x_batch):
        sum_vectors = torch.zeros((batch_size, x_out.shape[1]))
        if next(self.parameters()).is_cuda:
            sum_vectors = sum_vectors.to("cuda")
        sum_vectors = sum_vectors.index_add(0, x_batch, x_out)
        # bincount = torch.bincount(x_batch, minlength=batch_size)
        # node_counts = bincount.float()
        # node_counts[bincount == 0] = 1.0
        # mean = sum_vectors / node_counts
        return sum_vectors

    def get_embedding(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """For testing loaded self.bert"""
        last_hidden_state, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        return last_hidden_state.detach().numpy()

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    config = BertConfig(hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2
    model = SimBertGNN(config, num_labels)

    with open(os.path.join(RAW_DATA_DIR, WIKI_DOC_DIR, "3D_computer_graphics.txt"), "r") as input_file:
        input_text = input_file.read()
    # text ='what is a pug'
    input_text = input_text[:300]
    zz = tokenizer.tokenize(input_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(zz)])

    embed = model.get_embedding(tokens_tensor)

    print(zz)
    print(embed)
    print(embed.shape)
