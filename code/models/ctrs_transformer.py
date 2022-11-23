from typing import Union, Tuple, List, Iterable, Dict
import math
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from transformers import (BertTokenizer, BertModel, BertConfig,
                            RobertaConfig, RobertaModel, RobertaTokenizer,
                            XLNetConfig, XLNetModel, XLNetTokenizer)
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import numpy as np


WORD_MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
}


class CtrsTransformer(nn.Module):
    def __init__(self, r=32, m=0.999, T=0.1, args=None, yaml_config=None):
        super(CtrsTransformer, self).__init__()
        self.yaml_config = yaml_config
        self.args = args

        if args.dataset == 'word':
            config_class, model_class, _ = WORD_MODEL_CLASSES[args.model]
            self.config = config_class.from_pretrained(yaml_config['MODEL']['WORD'][args.model])
            self.encoder_q = model_class.from_pretrained(yaml_config['MODEL']['WORD'][args.model])
            self.encoder_k = model_class.from_pretrained(yaml_config['MODEL']['WORD'][args.model])

        elif args.dataset in ['cnse', 'cnss']:
            self.config = AutoConfig.from_pretrained(yaml_config['MODEL']['CNS'][args.model], output_hidden_states=True)

            if args.model != 'xlnet':
                self.encoder_q = AutoModelForMaskedLM.from_pretrained(yaml_config['MODEL']['CNS'][args.model], config = self.config)
                self.encoder_k = AutoModelForMaskedLM.from_pretrained(yaml_config['MODEL']['CNS'][args.model], config = self.config)
            else:
                self.encoder_q = AutoModelForCausalLM.from_pretrained(yaml_config['MODEL']['CNS'][args.model], config = self.config)
                self.encoder_k = AutoModelForCausalLM.from_pretrained(yaml_config['MODEL']['CNS'][args.model], config = self.config)

        else:
            raise NotImplementedError

        self.mlp_q = MLPLayer(self.config)
        self.mlp_k = MLPLayer(self.config)
        self.pooler = Pooler(yaml_config['MODEL']['POOLER_TYPE'])

        # ---------------------- CRE ----------------------
        self.cre_loss = SoftmaxLoss(
                                sentence_embedding_dimension = self.config.hidden_size,
                                num_labels = 2,
                                concatenation_sent_rep = True,
                                concatenation_sent_difference = True,
                                concatenation_sent_multiplication = False)

        # ---------------------- GCCL ----------------------
        self.r = r
        self.m = m
        self.T = T

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.config.hidden_size, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, 
                srs_input_ids = None, 
                srs_seg_masks = None, 
                srs_att_masks = None, 
                tgt_input_ids = None, 
                tgt_seg_masks = None, 
                tgt_att_masks = None, 
                labels = None,
                srs_ids = None,
                tgt_ids = None,
                cluster_result = None,
                beta = 0.1,
                gamma = 0.1,
                ):

        loss_details = {}
        loss = 0

        side_by_side_input_ids = torch.cat([srs_input_ids, tgt_input_ids], dim = -1)

        srs_tgt_features = self.get_doc_embedding(side_by_side_input_ids, mode = 'q')
        srs_cls = srs_tgt_features['last_hidden_states'][:, 0, 0]
        tgt_cls = srs_tgt_features['last_hidden_states'][:, 0, self.yaml_config["DATASET"]["MAX_SEG_LEN"]]
        loss_cre, cre_logits = self.cre_loss([srs_cls, tgt_cls], labels.unsqueeze(-1) * 1.)

        loss_details["loss_cre"] = loss_cre
        loss += loss_cre

        if cluster_result is not None:
            # ---------------------- nce loss ----------------------
            # compute key features
            with torch.no_grad():  # no gradient to keys

                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                shuffled_x, idx_unshuffle = self._batch_shuffle_ddp([side_by_side_input_ids])

                k_features = self.get_doc_embedding(shuffled_x[0], mode = "k")
                k_srs_cls = k_features['last_hidden_states'][:, 0, 0]
                k_tgt_cls = k_features['last_hidden_states'][:, 0, self.yaml_config["DATASET"]["MAX_SEG_LEN"]]
                k_srs = self.mlp_k(k_srs_cls)
                k_tgt = self.mlp_k(k_tgt_cls)
                k_srs = nn.functional.normalize(k_srs, dim=1)
                k_tgt = nn.functional.normalize(k_tgt, dim=1)

                # undo shuffle
                k_srs = self._batch_unshuffle_ddp(k_srs, idx_unshuffle)
                k_tgt = self._batch_unshuffle_ddp(k_tgt, idx_unshuffle)
                k = torch.cat([k_srs, k_tgt], dim = 0)

            # compute query features
            srs_mlp = self.mlp_q(srs_cls)
            tgt_mlp = self.mlp_q(tgt_cls)
            q = torch.cat([srs_mlp, tgt_mlp], dim = 0)
            q = nn.functional.normalize(q, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: Nxr
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+r)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            loss_infonce = self.criterion(logits, labels)
            loss_details["loss_infonce"] = loss_infonce
            loss += loss_infonce * gamma

            # ---------------------- gc-nce loss ----------------------
            concept2cluster, prototypes, density = cluster_result['concept2cluster'], cluster_result['centroids'], cluster_result['density']

            # get positive component
            both_ids = torch.cat([srs_ids, tgt_ids], dim = 0)
            pos_proto_id = concept2cluster[both_ids]
            pos_prototypes = prototypes[pos_proto_id]
            
            # sample negative components
            all_proto_id = [i for i in range(len(prototypes))]
            neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
            neg_proto_id = sample(neg_proto_id, self.r) #sample r negative components
            neg_prototypes = prototypes[neg_proto_id]

            proto_selected = torch.cat([pos_prototypes, neg_prototypes],dim=0)
            
            # compute component logits
            logits_proto = torch.mm(q, proto_selected.t())
            
            # targets for component assignment
            labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()

            # scaling temperatures for the selected components
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()], dim=0)]  
            logits_proto /= temp_proto

            loss_proto = self.criterion(logits_proto, labels_proto)
            loss_details["loss_proto"] = loss_proto
            loss += loss_proto * beta

        loss_details["loss"] = loss

        return loss, loss_details, cre_logits

    def get_doc_embedding(self, input_ids = None, seg_masks = None, att_masks = None, mode = "q"):
        batch_size, seg_num, seg_len = input_ids.shape
        input_ids_flat = input_ids.view((-1, seg_len))
        if att_masks is not None:
            att_masks_flat = att_masks.view((-1, seg_len))
        else:
            att_masks_flat = None
        if mode == "q":
            encoder_outputs = self.encoder_q(input_ids = input_ids_flat, 
                                            attention_mask = att_masks_flat, 
                                            output_hidden_states=True)
        elif mode == "k":
            encoder_outputs = self.encoder_k(input_ids = input_ids_flat, 
                                            attention_mask = att_masks_flat, 
                                            output_hidden_states=True)
        else:
            raise NotImplementedError

        pooler_output = self.pooler(att_masks_flat, encoder_outputs, self.args)
        seg_embeddings = pooler_output.view((batch_size, seg_num, -1))
        last_hidden_states = encoder_outputs.hidden_states[-1].view((batch_size, seg_num, seg_len, -1))
        doc_embeddings = seg_embeddings[:, 0, :]

        if mode == "q":
            seg_embed = self.mlp_q(doc_embeddings)
        elif mode == "k":
            seg_embed = self.mlp_k(doc_embeddings)
        else:
            raise NotImplementedError

        features = {'doc_embeddings' : doc_embeddings,
                    'mlp_embeddings' : seg_embed,
                    # 'cre_embeddings' : None,
                    'last_hidden_states' : last_hidden_states}

        return features

    def predict_labels(self, logits):
        pred_labels = (logits[:, 0] > 0) * 1.
        sigma = nn.Sigmoid()
        pred_probs = sigma(logits)

        return pred_labels, pred_probs

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        remain_size = batch_size
        while True:
            avail_size = self.queue[:, ptr:ptr + remain_size].size(1)
            if avail_size < remain_size:
                self.queue[:, ptr:ptr + avail_size] = keys[:avail_size, :].T
                keys = keys[avail_size:, :]
                ptr = (ptr + avail_size) % self.r  # move pointer
                remain_size -= avail_size
            else:
                self.queue[:, ptr:ptr + remain_size] = keys[:remain_size, :].T
                ptr = (ptr + remain_size) % self.r  # move pointer
                break

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x_list):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # batch_size_this = x_list[0].shape[0]
        # x_list_gather = [concat_all_gather(x) for x in x_list]
        # batch_size_all = x_list_gather[0].shape[0]

        # num_gpus = batch_size_all // batch_size_this
        batch_size = x_list[0].shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)

        shuffled_x_list = [x[idx_shuffle] for x in x_list]

        return shuffled_x_list, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        # batch_size_all = x_gather.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_unshuffle]

# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs, args):
        if args.dataset == 'word':
            hidden_states = outputs.hidden_states

        elif args.dataset in ['cnse', 'cnss']:

            if args.model != 'xlnet':
                hidden_states = outputs.hidden_states
            else:
                hidden_states = outputs[-1]

        else:
            raise NotImplementedError

        # last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            last_hidden = hidden_states[-1]
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            last_hidden = hidden_states[-1]
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class SoftmaxLoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                #  model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        super(SoftmaxLoss, self).__init__()
        # self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        # logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))

        if num_labels > 2:
            self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
            self.loss_fct = nn.CrossEntropyLoss()
        elif num_labels == 2:
            self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, 1)
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def forward(self, reps, labels: Tensor):
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        # print(labels.view(-1).shape)
        loss = self.loss_fct(output, labels)

        # if labels is not None:
        #     loss = loss_fct(output, labels.view(-1))
        #     return loss
        # else:
        #     return reps, output
        return loss, output

