# ---------- Basic dependencies ----------
from tqdm import tqdm
import os
import pickle
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import random
import math
import networkx as nx
import time
import copy
import wandb
from tensorboardX import SummaryWriter
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Model dependencies ----------
import torch
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
import sklearn
import torch.nn as nn
from torch import Tensor

# ---------- local ----------
from transformers import get_linear_schedule_with_warmup
# from download import read_wiki_doc
# from configs.config import MAX_SEQ_LEN, SCORE_THRESHOLD, HISTORY_NAME
# from models.cns_transformer import ConcreteTransformer

class HistoryLogger():
    def __init__(self, project : str, 
                        tb_dir = None, 
                        args = None, 
                        yaml = None):

        self.tb_dir = tb_dir
        # self.log_current_values = log_current_values
        wandb.init(project=project, entity="panmani", config=yaml)

        if args is not None:
            wandb.config.update(args)
        if self.tb_dir is not None:
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        self.head = {}
        self.tail = {}
        self.tail_step = {}
        self.head_step = {}

    def update(self, values : dict):
        for mtc in values:
            if isinstance(values[mtc], float):
                new_value = values[mtc]
            else:
                new_value = values[mtc].mean().item()

            if mtc not in self.head:
                self.head[mtc] = new_value
                self.head_step[mtc] = 1
                self.tail[mtc] = 0.0
                self.tail_step[mtc] = 0
            else:
                self.head[mtc] += new_value
                self.head_step[mtc] += 1

    def log(self):
        for mtc in self.head:
            if self.head_step[mtc] > self.tail_step[mtc]:
                logging_value = (self.head[mtc] - self.tail[mtc]) / (self.head_step[mtc] - self.tail_step[mtc])
                logging_dict = {mtc : logging_value}
                # logging_dict[mtc] = logging_value
                self.tail[mtc] = self.head[mtc]
                self.tail_step[mtc] = self.head_step[mtc]

                wandb.log(logging_dict, step=self.head_step[mtc])
                if self.tb_dir is not None:
                    self.tb_writer.add_scalar(mtc, logging_value, self.head_step[mtc])

    def direct_log(self, mtc_value):
        step = self.head_step['loss']
        wandb.log(mtc_value, step=step)
        if self.tb_dir is not None:
            for mtc in mtc_value:
                self.tb_writer.add_scalar(mtc, mtc_value[mtc], step)

    def get_value(self, mtc):
        return self.head[mtc] / self.head_step[mtc]

def train_model(model, dataloaders_dict, dataset_sizes, 
                        ckpt_dir = None, 
                        train_history = None, 
                        args = None, 
                        yaml_config = None):
    since = time.time()

    if train_history is None:
        train_history = {
                'train' : {
                    'loss': [],
                    # 'cls_loss': [],
                    # 'mse_loss': [],
                    'acc' : [],
                },
                'val'   : {
                    'loss': [],
                    # 'cls_loss': [],
                    # 'mse_loss': [],
                    'acc' : [],
                },
            }
        best_loss = 100
    else:
        best_loss = train_history['val']['loss'][args.start_epoch-1]

    # Each epoch has a training and validation phase
    phases = ['train', 'val']

    # print('Start training with best loss:', best_loss)
    # best_model_wts = copy.deepcopy(model.state_dict())
    history_logger = HistoryLogger("CreGNN", tb_dir = os.path.join(ckpt_dir, "tb_log"), 
                                    args = args, yaml = yaml_config)

    # Prepare optimizer and schedule (linear warmup and decay)

    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    t_total = len(dataloaders_dict['train']) * args.num_epochs
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    for epoch in range(args.start_epoch, args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            similarity_corrects = 0
            # Iterate over data.
            epoch_iterator = tqdm(dataloaders_dict[phase])
            for step, batch in enumerate(epoch_iterator):

                data, labels = batch
                data = data.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                #     # outputs = model(data.x_s, data.edge_index_s, data.x_s_batch,
                #     #                 data.x_t, data.edge_index_t, data.x_t_batch)
                #     print(data.x_s.shape, data.edge_index_s.shape, data.x_s_batch.shape, data.root_n_id_s.shape)
                #     loss, outputs = model(data.x_s, data.edge_index_s, data.x_s_batch, data.root_n_id_s, 
                #                     data.x_t, data.edge_index_t, data.x_t_batch, data.root_n_id_t)
                #     # loss = criterion(outputs, labels)
                #     # backward + optimize only if in training phase
                #     if phase == 'train':
                #         loss.backward()
                #         optimizer.step()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss, loss_details, logits = model(data.x_s, data.edge_index_s, data.x_s_batch, data.root_n_id_s, 
                                                        data.x_t, data.edge_index_t, data.x_t_batch, data.root_n_id_t,
                                                        labels)
                    # loss, loss_details, logits = model(cluster_result = cluster_result, **batch)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                elif phase == 'val':
                    with torch.no_grad():
                        # Only generate CRE logits
                        loss, loss_details, logits = model(data.x_s, data.edge_index_s, data.x_s_batch, data.root_n_id_s, 
                                                            data.x_t, data.edge_index_t, data.x_t_batch, data.root_n_id_t,
                                                            labels)
                else:
                    raise ValueError

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_labels, _ = model.predict_labels(logits)
                # print(pred_labels, labels, pred_labels == labels.cuda())
                similarity_corrects += torch.sum(pred_labels == labels)

                # group_idx, param_idx = 0, 0
                # current_lr = get_current_lr(optimizer, group_idx, param_idx)
                # print('Current learning rate (g:%d, p:%d): %.4f'%(group_idx, param_idx, current_lr))

                global_step = step + epoch * len(epoch_iterator)
                epoch_iterator.set_description(
                    (f'Phase {phase}; Step: {global_step}; loss: {loss.item():.3f}; ')
                )

                if phase == 'train':
                    history_logger.update(loss_details)
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        history_logger.log()

            if phase == 'train':
                save_checkpoint(model, epoch+1, ckpt_dir, args)

            epoch_loss = running_loss / dataset_sizes[phase]
            similarity_acc = similarity_corrects.double() / dataset_sizes[phase]
            cur_acc = similarity_acc.item()

            # print('{} loss: {} '.format(phase, epoch_loss))
            # print('{} similarity_acc: {}'.format(phase, similarity_acc))

            history_logger.direct_log({ phase + '_epoch_loss'     : epoch_loss,
                                        phase + '_similarity_acc' : cur_acc} )

            train_history[phase]['loss'].append(epoch_loss)
            train_history[phase]['acc'].append(similarity_acc.item())
            with open(os.path.join(ckpt_dir, yaml_config['MODEL']['HISTORY_NAME']), "w") as history_file:
                json.dump(train_history, history_file)

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                        'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                save_checkpoint(model, -1, ckpt_dir, args)
                best_model = copy.deepcopy(model)
                # torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_best.bin'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(best_loss)))

    # # load best model weights
    # print("Best val loss from epoch:", np.argmin(train_history['val']['loss']))
    # model.load_state_dict(best_model_wts)

    return best_model, model, train_history

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

def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def batch_to_device(batch, target_device, exclude = []):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for input_name in batch.keys():
        if input_name not in exclude:
            batch[input_name] = batch[input_name].to(target_device)
    return batch

def save_checkpoint(model, epoch, ckpt_dir, args):
    # save the full model and optmizer into a checkpoint
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'args': model_to_save.args,
    }

    if not os.path.exists(ckpt_dir) and args.local_rank in [-1, 0]:
        os.makedirs(ckpt_dir)

    logger.info("Saving full checkpoint to %s", ckpt_dir)
    torch.save(checkpoint, os.path.join(ckpt_dir, 'model_epoch_{}.bin'.format(epoch)))

def load_checkpoint(model, args, yaml_config, device):
    if args.load_ckpt is None:
        print("Train from scratch!")
        args.start_epoch = 0

        ckpt_name = args.dataset
        ckpt_name += "_" + args.model
        ckpt_name += "_seed@" + (str(args.random_seed) if args.random_seed is not None else "n")
        # ckpt_name += "_segs@{}x{}".format(yaml_config['DATASET']['MAX_SEG_NUM'], yaml_config['DATASET']['MAX_SEG_LEN'])

        # if args.use_comm:
        #     ckpt_name += "_comm"

        # if args.use_aug:
        #     ckpt_name += "_aug@" + str(args.aug_ratio) if args.aug_ratio is not None else "n"
        #     ckpt_name += f"+{args.k_hops}hop"
        #     # ckpt_name += "+" + args.edge_weight_mapping
        #     ckpt_name += "+" + args.score_estimator
        #     ckpt_name += "+ft" + (str(args.score_threshold) if args.score_threshold is not None else "n")
        #     # ckpt_name += f"+flr({args.filter_length_range[0]},{args.filter_length_range[1]})"
        #     # ckpt_name += f"+suc{args.success_rate_threshold}"

        # if args.use_pcl:
        #     ckpt_name += "_pcl"

        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

        # if not os.path.isdir(ckpt_path):
        #     os.makedirs(ckpt_path)
        print("Checkpoints will be saved into:", ckpt_path)

        train_history = None

    else:
        ckpt_path = args.load_ckpt
        if os.path.isfile(ckpt_path):
            print("Load model saved at checkpoint:", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            ckpt_path_head, ckpt_path_tail = os.path.split(ckpt_path)
            if len(ckpt_path_tail) > 0:
                args.start_epoch = int(ckpt_path_tail.split(".")[0][len("model_epoch_"):])
                args.ckpt_dir = ckpt_path_head
                print("Checkpoints will continue to be saved into:", args.ckpt_dir)

                # load history
                if os.path.isfile(os.path.join(args.ckpt_dir, yaml_config['MODEL']['HISTORY_NAME'])):
                    with open(os.path.join(args.ckpt_dir, yaml_config['MODEL']['HISTORY_NAME']), "r") as history_file:
                        train_history = json.load(history_file)
                    train_loss = train_history['train']['loss'][args.start_epoch-1]
                    val_loss = train_history['val']['loss'][args.start_epoch-1]
                    print("At epoch {}: train loss = {} , val loss = {}".format(args.start_epoch, train_loss, val_loss))
                else:
                    print("History not found")
                    train_history = None
            else:
                print("Checkpoint file does not exist:", ckpt_path)
                exit(1)
        else:
            print("Checkpoint file does not exist:", ckpt_path)
            exit(1)

    return model, ckpt_path, train_history


# =========================== Evaluate ===========================
def model_predict(model, split_dataloader, device):

    model.eval()   # Set model to evaluate mode
    pred_labels = []
    pred_probs = []
    # true_labels = []
    for batch in tqdm(split_dataloader):
        data, labels = batch
        data = data.to(device)
        labels = labels.to(device)

        # srs_input_ids, srs_seg_masks, srs_att_masks, srs_ids, tgt_input_ids, tgt_seg_masks, tgt_att_masks, tgt_ids, labels = batch
        # batch = {
        #     "srs_input_ids" : srs_input_ids, 
        #     "srs_seg_masks" : srs_seg_masks, 
        #     "srs_att_masks" : srs_att_masks, 
        #     "srs_ids" : srs_ids,
        #     "tgt_input_ids" : tgt_input_ids, 
        #     "tgt_seg_masks" : tgt_seg_masks, 
        #     "tgt_att_masks" : tgt_att_masks, 
        #     "tgt_ids" : tgt_ids,
        #     "labels"  : labels,
        # }

        # input_ids, attention_masks, seg_ids, srs_seg_masks, srs_ids, tgt_seg_masks, tgt_ids, labels = batch
        # batch = {
        #     "input_ids" : input_ids, 
        #     "attention_masks" : attention_masks, 
        #     "seg_ids" : seg_ids,
        #     "srs_seg_masks" : srs_seg_masks, 
        #     "srs_ids" : srs_ids,
        #     "tgt_seg_masks" : tgt_seg_masks, 
        #     "tgt_ids" : tgt_ids,
        #     "labels"  : labels,
        # }

        # input_ids, att_masks, labels = batch
        # batch = {
        #     "input_ids" : input_ids, 
        #     "att_masks" : att_masks, 
        #     "labels"  : labels,
        # }

        # batch = batch_to_device(batch, device)

        with torch.no_grad():
            loss, loss_details, logits = model(data.x_s, data.edge_index_s, data.x_s_batch, data.root_n_id_s, 
                                                data.x_t, data.edge_index_t, data.x_t_batch, data.root_n_id_t,
                                                labels)
            cur_pred_labels, cur_pred_probs = model.predict_labels(logits)

        pred_labels += cur_pred_labels.detach().to("cpu").tolist()
        pred_probs.append(cur_pred_probs.cpu().numpy())
        # print(pred_logits)
        # true_labels += labels.tolist()

    pred_probs = np.concatenate(pred_probs, axis = 0)
    return pred_labels, pred_probs

def get_true_labels(dataset, n_classes = 2):
    labels = []
    one_hot_labels = []
    for ex in dataset:
        # print(ex)
        cur_label = ex[-1]
        cur_one_hot = [0] * n_classes
        cur_one_hot[cur_label] = 1
        labels.append(cur_label)
        one_hot_labels.append(cur_one_hot)
    return labels, one_hot_labels

def specificity_score(test_label, pred_label):
    recall = sklearn.metrics.recall_score(test_label, pred_label)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(test_label, pred_label)
    specificity = balanced_acc * 2 - recall
    return specificity

def evaluate_model(model, dataset, device, args):
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False,
                          follow_batch=['x_s', 'x_t'],
                          num_workers = 0,
                          pin_memory = True)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    true_labels, true_one_hot = get_true_labels(dataset)
    pred_labels, pred_probs = model_predict(model, dataloader, device)
    # print(true_labels)
    # print(pred_labels)
    # print(pred_probs)
    # pred_classes = (pred_logits > 0) * 1
    # true_classes = (true_labels > SCORE_THRESHOLD) * 1
    # print(len(true_classes), len(pred_classes))
    # print(true_classes, pred_classes)
    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    precision = sklearn.metrics.precision_score(true_labels, pred_labels)
    recall = sklearn.metrics.recall_score(true_labels, pred_labels)
    f1_score = sklearn.metrics.f1_score(true_labels, pred_labels)
    specificity = specificity_score(true_labels, pred_labels)
    # auc = sklearn.metrics.roc_auc_score(true_one_hot, pred_probs)
    auc = sklearn.metrics.roc_auc_score(true_labels, pred_probs)

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
