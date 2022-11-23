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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import sklearn

# ---------- Ours ----------
from data import get_concrete_graph, get_connected_components


class HistoryLogger():
    def __init__(self, project : str, 
                        tb_dir = None, 
                        history_dict = None,
                        args = None, 
                        yaml = None):

        self.tb_dir = tb_dir
        self.yaml = yaml
        # self.log_current_values = log_current_values
        wandb.init(project=project, entity="panmani", config=yaml)

        if args is not None:
            wandb.config.update(args)
        if self.tb_dir is not None:
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        if history_dict is None:
            self.history_dict = {
                    'train' : {
                        'loss': [],
                        'acc' : [],
                    },
                    'val'   : {
                        'loss': [],
                        'acc' : [],
                    },
                }
            self.best_loss = 100
        else:
            self.history_dict = history_dict
            self.best_loss = history_dict['val']['loss'][args.start_epoch-1]

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

    def update_history_dict(self, phase, loss, acc, ckpt_dir):
        self.history_dict[phase]['loss'].append(loss)
        self.history_dict[phase]['acc'].append(acc)
        with open(os.path.join(ckpt_dir, self.yaml['MODEL']['HISTORY_NAME']), "w") as history_file:
            json.dump(self.history_dict, history_file)

        if phase == 'val' and loss < self.best_loss:
            print('saving with loss of {}'.format(loss),
                        'improved over previous {}'.format(self.best_loss))
            self.best_loss = loss
            return True
        else:
            return False

def train_model(model, dataloaders_dict, dataset_sizes, 
                        ckpt_dir, 
                        concrete_graph = None,
                        id2concept = None,
                        train_history = None, 
                        args = None, 
                        yaml_config = None):
    since = time.time()


    # Each epoch has a training and validation phase
    phases = ['train', 'val']

    # print('Start training with best loss:', best_loss)
    # best_model_wts = copy.deepcopy(model.state_dict())
    history_logger = HistoryLogger("CtrsCRE", tb_dir = os.path.join(ckpt_dir, "tb_log"), history_dict = train_history,
                                    args = args, yaml = yaml_config)

    # Prepare optimizer and schedule (linear warmup and decay)
    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = len(dataloaders_dict['train']) * args.num_epochs
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    for epoch in range(args.start_epoch, args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode

                if args.use_gccl:
                    all_connected_components, concept2cluster = get_connected_components(concrete_graph, id2concept)
                    features = compute_features(concrete_graph, id2concept, dataloaders_dict['train'], model, args)
                    cluster_result = update_centroids(features, all_connected_components, concept2cluster, args)
                else:
                    cluster_result = None

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            similarity_corrects = 0
            # Iterate over data.
            epoch_iterator = tqdm(dataloaders_dict[phase])
            for step, batch in enumerate(epoch_iterator):
                srs_input_ids, srs_seg_masks, srs_att_masks, srs_ids, tgt_input_ids, tgt_seg_masks, tgt_att_masks, tgt_ids, labels = batch

                batch = {
                    "srs_input_ids" : srs_input_ids, 
                    "srs_seg_masks" : srs_seg_masks, 
                    "srs_att_masks" : srs_att_masks, 
                    "srs_ids" : srs_ids,
                    "tgt_input_ids" : tgt_input_ids, 
                    "tgt_seg_masks" : tgt_seg_masks, 
                    "tgt_att_masks" : tgt_att_masks, 
                    "tgt_ids" : tgt_ids,
                    "labels"  : labels,
                    "beta" : args.beta,
                    "gamma" : args.gamma,
                }

                batch = batch_to_device(batch, args.device, exclude = ["beta", "gamma"])

                # zero the parameter gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss, loss_details, logits = model(cluster_result = cluster_result, **batch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                elif phase == 'val':
                    with torch.no_grad():
                        # Only generate CRE logits
                        loss, loss_details, logits = model(cluster_result = None, **batch)
                else:
                    raise ValueError

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_labels, _ = model.predict_labels(logits)
                similarity_corrects += torch.sum(pred_labels == batch["labels"])

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

            history_logger.direct_log({ phase + '_epoch_loss'     : epoch_loss,
                                        phase + '_similarity_acc' : cur_acc} )

            if history_logger.update_history_dict(phase, epoch_loss, cur_acc, ckpt_dir):
                # best_loss = epoch_loss
                save_checkpoint(model, -1, ckpt_dir, args)
                best_model = copy.deepcopy(model)
                # torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_best.bin'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(history_logger.best_loss)))

    # # load best model weights
    # print("Best val loss from epoch:", np.argmin(train_history['val']['loss']))
    # model.load_state_dict(best_model_wts)

    return best_model, model, train_history

def compute_features(concrete_graph, id2concept, dataloader, model, args):
    all_train_ids = list(concrete_graph.nodes)
    model.eval()

    features = torch.zeros(len(id2concept), model.config.hidden_size).cuda()
    doc_count = torch.zeros(len(id2concept)).cuda()
    for step, batch in enumerate(tqdm(dataloader, desc = "Compute Features")):
        srs_input_ids, srs_seg_masks, srs_att_masks, srs_ids, tgt_input_ids, tgt_seg_masks, tgt_att_masks, tgt_ids, labels = batch

        batch = {
            "srs_input_ids" : srs_input_ids, 
            "srs_seg_masks" : srs_seg_masks, 
            "srs_att_masks" : srs_att_masks, 
            "srs_ids" : srs_ids,
            "tgt_input_ids" : tgt_input_ids, 
            "tgt_seg_masks" : tgt_seg_masks, 
            "tgt_att_masks" : tgt_att_masks, 
            "tgt_ids" : tgt_ids,
            "labels"  : labels,
        }

        batch = batch_to_device(batch, "cuda")

        with torch.no_grad():
            side_by_side_input_ids = torch.cat([batch['srs_input_ids'], batch['tgt_input_ids']], dim = -1)
            srs_tgt_features = model.get_doc_embedding(side_by_side_input_ids, mode = 'q')
            srs_cls = srs_tgt_features['last_hidden_states'][:, 0, 0]
            tgt_cls = srs_tgt_features['last_hidden_states'][:, 0, model.yaml_config["DATASET"]["MAX_SEG_LEN"]]

            srs_embed = model.mlp_q(srs_cls)
            tgt_embed = model.mlp_q(tgt_cls)

            features[srs_ids, :] = srs_embed
            doc_count[srs_ids] += 1
            features[tgt_ids, :] = tgt_embed
            doc_count[tgt_ids] += 1

    features /= doc_count.unsqueeze(-1)

    return features.cpu()

def update_centroids(x, all_connected_components, concept2cluster, args):
    """
    Args:
        x: data to be clustered
    """

    centroids = []
    Dcluster = []
    for cc in all_connected_components:
        cc_feat = x[list(cc), :]
        centroid = torch.mean(cc_feat, dim=0)
        centroids.append(centroid.unsqueeze(0))

        dist = (cc_feat - centroid).pow(2).sum(1).sqrt()
        dist = dist.cpu().tolist()
        Dcluster.append(dist)

    # concentration estimation (phi)
    density = np.zeros(len(all_connected_components))
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist)**0.5).mean() / np.log(len(dist)+10)
            density[i] = d
            
    #if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10), np.percentile(density,90)) #clamp extreme values for stability
    density = args.temperature * density / density.mean()  #scale the mean to temperature 

    concept2cluster = torch.LongTensor(concept2cluster).cuda()
    centroids = torch.cat(centroids, dim=0).cuda()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    density = torch.Tensor(density).cuda()

    results = { 'concept2cluster'   : concept2cluster,
                'centroids'         : centroids,
                'density'           : density}

    return results

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
        ckpt_name += "_segs@{}x{}".format(yaml_config['DATASET']['MAX_SEG_NUM'], yaml_config['DATASET']['MAX_SEG_LEN'])

        if args.use_aug:
            ckpt_name += "_aug@" + str(args.aug_ratio) if args.aug_ratio is not None else "n"
            ckpt_name += f"+{args.k_hops}hop"
            ckpt_name += "+" + args.score_estimator
            ckpt_name += "+ft" + (str(args.score_threshold) if args.score_threshold is not None else "n")

        if args.use_gccl:
            ckpt_name += "_gccl"

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
        srs_input_ids, srs_seg_masks, srs_att_masks, srs_ids, tgt_input_ids, tgt_seg_masks, tgt_att_masks, tgt_ids, labels = batch
        batch = {
            "srs_input_ids" : srs_input_ids, 
            "srs_seg_masks" : srs_seg_masks, 
            "srs_att_masks" : srs_att_masks, 
            "srs_ids" : srs_ids,
            "tgt_input_ids" : tgt_input_ids, 
            "tgt_seg_masks" : tgt_seg_masks, 
            "tgt_att_masks" : tgt_att_masks, 
            "tgt_ids" : tgt_ids,
            "labels"  : labels,
        }

        batch = batch_to_device(batch, device)

        with torch.no_grad():
            _, _, logits = model(cluster_result = None, **batch)
            cur_pred_labels, cur_pred_probs = model.predict_labels(logits)

        pred_labels += cur_pred_labels.detach().to("cpu").tolist()
        pred_probs.append(cur_pred_probs.cpu().numpy())

    pred_probs = np.concatenate(pred_probs, axis = 0)
    return pred_labels, pred_probs

def get_true_labels(dataset, n_classes = 2):
    labels = []
    one_hot_labels = []
    for ex in dataset:
        cur_label = ex[-1].item()
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    true_labels, true_one_hot = get_true_labels(dataset)
    pred_labels, pred_probs = model_predict(model, dataloader, device)
    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    precision = sklearn.metrics.precision_score(true_labels, pred_labels)
    recall = sklearn.metrics.recall_score(true_labels, pred_labels)
    f1_score = sklearn.metrics.f1_score(true_labels, pred_labels)
    specificity = specificity_score(true_labels, pred_labels)
    auc = sklearn.metrics.roc_auc_score(true_labels, pred_probs)

    print("-------------- Test model --------------")
    print('Accuracy = {} ;\nPrecision = {} ;\nRecall/Sensitivity = {} ;\nF1 score = {} ;\nSpecificity = {} ;\nAUC = {}'.format(acc, precision, recall, f1_score, specificity, auc))
    print('Latex table row: {}\% & {}\% & {}\% & {}\% & {}\% & {}\%'.format(round(acc * 100., 2), round(precision * 100., 2), round(recall * 100., 2), round(specificity * 100., 2), round(f1_score * 100., 2), round(auc * 100., 2)))
