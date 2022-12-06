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
from sklearn.model_selection import train_test_split
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
from data import WholeDocPairDataset, load_word_for_bert, augment_dataset, train_val_test_split, official_train_test_split, collate_doc_pair
from eval import evaluate_model
# from eval import evaluate_accuracy

def train_model(model, dataloaders_dict, dataset_sizes, criterion, optimizer, scheduler, device, ckpt_dir, start_epoch=0, num_epochs=5, train_history = None):
    since = time.time()

    if train_history is None:
        train_history = {
                'train' : {
                    'loss': [],
                    'acc' : [],
                },
                'val'   : {
                    'loss': [],
                    'acc' : [],
                },
            }
        best_loss = 100
    else:
        best_loss = train_history['val']['loss'][start_epoch-1]

    print('Start training with best loss:', best_loss)
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            similarity_corrects = 0
            # Iterate over data.
            for padded_batch_s, padded_batch_t, sent_len_s_t, labels in tqdm(dataloaders_dict[phase]):

                padded_batch_s = padded_batch_s.to(device)
                padded_batch_t = padded_batch_t.to(device)
                sent_len_s_t = sent_len_s_t.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # use mixed precision, combined single-precision (FP32) with half-precision (e.g. FP16)
                    # with torch.cuda.amp.autocast():
                    outputs = model(padded_batch_s, padded_batch_t, sent_len_s_t, device)
                    # outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, labels.type_as(outputs)[:, None])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                pred_labels = (outputs[:, 0] > 0) * 1
                # print(pred_labels, labels)
                similarity_corrects += torch.sum(pred_labels == labels)

                # similarity_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

                # group_idx, param_idx = 0, 0
                # current_lr = get_current_lr(optimizer, group_idx, param_idx)
                # print('Current learning rate (g:%d, p:%d): %.4f'%(group_idx, param_idx, current_lr))

            epoch_loss = running_loss / dataset_sizes[phase]
            total_acc = similarity_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {} '.format(phase, epoch_loss))
            print('{} total acc: {}'.format(phase, total_acc))

            train_history[phase]['loss'].append(epoch_loss)
            train_history[phase]['acc'].append(total_acc.item())

            if phase == 'val':
                ckpt_path = os.path.join(ckpt_dir, 'model_epoch_{}.pth'.format(epoch + 1))
                torch.save(model.state_dict(), ckpt_path)

                with open(os.path.join(ckpt_dir, HISTORY_NAME), "w") as history_file:
                    json.dump(train_history, history_file)

                if epoch_loss < best_loss:
                    print('saving with loss of {}'.format(epoch_loss),
                          'improved over previous {}'.format(best_loss))
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_best.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(best_loss)))

    # load best model weights
    print("Best val loss from epoch:", np.argmin(train_history['val']['loss']))
    model.load_state_dict(best_model_wts)
    return model, train_history


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluate saved model')
    parser.add_argument('-m', '--model', dest = 'ckpt_name', action = 'store',
                    default = None,
                    help = 'Filename of the model checkpoint that is to be restored')
    parser.add_argument('--no_cuda', dest = 'use_gpu', action = 'store_false',
                    help = 'Use GPU or not')
    parser.add_argument('--ckpt_dir', dest = 'ckpt_dir', action = 'store',
                    default = DEFAULT_CKPT_DIR,
                    help = 'Overwrite the default checkpoint directory')
    # Model training settings
    parser.add_argument('--torch_seed', dest = 'torch_seed',
                    type = int, default = None,
                    help = 'Random seed for PyTorch')
    parser.add_argument('--new_split', dest = 'new_split',
                    action = 'store_true',
                    help = 'Generate a new split')
    parser.add_argument('--split_seed', dest = 'split_seed',
                    type = int, default = None,
                    help = 'Random seed for new dataset split')
    parser.add_argument('--max_len', dest ='max_len',
                    type = int, default = None,
                    help = 'Max input document length')
    parser.add_argument('--use_comm', dest = 'use_comm',
                    action = 'store_true',
                    help = 'Use commutativity')
    parser.add_argument('--use_trans', dest = 'use_trans',
                    action = 'store_true',
                    help = 'Use transitivity')
    parser.add_argument('--cluster_threshold', dest = 'cluster_threshold',
                    type = float, default = 0.9,
                    help = 'Threshold for creating strong clusters in data augmentation')

    args = parser.parse_args()

    if args.torch_seed is not None:
        print("For PyTorch, using random seed =", args.torch_seed)
        torch.manual_seed(args.torch_seed)

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
    if args.ckpt_name is None:
        print("Train from scratch!")
        start_epoch = 0
        if args.ckpt_dir == DEFAULT_CKPT_DIR:
            if args.torch_seed is not None:
                ckpt_dir = "ckpt_ts@{}".format(args.torch_seed)
            else:
                ckpt_dir = "ckpt_ts@n"

            if args.max_len is not None:
                ckpt_dir += "_len@{}".format(args.max_len)
            else:
                ckpt_dir += "_len@{}".format(MAX_SEQ_LEN)

            if args.use_comm:
                ckpt_dir += "_comm"

            if args.use_trans:
                ckpt_dir += "_trans@ct" + str(args.cluster_threshold)
        else:
            # Overwrite
            ckpt_dir = args.ckpt_dir

        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        print("Checkpoints will be saved into:", ckpt_dir)

    else:
        ckpt_path = args.ckpt_name
        if os.path.isfile(ckpt_path):
            print("Load model saved at checkpoint:", ckpt_path)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            ckpt_path_head, ckpt_path_tail = os.path.split(ckpt_path)
            if len(ckpt_path_tail) > 0:
                start_epoch = int(ckpt_path_tail.split(".")[0][len("model_epoch_"):])
                ckpt_dir = ckpt_path_head
                print("Checkpoints will continue to be saved into:", ckpt_dir)
                # load history
                with open(os.path.join(ckpt_dir, HISTORY_NAME), "r") as history_file:
                    train_history = json.load(history_file)
                train_loss = train_history['train']['loss'][start_epoch-1]
                val_loss = train_history['val']['loss'][start_epoch-1]
                print("At epoch {}: train loss = {} , val loss = {}".format(start_epoch, train_loss, val_loss))
            else:
                print("Checkpoint file does not exist:", ckpt_path)
                exit(1)
        else:
            print("Checkpoint file does not exist:", ckpt_path)
            exit(1)

    model.to(device)

    # ------------------ Data ------------------
    if args.new_split:
        with open(os.path.join(RAW_DATA_DIR, META_DATA_NAME), "r") as meta_file:
            meta_data_list = json.load(meta_file)

        if args.split_seed is not None:
            print("New split generated with random seed:", args.split_seed)
        else:
            print("New split generated randomly")
        train_meta, test_meta = official_train_test_split(meta_data_list)
        train_meta, val_meta = train_test_split(train_meta,
                                                test_size=0.1,
                                                random_state=42)
        # train_meta, val_meta, test_meta = train_val_test_split(meta_data_list,
        #                                                     split_sizes=[0.7, 0.2, 0.1],
        #                                                     random_seed = args.split_seed)

        with open(FIXED_SPLIT_FILE, "w") as split_file:
            json.dump([train_meta, val_meta, test_meta], split_file)
        print("Stored new split in file:", FIXED_SPLIT_FILE)
    else:
        with open(FIXED_SPLIT_FILE, "r") as split_file:
            train_meta, val_meta, test_meta = json.load(split_file)
        print("Loaded split from file:", FIXED_SPLIT_FILE)

    # Create training set: Step 1 & 2
    # Step 1: Data augmentation (Clustering & Truncation)
    augmented_train_meta = augment_dataset(train_meta, args.cluster_threshold,
                                            use_transitivity = args.use_trans,
                                            use_commutativity = args.use_comm)

    # Step 2: Create final training set by merging original set with the augmentation sets
    print("-------------- Load training, validation & test set --------------")
    # doc_s_list_cluster, doc_t_list_cluster, scores_cluster = load_word_for_bert(augmented_train_meta, "train", debug_size = None)
    # doc_s_list = doc_s_list_trunc + doc_s_list_cluster
    # doc_t_list = doc_t_list_trunc + doc_t_list_cluster
    # scores = scores_trunc + scores_cluster

    doc_s_list, doc_t_list, scores = load_word_for_bert(augmented_train_meta, debug_size = 100)
    print("Original training set size: {}; Final training set size: {}".format( len(train_meta), len(scores) ))
    train_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # Create val set
    doc_s_list, doc_t_list, scores = load_word_for_bert(val_meta, debug_size = 20)
    val_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    # Create test set
    doc_s_list, doc_t_list, scores = load_word_for_bert(test_meta, debug_size = 10)
    test_set = WholeDocPairDataset(doc_s_list, doc_t_list, scores, score_threshold = SCORE_THRESHOLD)

    dataloaders_dict = {'train' : torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_doc_pair),
                        'val'   : torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_doc_pair),
                        'test'  : torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_doc_pair)
                        }
    dataset_sizes = {   'train' : len(train_set),
                        'val'   : len(val_set),
                        'test'  : len(test_set)
                    }

    print(dataset_sizes)

    # ------------------ Train ------------------
    optimizer_ft = optim.Adam(
        [
            {"params":model.bert.parameters(),"lr": 1e-5},
            {"params":model.GCN_backbone.parameters(), "lr": 5e-2},
            {"params":model.fc1.parameters(), "lr": 5e-2},
        ])

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # Decay LR by a factor of 0.1 every 3 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    if start_epoch == 0:
        model_ft1, train_history = train_model(model, dataloaders_dict, dataset_sizes,
                                    criterion, optimizer_ft, None, device, ckpt_dir,
                                    num_epochs = NUM_EPOCHS)
    else:
        model_ft1, train_history = train_model(model, dataloaders_dict, dataset_sizes,
                                    criterion, optimizer_ft, None, device, ckpt_dir,
                                    start_epoch = start_epoch,
                                    num_epochs = NUM_EPOCHS,
                                    train_history = train_history)

    print("Training finished! Training history:")
    print(train_history)

    # ------------------ Test ------------------
    print("Test results for model in checkpoint:", ckpt_dir)
    evaluate_model(model, test_set, device)
