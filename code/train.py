from __future__ import print_function, division

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
from transformers import AutoTokenizer

# ---------- Ours ----------
from models.ctrs_transformer import CtrsTransformer, WORD_MODEL_CLASSES
from utils import train_model, load_checkpoint, evaluate_model, fix_random_seed
from data import get_concrete_graph, graph_augment_dataset, read_word_data, read_cns_data, load_dataset

DEBUG_SIZE = None


if __name__ == '__main__':

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
                    required = True,
                    type = str,
                    choices = ['bert', 'roberta', 'xlnet'],
                    help = 'Choose the model among: bert, roberta, xlnet')
    parser.add_argument('--dataset',
                    required = True,
                    help = 'Choose the dataset among: word, cnse, cnss',
                    choices = ['word', 'cnse', 'cnss'])
    parser.add_argument('--config',
                    default = "code/configs/config.yaml",
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

    # # Distributed training settings
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-addr', default='127.0.0.1', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-port', default='9990', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use.')
    # parser.add_argument('--server_ip', type=str, default='127.0.0.1', 
    #                     help="torch.distributed.init_process_group")
    # parser.add_argument('--server_port', type=str, default='9990', 
    #                     help="torch.distributed.init_process_group")

    # Model training settings
    parser.add_argument('--random_seed',
                    type = int, default = 42,
                    help = 'Random seed')
    parser.add_argument('--eval',
                    action = 'store_true',
                    help = 'Whether do evaluation during training')
    parser.add_argument('--logging_steps', type=int, default=125,
                        help="Log every X updates steps.")

    # Optimizer, scheduler
    parser.add_argument('--batch_size',
                    type = int, default = 1,
                    help = 'Batch size')
    parser.add_argument('--num_epochs',
                    type = int, default = 1,
                    help = 'Epoch number')
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='softmax temperature')

    # Data augmentation
    parser.add_argument('--use_aug',
                    action = 'store_true',
                    help = 'Use data augmentation')
    parser.add_argument('--aug_ratio',
                    type = float, default = None,
                    help = 'Augmentation ratio using transitivity; effective only when it is > 1')
    parser.add_argument('--score_estimator',
                    type = str, default = 'mean',
                    choices=['mean', 'min', 'prod'],
                    help = 'The estimator for the score between a new pair from transitivity')
    parser.add_argument('--score_threshold',
                    type = float, default = None,
                    help = 'Threshold for creating strong clusters in data augmentation')
    parser.add_argument('--k_hops',
                    default = 2,
                    type = int,
                    help = 'Use k-hop neighbors for data augmentation')

    # GCCL
    parser.add_argument('--use_gccl',
                    action = 'store_true',
                    help = 'Use prototypical contrastive learning')
    parser.add_argument('--beta',
                    type = float, default = 0.1,
                    help = 'GCCL loss coefficient')
    parser.add_argument('--gamma',
                    type = float, default = 0.1,
                    help = 'MoCo loss coefficient')

    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join("ckpts", args.dataset + "_" + args.model)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    args.dataset_cache_path = os.path.join(args.ckpt_dir, args.dataset_cache_path)

    if args.k_hops <= 1:
        args.k_hops = None

    fix_random_seed(args.random_seed)

    with open(args.config) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    # ------------------ Environment ------------------
    # if args.torch_seed is not None:
    #     logger.info("For PyTorch, using random seed = %d", args.torch_seed)
    #     torch.manual_seed(args.torch_seed)

    if args.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA is used")
        else:
            device = "cpu"
            logger.info("GPU not available, CPU is used")
    else:
        device = "cpu"
        logger.info("CPU is used")
    args.device = torch.device(device)

    # ngpus_per_node = torch.cuda.device_count()
    # # if args.dist_url == "env://" and args.world_size == -1:
    # #     args.world_size = int(os.environ["WORLD_SIZE"])
    # # # if args.dist_url == "env://" and args.rank == -1:
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    # # if args.multiprocessing_distributed:
    # #     # For multiprocessing distributed training, rank needs to be the
    # #     # global rank among all the processes
    # #     args.rank = args.rank * ngpus_per_node + args.gpu

    # dist.init_process_group(backend=args.dist_backend,
    #                         init_method=f"tcp://{args.server_ip}:{args.server_port}",
    #                         world_size=args.world_size,
    #                         rank=args.local_rank)

    # ------------------ Data ------------------
    if args.dataset == 'word':
        _, _, tokenizer_class = WORD_MODEL_CLASSES[args.model]
        tokenizer = tokenizer_class.from_pretrained(yaml_config['MODEL']['WORD'][args.model])
        model = CtrsTransformer(args = args, yaml_config = yaml_config)

        train_meta, val_meta, test_meta, id2concept = read_word_data(args, yaml_config, tokenizer, val_size = 0)

    elif args.dataset in ['cnse', 'cnss']:
        tokenizer = AutoTokenizer.from_pretrained(yaml_config['MODEL']['CNS'][args.model])
        model = CtrsTransformer(args = args, yaml_config = yaml_config)

        train_meta, val_meta, test_meta, id2concept = read_cns_data(args, yaml_config, tokenizer)

    else:
        raise NotImplementedError

    model, ckpt_path, train_history = load_checkpoint(model, args, yaml_config, device)
    model.to(device)

    # Create training set: Step 1 & 2
    # Step 1: Data augmentation (Clustering & Truncation)
    concrete_graph = get_concrete_graph(train_meta, yaml_config['DATASET']['SCORE_THRESHOLD'])

    if args.use_aug:
        augmented_train_meta = graph_augment_dataset(train_meta, concrete_graph, 
                                    score_threshold = yaml_config['DATASET']['SCORE_THRESHOLD'], 
                                    args = args)
    else:
        augmented_train_meta = train_meta

    # Step 2: Create final training set by merging original set with the augmentation sets
    # logger.info("-------------- Load training, validation & test set --------------")
    logger.info("Original training set size: {}; Final training set size: {}".format( len(train_meta), len(augmented_train_meta) ))
    # get_dataset_statistics(doc_s_list, doc_t_list, scores, tokenizer,
    #                                         score_threshold = yaml_config['MODEL']['SCORE_THRESHOLD'])

    train_set = load_dataset(augmented_train_meta, id2concept, 
                                    tokenizer,
                                    debug_size = DEBUG_SIZE, 
                                    yaml_config = yaml_config)

    val_meta = None
    if val_meta is None:
        logger.warning("Val meta data is None, using test meta as val meta")
        val_meta = test_meta

    val_set = load_dataset(val_meta, id2concept,
                                tokenizer,
                                debug_size = DEBUG_SIZE, 
                                yaml_config = yaml_config)

    test_set = load_dataset(test_meta, id2concept, 
                                    tokenizer,
                                    debug_size = DEBUG_SIZE, 
                                    yaml_config = yaml_config)

    dataloaders_dict = {'train' : torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
                        'val'   : torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
                        'test'  : torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
                        }
    dataset_sizes = {'train' : len(train_set),
                    'val'   : len(val_set),
                    'test'  : len(test_set)
                    }

    logger.info("Dataset sizes: %d / %d / %d", dataset_sizes['train'], dataset_sizes['val'], dataset_sizes['test'])

    # ------------------ Train ------------------
    if not args.eval:
        model, last_model, train_history = \
                        train_model(model, dataloaders_dict, dataset_sizes,
                                    ckpt_path,
                                    concrete_graph = concrete_graph,
                                    id2concept = id2concept,
                                    train_history = train_history, 
                                    args = args,
                                    yaml_config = yaml_config)

        logger.info("Training finished! Training history: %s", train_history)

        # ------------------ Test ------------------
        print("-------- last model --------")
        evaluate_model(last_model, test_set, device, args)

    logger.info("Test results for model in checkpoint: %s", ckpt_path)
    evaluate_model(model, test_set, device, args)