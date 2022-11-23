from __future__ import print_function, division

# ---------- Basic dependencies ----------
import os
import argparse
import json
import yaml
import pickle
import copy
import builtins
import random
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib import font_manager

font_dir = ['./code/fonts/lucida-fax']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# ---------- Config ----------
# from configs.config import MAX_SEQ_LEN, SCORE_THRESHOLD
# from configs.word_config import *
from models.ctrs_transformer import CtrsTransformer, WORD_MODEL_CLASSES, ConcreteTransformer
from utils import HistoryLogger, train_model, load_checkpoint, evaluate_model, fix_random_seed, compute_features, update_centroids
from data import get_concrete_graph, graph_augment_dataset, read_word_data, read_cns_data, load_dataset, get_connected_components

DEBUG_SIZE = None

# ====================== Configure before run ======================
NUM_COMPONENTS = 200
OVERWRITE_FEAT_CACHE = False
TITLE_NAME = "GCCL Representations"
# TITLE_NAME = "Original Representations"
# ====================== Configure before run ======================

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# # extract x and y coordinates representing the positions of the images on T-SNE plot
# tx = tsne[:, 0]
# ty = tsne[:, 1]

# tx = scale_to_01_range(tx)
# ty = scale_to_01_range(ty)

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

    # Distributed training settings
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
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
    # parser.add_argument('--do_train',
    #                 action = 'store_true',
    #                 help = 'Whether do training')
    parser.add_argument('--eval',
                    action = 'store_true',
                    help = 'Whether do evaluation during training')
    # parser.add_argument('--split_seed',
    #                 type = int, default = 42,
    #                 help = 'Random seed for new dataset split')
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
    # parser.add_argument('--use_comm',
    #                 action = 'store_true',
    #                 help = 'Use commutativity')
    # parser.add_argument('--use_trans',
    #                 action = 'store_true',
    #                 help = 'Use transitivity')
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
    # parser.add_argument('--filter_length_range',
    #                 nargs=2,
    #                 type = int, default = [0, -1],
    #                 help = 'Filter using the length of the path; '
    #                         'min is non-negative; '
    #                         'no upper bound if max == -1, '
    #                         'otherwise, max is positive and max >= min')
    # parser.add_argument('--edge_weight_mapping',
    #                 type = str, default = 'linear',
    #                 choices=['linear', 'recip', 'poly'],
    #                 help = 'The estimator for the score between a new pair from transitivity')
    # parser.add_argument('--success_rate_threshold',
    #                 type = float, default = 0.01,
    #                 help = 'The sampling stops when the success rate falls below this threshold')
    # parser.add_argument('--shuffle_sample_pairs',
    #                 action = "store_true",
    #                 help = 'Whether randomly sample concept pairs')

    parser.add_argument('--use_gccl',
                    action = 'store_true',
                    help = 'Use prototypical contrastive learning')

    args = parser.parse_args()

    args.title = TITLE_NAME
    if NUM_COMPONENTS is not None:
        args.title += f", {NUM_COMPONENTS} Components"
    title_lower = "_".join(TITLE_NAME.lower().split(" "))
    spec_name = "_".join([args.dataset, args.model, title_lower])
    args.out_file = spec_name + ".pdf"
    args.feature_file = spec_name + ".pkl"

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
        # model, ckpt_path, train_history = load_checkpoint(model, args, yaml_config)
        # model.to(device)

        train_meta, val_meta, test_meta, id2concept = read_cns_data(args, yaml_config, tokenizer)

    else:
        raise NotImplementedError

    model, ckpt_path, train_history = load_checkpoint(model, args, yaml_config, device)
    model.to(device)

    concept_count = [0] * len(id2concept)
    for meta in train_meta:
        for id in meta['id_pair']:
            concept_count[id] += 1
        # srs_id, tgt_id = meta['id_pair'][0], meta['id_pair'][1]

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
    # optimizer_ft = torch.optim.AdamW(
    #     [
    #         {"params":model.encoder_q.parameters(), "lr": 1e-5},
    #         # {"params":model.encoder_k.parameters(), "lr": 1e-5},
    #         {"params":model.mlp_q.parameters(),  "lr": 1e-3},
    #     ])
    all_connected_components, concept2cluster = get_connected_components(concrete_graph, id2concept)

    if not os.path.isfile(args.feature_file) or OVERWRITE_FEAT_CACHE:
        features = compute_features(concrete_graph, id2concept, dataloaders_dict['train'], model, args)
        with open(args.feature_file, "wb") as feat_file:
            pickle.dump(features, feat_file)
        print("Features Cached into:", args.feature_file)
    else:
        with open(args.feature_file, "rb") as feat_file:
            features  = pickle.load(feat_file)

    # cluster_result = update_centroids(features, all_connected_components, concept2cluster, args)

    # cluster_result = { 'concept2cluster'   : concept2cluster,
    #                     'centroids'         : centroids,
    #                     'density'           : density}

    features = features.cpu().numpy()
    # train_features = features[cluster_result['concept2cluster'] != -1, :].cpu().numpy()
    # train_features = scale_to_01_range(train_features)
    # train_concept2cluster = cluster_result['concept2cluster'][cluster_result['concept2cluster'] != -1]
    # latent_var_2d = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(train_features)

    # cluster_num = cluster_result['centroids'].shape[0]
    print("Number of clusters", len(all_connected_components))

    # cluster2feat = {}
    # for i in tqdm(range(train_features.shape[0])):
    #     cluster_idx = train_concept2cluster[i]
    #     if cluster_idx in cluster2feat:
    #         cluster2feat[cluster_idx].append(train_features[i])
    #     else:
    #         cluster2feat[cluster_idx] = [train_features[i]]
    # print(cluster2feat)
    # cluster2feat.sort(key = len)

    cc2feat = {}
    for id, cc in enumerate(all_connected_components):
        cc_indices = list(cc)
        cc_feats = features[cc_indices, :]
        # for concept_id in cc:
        #     cc_feats.append(features[concept_id, :])
        cc2feat[id] = cc_feats

    sort_orders = sorted([(cc_id, cc2feat[cc_id]) for cc_id in cc2feat], key=lambda x: x[1].shape[0], reverse=True)
    sorted_cc_ids = [cc_feat[0] for cc_feat in sort_orders]

    feat2cc = []
    large_cc_feats = []
    if NUM_COMPONENTS is not None:
        sorted_cc_ids = sorted_cc_ids[:NUM_COMPONENTS]

    for color_id, cc_id in enumerate(sorted_cc_ids):
        large_cc_feats.append(cc2feat[cc_id])
        feat2cc += [color_id] * cc2feat[cc_id].shape[0]
    
    large_cc_feats = np.concatenate(large_cc_feats, axis = 0)

    doc_embed_2d = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(large_cc_feats)
    tx = doc_embed_2d[:, 0]
    ty = doc_embed_2d[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    dot_size = 7.5
    plt.rcParams["figure.figsize"] = (16, 9)
    # plt.title("Original Representations", fontsize=35, fontname="Lucida Fax", y=1.01)
    # out_file = 'original_embeddings.pdf'

    plt.title(args.title, fontsize=35, fontname="Lucida Fax", y=1.01)
    

    plt.tick_params(
        # axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False, 
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False, 
        labelbottom=False) # labels along the bottom edge are off

    # plt.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)

    # plt.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    unique_colors = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
        ]

    large_cc_colors = []
    for name in ["Set3", "Set2", "Set1", "Dark2", "Accent", "Paired", "Pastel1", "Pastel2", "Accent", "tab10", "tab20", "tab20b", "tab20c"]:
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        large_cc_colors += cmap.colors  # type: list
        # axes.set_prop_cycle(color=colors)

    # colors = cm.rainbow(np.linspace(0, 1, len(sorted_cc_ids)))
    small_cc_colors = cm.rainbow(np.linspace(0, 1, len(sorted_cc_ids) - len(large_cc_colors) ))
    random.shuffle(small_cc_colors)

    for i in tqdm(range(large_cc_feats.shape[0])):
        color_id = feat2cc[i]
        # color = colors[color_id]

        if color_id < len(large_cc_colors):
            color = large_cc_colors[color_id]
        else:
            color = small_cc_colors[color_id - len(large_cc_colors), :]
            # color = unique_colors[0] # Black
        # if color_id < len(unique_colors) - 1:
        #     color = unique_colors[color_id + 1]
        # # else:
        # #     color = unique_colors[0] # Black
        #     plt.scatter(tx[i], ty[i], s = dot_size, color = color)

        plt.scatter(tx[i], ty[i], s = dot_size, color = color)

        # cluster_idx = train_concept2cluster[i]
        # if cluster_idx in sorted_cc_ids[:num_cluster_to_show]:
        #     feat = train_features[i]
        #     color_idx = sorted_cc_ids[:num_cluster_to_show].index(cluster_idx)
        #     color = colors[color_idx]
        #     plt.scatter(latent_var_2d[i, 0], latent_var_2d[i, 1], s = dot_size, color = color)

    plt.savefig(args.out_file)
    print("Image saved to:", args.out_file)
