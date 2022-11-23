# ---------- Basic dependencies ----------
from tqdm import tqdm
import os
import pickle
import numpy as np
import json
import argparse
from collections import defaultdict
import random
from statistics import mean

# ---------- Model dependencies ----------
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager
# Add every font at the specified location
font_dir = ['./fonts/Merriweather']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# ---------- local ----------
from data import *
# from config import *
import yaml

with open("code/configs/config.yaml") as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)


def get_concrete_graph(meta_list, use_commutativity = True,
                                use_transitivity = True,
                                trans_aug_ratio = None,
                                edge_weight_mapping = 'linear',
                                score_estimator = 'mean',
                                filter_threshold = None,
                                filter_length_range = [0, -1],
                                success_rate_threshold = 0.01):

    doc_graph = nx.Graph()
    for meta in meta_list:
        doc_s, doc_t = meta['doc_pair'][0], meta['doc_pair'][1]
        # if meta['score'] >= filter_threshold:
        if meta['score'] > yaml_config['MODEL']['SCORE_THRESHOLD']:
            edge_weight_dict = {
                "linear" : 1. - meta['score'],
                "recip"  : 1. / meta['score'] - 1,
                "poly"   : (1. - meta['score']) ** 2,
            }
            # edge_weight = 1. - meta['score']
            # edge_weight = 1. / meta['score'] - 1
            # edge_weight = (1. - meta['score']) ** 2
            if not filter_node(doc_s) and not filter_node(doc_t):
                edge_weight = edge_weight_dict[edge_weight_mapping]
                doc_graph.add_edge(doc_s, doc_t, weight=edge_weight, score=meta['score'])
        else:
            doc_graph.add_node(doc_s)
            doc_graph.add_node(doc_t)

    # all_connected_components = sorted(nx.connected_components(doc_graph), key=len, reverse=False)

    return doc_graph

def filter_node(node):
    filter_list = ['porn', 'sex', 'economic,']
    for filter_keyword in filter_list:
        if filter_keyword in node.lower():
            return True
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description ='Evaluate saved model')
    parser.add_argument('--model', dest = 'model', action = 'store',
                    default = 'bert',
                    help = 'Choose the model among: bert, roberta, xlnet',
                    choices = [''])
    parser.add_argument('--trans_aug_ratio', dest = 'trans_aug_ratio',
                    type = float, default = None,
                    help = 'Augmentation ratio using transitivity; effective only when it is > 1')
    parser.add_argument('--filter_threshold',
                    type = float, default = None,
                    help = 'Threshold for creating strong clusters in data augmentation')
    parser.add_argument('--filter_length_range',
                    nargs=2,
                    type = int, default = [0, -1],
                    help = 'Filter using the length of the path; '
                            'min is non-negative; '
                            'no upper bound if max == -1, '
                            'otherwise, max is positive and max >= min')
    parser.add_argument('--score_estimator',
                    type = str, default = 'mean',
                    choices=['mean', 'min', 'prod'],
                    help = 'The estimator for the score between a new pair from transitivity')
    parser.add_argument('--edge_weight_mapping',
                    type = str, default = 'linear',
                    choices=['linear', 'recip', 'poly'],
                    help = 'The estimator for the score between a new pair from transitivity')
    parser.add_argument('--success_rate_threshold',
                    type = float, default = 0.01,
                    help = 'The sampling stops when the success rate falls below this threshold')

    args = parser.parse_args()

    with open(os.path.join(yaml_config['DATASET']['WORD']['RAW_DATA_DIR'], yaml_config['DATASET']['WORD']['META_DATA_NAME']), "r") as meta_file:
        meta_data_list = json.load(meta_file)

    pos_count = 0
    for meta in meta_data_list:
        if meta['score'] > yaml_config['MODEL']['SCORE_THRESHOLD']:
            pos_count += 1
    print("Positive ratio: {} / {}".format(pos_count, len(meta_data_list)))

    train_meta, test_meta = official_train_test_split(meta_data_list)
    train_meta, val_meta = train_test_split(train_meta,
                                            test_size=0.1,
                                            random_state=42)

    # print(len(train_meta))
    # print(len(val_meta))
    # print(len(test_meta))

    # Create training set: Step 1 & 2
    # Step 1: Data augmentation (Clustering & Truncation)
    concrete_graph = get_concrete_graph(train_meta, \
                                use_commutativity = True,
                                use_transitivity = True,
                                trans_aug_ratio = args.trans_aug_ratio,
                                score_estimator = args.score_estimator,
                                edge_weight_mapping = args.edge_weight_mapping,
                                filter_threshold = args.filter_threshold,
                                filter_length_range = args.filter_length_range,
                                success_rate_threshold = args.success_rate_threshold,
                                )

    # make new undirected graph H without multi-edges

    node_degrees = sorted(concrete_graph.degree, key=lambda x: x[1], reverse=True)
    hub_nodes = [node for node, degree in node_degrees]
    # print(hub_nodes)
    # central_node = hub_nodes[5]
    central_node = "Open-source_software"
    k_hop_neighbors = nx.single_source_shortest_path_length(concrete_graph, central_node, cutoff=4)
    # print(k_hop_neighbors)
    # show_nodes = list(k_hop_neighbors.keys())[:25]
    show_nodes = []
    node_degrees = {}
    for neighbor in k_hop_neighbors.keys():
        # if concrete_graph.degree[neighbor] > 10:
        show_nodes.append(neighbor)
        node_degrees[neighbor] = concrete_graph.degree[neighbor]
    # print(show_nodes)
    H = concrete_graph.subgraph(show_nodes)
    # print(H.nodes())
    # print()
    # H = nx.Graph(concrete_graph)

    # edge width is proportional number of games played
    # edgewidth = [len(concrete_graph.get_edge_data(u, v)) for u, v in H.edges()]
    edgewidth = [1 for u, v in H.edges()]
    # print(edgewidth)

    # node size is proportional to number of games won
    # rel_counts = dict.fromkeys(concrete_graph.nodes(), 0.0)
    # for (u, v, d) in concrete_graph.edges(data=True):
    #     # rel_counts[u] += d["score"]
    #     # rel_counts[v] += d["score"]
    #     rel_counts[u] += 1
    #     rel_counts[v] += 1

    nodesize = [node_degrees[v] * 5 for v in H]

    # Generate layout for visualization
    pos = nx.nx_pydot.graphviz_layout(H, prog="neato")
    # # Manual tweaking to limit node label overlap in the visualization
    # pos["Reshevsky, Samuel H"] += (0.05, -0.10)
    # pos["Botvinnik, Mikhail M"] += (0.03, -0.06)
    # pos["Smyslov, Vassily V"] += (0.05, -0.03)

    fig, ax = plt.subplots(figsize=(12, 12))
    # Visualize graph components
    nx.draw_networkx_edges(H, pos, alpha=0.3, width=edgewidth, edge_color="#f37021")
    nx.draw_networkx_nodes(H, pos, node_size=nodesize, node_color="#0ABAB5", alpha=0.9)

    labels = {}
    for node in H.nodes():
        if concrete_graph.degree[node] > 3:
            #set the node name as the key and the label as its value
            node_label = node.replace('_', ' ')
            labels[node] = node_label
    label_options = {"ec": "k",
                    "fc": "white",
                    "alpha": 0.3,
                    "boxstyle": "round",
                    "linewidth": 0.5,}
    nx.draw_networkx_labels(H, pos, labels, font_size=5, font_family="Merriweather", bbox=label_options)

    # Title/legend
    # font = {"fontname": "Merriweather", "color": "k", "fontweight": "bold", "fontsize": 14}
    # ax.set_title("Central Node: "+central_node, font)
    # # Change font color for legend
    # font["color"] = "r"

    # ax.text(
    #     0.80,
    #     0.10,
    #     "edge width = # games played",
    #     horizontalalignment="center",
    #     transform=ax.transAxes,
    #     fontdict=font,
    # )
    # ax.text(
    #     0.80,
    #     0.06,
    #     "node size = # games won",
    #     horizontalalignment="center",
    #     transform=ax.transAxes,
    #     fontdict=font,
    # )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()
