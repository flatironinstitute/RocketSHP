# %% Imports
import itertools
import pickle as pk

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from loguru import logger

from rocketshp import config
from rocketshp.network import display_network, pairwise_correlation_to_network

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation    "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 16,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% Parameters
# import argparse
# parser = argparse.ArgumentParser(description="Plot KRAS network with given model")
# parser.add_argument("eval_key", type=str, help="Evaluation key")

# args = parser.parse_args()
# EVAL_KEY = args.eval_key
# DIST_THRESH_ANGSTROM = 6.0

EVAL_KEY = "large_model_20250427"

# %% Load pickle
with open(config.REPORTS_DIR / EVAL_KEY / "mutant_results.pkl", "rb") as f:
    mutant_results = pk.load(f)
    logger.info(f"Loaded {len(mutant_results)} mutant results")
if mutant_results[0][0] == 0:
    wild_type_result = mutant_results[0]
    mutant_results = mutant_results[1:]  # remove the first element which is a dummy

# %% Process networks
DIST_THRESH_ANGSTROM = 8.0
DIST_THRESH_NM = DIST_THRESH_ANGSTROM / 10.0
NETWORK_PRED_THRESH = 0.5


gcc_network = wild_type_result[3]["gcc_lmi"].numpy()
pred_distance = wild_type_result[3]["ca_dist"].numpy()
mask = pred_distance < DIST_THRESH_NM
masked_net = gcc_network * mask
np.fill_diagonal(masked_net, 0)  # remove self-edges

# %% Cluster network
G = nx.from_numpy_array(masked_net)
comp = nx.community.girvan_newman(G)
k = 5
limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    clusts = tuple(sorted(c) for c in communities)
for c in clusts:
    print(" ".join([str(i) for i in c]))

# %% Write out clusters
with open(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_clusters.txt", "w") as f:
    for c in clusts:
        f.write(" ".join([str(i) for i in c]) + "\n")


# write pymol commands to color the clusters darkpurple, darkblue, deepteal, green, and yellow
def res_list_to_pymol(res_list):
    # contiguous residues should be dashed and groups comma separated i.e 1-10,15-25,30-42
    res_list = sorted(res_list)
    result = [res_list[0]]
    range_start = res_list[0]
    last = res_list[0]
    for i in res_list[1:]:
        if int(i) - last == 1:
            result[-1] = f"{range_start}-{i}"
            last = i
        else:
            range_start = i
            last = i
            result.append(i)
    rstring = " or ".join(f"(resi {r})" for r in result)
    return rstring


with open(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_clusters.pml", "w") as f:
    for i, c in enumerate(clusts):
        color = ["violetpurple", "tv_blue", "deepteal", "tv_green", "yellow"][i]
        f.write(f"color {color}, {res_list_to_pymol(c)}\n")
    f.write("\n")
    f.write("spectrum resi, firebrick_paleyellow_deepblue, chain A")

# %% Convert clusters to colors


# create a list of length number of nodes with colors for each cluster
def get_cluster_colors(clusters, num_nodes):
    colors = np.zeros(num_nodes)
    for i, cluster in enumerate(clusters):
        for node in cluster:
            colors[node] = i
    return colors


clust_colors = get_cluster_colors(clusts, masked_net.shape[0])

# %% Plot networks

NETWORK_PLOT_SEED = 0

fig, ax = plt.subplots(1, 3, figsize=(24, 10))
ax[0].imshow(gcc_network, cmap="coolwarm")
pairwise_correlation_to_network(
    masked_net, thresh=0.5, title="By Node Index", seed=NETWORK_PLOT_SEED, ax=ax[1]
)

# to color pymol chain: spectrum resi, firebrick_paleyellow_deepblue, chain A

display_network(
    masked_net,
    title="By Community",
    node_color=clust_colors,
    edge_color="gray",
    seed=NETWORK_PLOT_SEED,
    ax=ax[2],
)

# %% Plot each panel as a separate figure and save
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(gcc_network, cmap="coolwarm")
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi.svg")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
pairwise_correlation_to_network(
    masked_net, thresh=0.5, title="By Node Index", seed=NETWORK_PLOT_SEED, ax=ax
)
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_by_node_index.svg")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
display_network(
    masked_net,
    title="By Community",
    node_color=clust_colors,
    edge_color="gray",
    seed=NETWORK_PLOT_SEED,
    ax=ax,
)
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_by_community.svg")
plt.close("all")
# %%
