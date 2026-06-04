# %% Imports
import argparse
import json
import pickle as pk

import networkx as nx
import numpy as np
from loguru import logger

from rocketshp import config
from rocketshp.network import build_allosteric_network, cluster_network

# %% Parameters
parser = argparse.ArgumentParser(description="Build KRAS GCC-LMI network and cluster")
parser.add_argument("eval_key", type=str, nargs="?", default="large_model_20250427")
parser.add_argument(
    "--dist-thresh",
    type=float,
    default=8.0,
    help="Distance threshold in angstrom for masking [8.0]",
)
parser.add_argument(
    "--k-communities",
    type=int,
    default=5,
    help="Number of Girvan-Newman communities [5]",
)
args = parser.parse_args()

EVAL_KEY = args.eval_key
DIST_THRESH_ANGSTROM = args.dist_thresh
K_COMMUNITIES = args.k_communities

# %% Load pickle (wild-type only)
with open(config.REPORTS_DIR / EVAL_KEY / "mutant_results.pkl", "rb") as f:
    mutant_results = pk.load(f)
    logger.info(f"Loaded {len(mutant_results)} mutant results")

if mutant_results[0][0] == 0:
    wild_type_result = mutant_results[0]

gcc_network = wild_type_result[3]["gcc_lmi"].numpy()
ca_dist = wild_type_result[3]["ca_dist"].numpy()

# %% Build masked network and cluster
G = build_allosteric_network(gcc_network, ca_dist, distance_cutoff=DIST_THRESH_ANGSTROM)
masked_net = nx.to_numpy_array(G)

clusts = cluster_network(G, k=K_COMMUNITIES)
logger.info(f"Found {len(clusts)} communities")

# %% Save portable outputs
PRECOMPUTED_DIR = config.REPORTS_DIR / EVAL_KEY / "precomputed"
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    PRECOMPUTED_DIR / "kras_wt_masked_network.npz",
    masked_net=masked_net,
    gcc_network=gcc_network,
    dist_thresh_angstrom=np.float64(DIST_THRESH_ANGSTROM),
)
logger.info(
    f"Saved masked network NPZ: {PRECOMPUTED_DIR / 'kras_wt_masked_network.npz'}"
)

clusters_as_lists = [list(map(int, c)) for c in clusts]
with open(PRECOMPUTED_DIR / "kras_gcc_lmi_clusters.json", "w") as f:
    json.dump(clusters_as_lists, f)
logger.info(f"Saved cluster JSON: {PRECOMPUTED_DIR / 'kras_gcc_lmi_clusters.json'}")


# %% Write PyMOL/text outputs (for structural visualization)
def res_list_to_pymol(res_list):
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


with open(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_clusters.txt", "w") as f:
    for c in clusts:
        f.write(" ".join([str(i) for i in c]) + "\n")

with open(config.REPORTS_DIR / EVAL_KEY / "kras_gcc_lmi_clusters.pml", "w") as f:
    for i, c in enumerate(clusts):
        color = ["violetpurple", "tv_blue", "deepteal", "tv_green", "yellow"][i]
        f.write(f"color {color}, {res_list_to_pymol(c)}\n")
    f.write("\n")
    f.write("spectrum resi, firebrick_paleyellow_deepblue, chain A")
