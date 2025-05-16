#%%
# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import softmax
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm
from biotite.structure.io import pdb
from biotite.structure import to_sequence
from biotite.structure.io import pdb, xtc

from rocketshp import config, RocketSHP, load_sequence, load_structure
from rocketshp.trajectory import (
    compute_rmsf,
    compute_generalized_correlation_lmi,
    compute_contacts,
    compute_shp,
)
from rocketshp.plot import plot_predictions

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

#%% Load KRAS data

kras_pdb_file = config.RAW_DATA_DIR / "KRAS_DMS/kras_afdb.pdb"
kras_top_file = config.RAW_DATA_DIR / "KRAS_DMS/kras_afdb_100ns/kras_afdb_100ns_top.pdb"
kras_traj_file = config.RAW_DATA_DIR / "KRAS_DMS/kras_afdb_100ns/kras_afdb_100ns_traj.xtc"

kras_struct = pdb.PDBFile.read(kras_pdb_file).get_structure()
kras_seq = str(to_sequence(kras_struct)[0][0])
kras_traj = md.load(kras_traj_file, top=kras_top_file)

kras_bs_top = pdb.PDBFile.read(kras_top_file).get_structure()
kras_bs_xtc = xtc.XTCFile.read(kras_traj_file).get_structure(kras_bs_top)

#%% Make model predictions
# Set compute device
device = torch.device("cuda:0")

# Load the model
model = RocketSHP.load_from_checkpoint("latest").to(device)

struct_features = load_structure(kras_struct, device=device)
seq_features = load_sequence(kras_seq, device=device)

with torch.no_grad():
    dynamics_pred = model({
        "seq_feats": seq_features,
        "struct_feats": struct_features,
    })

pred_rmsf = dynamics_pred["rmsf"].squeeze().cpu().numpy()
pred_gcc = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
pred_ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()
pred_shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()

#%% Compute simulation derivatives

true_rmsf = compute_rmsf(kras_traj)
true_gcc = compute_generalized_correlation_lmi(kras_top_file, kras_traj_file)
true_ca_dist = compute_contacts(kras_traj, ca_only=True)
true_shp = compute_shp(kras_bs_xtc)

# %% Plot true

plot_predictions(
    true_rmsf,
    true_gcc,
    true_shp,
    title = "KRAS True",
    output_path = config.REPORTS_DIR / "kras" / "kras_true.png",
    font_scale = 1.0,
)

# %% Plot predicted

plot_predictions(
    pred_rmsf,
    pred_gcc,
    pred_shp,
    title = "KRAS Predicted",
    output_path = config.REPORTS_DIR / "kras" / "kras_predicted.png",
    font_scale = 1.0,
)

# %% Load KRAS clusters

with open(config.REPORTS_DIR / "large_model_20250427" / "kras_gcc_lmi_clusters.txt" ,"r") as f:
    clusters = f.readlines()
clusters = [list(map(int, c.split())) for c in clusters]

def res_list_to_span(res_list):
    # contiguous residues should be dashed and groups comma separated i.e 1-10,15-25,30-42
    res_list = sorted(res_list)
    result = [res_list[0]]
    range_start = res_list[0]
    last = res_list[0]
    for i in res_list[1:]:
        if int(i) - last == 1:
            result[-1] = (range_start, i)
            last = i
        else:
            range_start = i
            last = i
            result.append(i)
    # rstring = " or ".join(f"(resi {r})" for r in result)
    # return rstring
    for r in result:
        if isinstance(r, tuple):
            result[result.index(r)] = (r[0], r[1]+1)
        else:
            result[result.index(r)] = (r, r+1)
    return result

# %% Plot each side by side

COLOR_LIST = [
    (0.55, 0.25, 0.69),
    (0.3, 0.3, 1.0),
    (0.1, 0.6, 0.6),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    ]
REGION_LIST = ["Core", "Pocket", "Helices", "Loop", "CTT"]

with plt.style.context({
    "font.size": 30,
    "legend.fontsize": 24,
    "axes.labelsize": 30,
    }):

    fig, ax = plt.subplots(figsize=(18,6))
    ax.plot(true_rmsf, label="True RMSF", color="gray", linestyle="--")
    ax.plot(pred_rmsf, label="Predicted RMSF", color="black", linestyle="-")
    ax.set_xlabel("Residue Index")

    for c, region, color in zip(clusters, REGION_LIST, COLOR_LIST):
        for i, span in enumerate(res_list_to_span(c)):
            if i == 0:
                ax.axvspan(span[0], span[1],
                           color=color,
                           alpha=0.3, label=region)
            else:
                ax.axvspan(span[0], span[1],
                           color=color,
                           alpha=0.3)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    plt.savefig(config.REPORTS_DIR / "kras" / "kras_rmsf_comparison.svg")
    plt.show()

with plt.style.context({
    "font.size": 24,
    "legend.fontsize": 24,
    "axes.labelsize": 24,
    }):
    fig, ax = plt.subplots(1, 2, figsize=(12,8), sharey=True)
    ax[0].imshow(true_gcc, cmap="coolwarm", aspect="equal", vmin=0, vmax=1)
    ax[0].set_title("True GCC-LMI")
    ax[0].set_xlabel("Residue Index")
    ax[0].set_ylabel("Residue Index")
    im2 = ax[1].imshow(pred_gcc, cmap="coolwarm", aspect="equal", vmin=0, vmax=1)
    ax[1].set_title("Predicted GCC-LMI")
    ax[1].set_xlabel("Residue Index")
    ax[1].set_ylabel("Residue Index")

    LEN = len(true_gcc)
    for c, region, color in zip(clusters, REGION_LIST, COLOR_LIST):
        for i, span in enumerate(res_list_to_span(c)):
            ax[0].axvspan(span[0], span[1], ymin=(LEN-span[0]) / LEN, ymax=(LEN-span[1]) / LEN,
                        color=color, alpha=0.3)
            ax[1].axvspan(span[0], span[1], ymin=(LEN-span[0]) / LEN, ymax=(LEN-span[1]) / LEN,
                        color=color, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)  # Make space for colorbar on right
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    plt.savefig(config.REPORTS_DIR / "kras" / "kras_gcc_comparison.svg")
    plt.show()


with plt.style.context({
    "font.size": 24,
    "legend.fontsize": 24,
    "axes.labelsize": 24,
    }):
    fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)
    ax[0].imshow(true_shp.T, cmap="binary", vmin=0, vmax=1)
    ax[0].set_title("True SHP")
    ax[0].set_ylabel("Structure\nToken")
    ax[0].set_ylim(21, -1)
    ax[1].imshow(pred_shp.T, cmap="binary", vmin=0, vmax=1)
    ax[1].set_title("Predicted SHP")
    ax[1].set_xlabel("Residue Index")
    ax[1].set_ylabel("Structure\nToken")
    ax[1].set_ylim(21, -1)
    
    for c, region, color in zip(clusters, REGION_LIST, COLOR_LIST):
        for i, span in enumerate(res_list_to_span(c)):
            ax[0].axvspan(span[0], span[1], 
                        color=color, alpha=0.3)
            ax[1].axvspan(span[0], span[1],
                        color=color, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "kras" / "kras_shp_comparison.svg")
    plt.show()
    plt.close("all")



# %%
