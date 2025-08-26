# %% Imports
import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from rocketshp import config

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation    "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 30,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DIST_THRESH_ANGSTROM = 8.0
DIST_THRESH_NM = DIST_THRESH_ANGSTROM / 10.0

# %% Load h5file

logger.info(f"Loading DMS data from {config.RAW_DATA_DIR / 'na_channels/trpm8_dms.h5'}...")
h5fi = h5py.File(config.RAW_DATA_DIR / "na_channels/trpm8_dms.h5", "r")

sequence = h5fi["wt/sequence"][()].decode("utf-8")
SEQ_LENGTH = len(sequence)
logger.info(f"Sequence length: {SEQ_LENGTH}")

#%% Process wild type
logger.info("Processing wild type data...")
wt_results = {k: h5fi[f"wt/{k}"][:] for k in h5fi["wt"] if k != "sequence"}
wt_mask = (wt_results["ca_dist"] < DIST_THRESH_NM)
wt_masked_gcc = wt_results["gcc_lmi"] * wt_mask
np.fill_diagonal(wt_masked_gcc, 0)  # remove self-edges
wt_prot_graph = nx.from_numpy_array(wt_masked_gcc)
wt_centrality = nx.betweenness_centrality(wt_prot_graph, normalized=True)

# %% Plot wild type centrality and rmsf

logger.info("Plotting wild type centrality and RMSF...")
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

ax[0].plot(wt_results["rmsf"], label="Wild Type RMSF", color="blue")
ax[0].set_ylabel("RMSF (Å)")
ax[1].plot(wt_centrality.values(), label="Wild Type Centrality", color="green")
ax[1].set_ylabel("Betweenness Centrality\n(Normalized)")
ax[1].set_xlabel("Residue Index")
plt.savefig(config.REPORTS_DIR / "na_channel/wt_centrality_rmsf.png", dpi=300)

fig, ax = plt.subplots(figsize=(15, 12))
ax.set_title("Wild Type GCC LMI")
plt.imshow(wt_masked_gcc, interpolation='none', aspect='auto')
plt.colorbar()
plt.savefig(config.REPORTS_DIR / "na_channel/wt_gcc_lmi.png", dpi=300)

#%% Process mutants

logger.info("Processing mutants...")
dms_data = []
for i in tqdm(range(SEQ_LENGTH), desc=f"Processing mutants"):
    idx = i + 1
    alt_aa_list = list(h5fi[str(idx)].keys())
    pbar = tqdm(alt_aa_list, desc='Alternate AA', leave=False)
    for alt in pbar:
        pbar.set_description(f"Processing {idx} {alt}")
        mask = h5fi[f"{idx}/{alt}/ca_dist"][:] < DIST_THRESH_NM
        masked_gcc = h5fi[f"{idx}/{alt}/gcc_lmi"][:] * mask
        prot_graph = nx.from_numpy_array(masked_gcc)
        # logger.info(f"Processing {idx} {alt} with graph size: {prot_graph.number_of_nodes()} nodes, {prot_graph.number_of_edges()} edges")
        centrality = nx.betweenness_centrality(prot_graph, normalized=True)
        pos_rmsf = h5fi[f"{idx}/{alt}/rmsf"][:].flatten()[i]
        pos_centrality = centrality[idx]

        dms_data.append((idx, alt, pos_rmsf, pos_centrality))

# %% Create DataFrame
dms_data = pd.DataFrame(dms_data, columns=["pos_i", "aa_alt", "rmsf", "centrality"])
variance_df = dms_data.groupby("pos_i")[["rmsf", "centrality"]].var().reset_index()
variance_df.rename(
    columns={"rmsf": "variance_rmsf", "centrality": "variance_centrality"}, inplace=True
)
dms_data = pd.merge(dms_data, variance_df, on="pos_i", how="left")
logger.info(f"Saving DMS data to {config.REPORTS_DIR / 'na_channel/dms_data.csv'}...")
dms_data.to_csv(config.REPORTS_DIR / "na_channel/dms_data.csv", index=False)

# %% Plot DMS

def add_cbar(ax, label, cmap, vmin, vmax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(label)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)  # Force scientific notation
    formatter.set_powerlimits((-4, 4))  # Adjust these values based on your needs
    cbar.formatter = formatter
    cbar.update_ticks()
    return cax


LINEWIDTH = 0.5
ALPHA = 1.0
POINT_SIZE = 80

logger.info("Plotting DMS data...")

# fig, ax = plt.subplots(4, 1, figsize=(30, 24), sharex=True)
fig, ax = plt.subplots(1, 1, figsize=(24, 12), sharex=True)

sns.scatterplot(
    data=dms_data,
    x="pos_i",
    y="centrality",
    hue="variance_centrality",
    palette="magma",
    hue_norm=(0, 1e-3),
    edgecolor="black",
    linewidth=LINEWIDTH,
    alpha=ALPHA,
    ax=ax,
    legend=False,
    s=POINT_SIZE,
)
add_cbar(ax, "Variance(Centrality)", "magma", 0, 1e-3)
ax.set_xlabel("")
ax.set_ylabel("Betweenness\nCentrality")

sns.lineplot(
    x=wt_centrality.keys(),
    y=wt_centrality.values(),
    ax=ax,
    label="Wild Type Centrality",
    color="green",
    linewidth=2,
    alpha=0.5,
)
ax.legend(loc="upper left", fontsize=25)
plt.savefig(config.REPORTS_DIR / "na_channel/dms_centrality.png", dpi=300)
# %%
