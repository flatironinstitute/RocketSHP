# %% Imports
import argparse
import pickle as pk
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from biotite.structure.io import pdb, xtc
from loguru import logger
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from statannotations.Annotator import Annotator
from torch.nn.functional import softmax
from tqdm import tqdm

from rocketshp import config
from rocketshp.metrics import (
    graph_diffusion_distance,
    ipsen_mikhailov_distance,
    kl_divergence_2d,
)
from rocketshp.trajectory import (
    compute_rmsf,
    compute_shp,
)

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 16,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

# %% Paths
parser = argparse.ArgumentParser(
    description="Evaluate RocketSHP across mdCATH temperatures"
)
parser.add_argument("eval_key", type=str, help="Evaluation key for the results")
parser.add_argument(
    "--split", choices=["valid", "test"], default="teest", help="Split to evaluate"
)
# args = parser.parse_args()
# EVAL_KEY = args.eval_key
# split = args.split

EVAL_KEY = "mdcath_large_ep10"
split = "test"

reference_traj_root = Path("/mnt/home/ssledzieski/Projects/rocketshp/data/raw/atlas")
rshp_results_pickle = (
    config.EVALUATION_DATA_DIR
    / "evaluations"
    / EVAL_KEY
    / f"{EVAL_KEY}_{split}_inference_results.pkl"
)
FIGURES_DIRECTORY = config.REPORTS_DIR / EVAL_KEY / "figures"
FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
log_file = config.REPORTS_DIR / EVAL_KEY / f"{EVAL_KEY}_{split}_temperature_evaluation.log"
logger.add(log_file, level="INFO", format="{message}", encoding="utf-8")

# %% Load predictions

TEMP_LIST = ["320", "348", "379", "413", "450"]

with open(rshp_results_pickle, "rb") as f:
    rshp_pickle = pk.load(f)
rshp_all = {"_".join(k.split("/")[:2]): v for k, v in rshp_pickle.items() if k.endswith("R1")}

rshp_rmsf = {k: v["rmsf"].numpy() for k, v in rshp_all.items()}
rshp_gcc = {k: v["gcc_lmi"].numpy() for k, v in rshp_all.items()}
rshp_shp = {k: v["shp"].numpy() for k, v in rshp_all.items()}

# %% Load references

MDCATH_H5 = config.PROCESSED_DATA_DIR / "mdcath" / "mdcath_processed.h5"
reference_rmsf = {}
reference_gcc = {}
reference_shp = {}
system_sizes = {}
with h5py.File(MDCATH_H5, "r") as h5fi:
    for k in tqdm(
        rshp_all.keys(), total=len(rshp_all), desc="Load Reference Results"
    ):
        
        k0, temp = k.split("_")

        if k0 not in h5fi:
            logger.error(f"Key {k} not found in reference data")
            continue

        reference_rmsf[k] = h5fi[k0][temp]["R1"]["rmsf"][:]
        reference_gcc[k] = h5fi[k0][temp]["R1"]["gcc_lmi"][:]
        reference_shp[k] = h5fi[k0][temp]["R1"]["shp"][:]

def scale_bfactors(bfactors_T1, T1, T2, k=0.0045):
    """
    Scale B-factors from temperature T1 to temperature T2.
    
    Parameters:
    -----------
    bfactors_T1 : array-like
        B-factors at temperature T1
    T1 : float
        Original temperature in Kelvin
    T2 : float
        Target temperature in Kelvin
    k : float
        Thermal constant (default=0.0045 K⁻¹)
        
    Returns:
    --------
    array-like
        Scaled B-factors at temperature T2
    """
    scaling_factor = np.exp(k * (T2 - T1))
    bfactors_T2 = bfactors_T1 * scaling_factor
    
    return bfactors_T2

# %% Compute RMSE and spearman correlation for all systems

results_df = []
scaled_results_df = []

for k in tqdm(rshp_rmsf.keys(), desc=f"Compute RMSF"):

    k0, temp = k.split("_")
    temp = int(temp.lstrip("T"))
    b_init = reference_rmsf[f"{k.split('_')[0]}_T320"]
    b_scaled = scale_bfactors(b_init, 320, temp)

    mse = np.mean((rshp_rmsf[k] - reference_rmsf[k]) ** 2)
    rmse = np.sqrt(mse)
    rmse_scaled = np.sqrt(np.mean((b_scaled - reference_rmsf[k]) ** 2))
    if temp == 320:
        assert rmse_scaled == 0

    sp_stat, sp_p = spearmanr(rshp_rmsf[k], reference_rmsf[k])
    sp_stat_scaled, sp_p_scaled = spearmanr(b_scaled, reference_rmsf[k])

    pr_stat, pr_p = pearsonr(rshp_rmsf[k], reference_rmsf[k])
    pr_stat_scaled, pr_p_scaled = pearsonr(b_scaled, reference_rmsf[k])

    k0, temp = k.split("_")
    temp = int(temp.lstrip("T"))
    results_df.append([
        k0,
        temp,
        rmse,
        sp_stat,
        sp_p,
        pr_stat,
        pr_p,
    ])

    scaled_results_df.append([
        k0,
        temp,
        rmse_scaled,
        sp_stat_scaled,
        sp_p_scaled,
        pr_stat_scaled,
        pr_p_scaled,
    ])

results_df = pd.DataFrame(
    results_df,
    columns=[
        "system",
        "temperature",
        "rmse",
        "spearman_stat",
        "spearman_p",
        "pearson_stat",
        "pearson_p",
    ]
)
results_df["method"] = "RocketSHP"
scaled_results_df = pd.DataFrame(
    scaled_results_df,
    columns=[
        "system",
        "temperature",
        "rmse",
        "spearman_stat",
        "spearman_p",
        "pearson_stat",
        "pearson_p",
    ]
)
scaled_results_df["method"] = "Exponential Thermal Dependence"

joint_results_df = pd.concat([results_df, scaled_results_df], axis=0, ignore_index=True)

#%% RocketSHP only temperature RMSE

# Create a figure with appropriate size
plt.figure(figsize=(15, 8))

# Create the box plot
sns.boxplot(
    x='temperature', 
    y='rmse', 
    data=results_df,
    fill=False,
    fliersize=0,
    linewidth=1,
    legend=False,
    color="black",
)

# Add jittered points to show individual data points
sns.stripplot(
    x='temperature', 
    y='rmse', 
    data=results_df,
    dodge=True,
    alpha=0.5,
    size=4,
)

# Enhance the plot
plt.xlabel('Temperature (K)', fontsize=24)
plt.ylabel('RMSE (Angstroms)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"rshp_rmse_temperature.svg",
    bbox_inches="tight",
    transparent=True,
)

# %% RocketSHP only temperature spearman

plt.figure(figsize=(15, 8))

# Create the box plot
sns.boxplot(
    x='temperature', 
    y='spearman_stat', 
    data=results_df,
    fill=False,
    fliersize=0,
    linewidth=1,
    legend=False,
    color="black",
)

# Add jittered points to show individual data points
sns.stripplot(
    x='temperature', 
    y='spearman_stat', 
    data=results_df,
    dodge=True,
    alpha=0.5,
    size=4,
)

# Enhance the plot
plt.xlabel('Temperature (K)', fontsize=24)
plt.ylabel('Spearman Correlation of RMSF', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"rshp_spearman_temperature.svg",
    bbox_inches="tight",
    transparent=True,
)

#%% Plot

plt.figure(figsize=(15, 8))

# Create the box plot
sns.boxplot(
    x='temperature', 
    y='rmse', 
    hue='method',
    hue_order=[ "Exponential Thermal Dependence", "RocketSHP"],
    data=joint_results_df,
    fill=False,
    fliersize=0,
    linewidth=1,
    legend=False,
    color="black",
)

# Add jittered points to show individual data points
sns.stripplot(
    x='temperature', 
    y='rmse', 
    hue='method',
    hue_order=["Exponential Thermal Dependence", "RocketSHP"],
    data=joint_results_df,
    dodge=True,
    alpha=0.5,
    size=4,
)

# Enhance the plot
plt.xlabel('Temperature (K)', fontsize=24)
plt.ylabel('RMSE (Angstroms)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title='Method', fontsize=16, title_fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"rshp_v_exponential_temperature_rmsf.svg",
    bbox_inches="tight",
    transparent=True,
)

#%% Plot

plt.figure(figsize=(15, 8))

# Create the box plot
sns.boxplot(
    x='temperature', 
    y='spearman_stat', 
    hue='method',
    hue_order=[ "Exponential Thermal Dependence", "RocketSHP"],
    data=joint_results_df,
    fill=False,
    fliersize=0,
    linewidth=1,
    legend=False,
    color="black",
)

# Add jittered points to show individual data points
sns.stripplot(
    x='temperature', 
    y='spearman_stat', 
    hue='method',
    hue_order=["Exponential Thermal Dependence", "RocketSHP"],
    data=joint_results_df,
    dodge=True,
    alpha=0.5,
    size=4,
)

# Enhance the plot
plt.xlabel('Temperature (K)', fontsize=24)
plt.ylabel('Spearman Correlation of RMSF', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title='Method', fontsize=16, title_fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"rshp_v_exponential_temperature_spearman.svg",
    bbox_inches="tight",
    transparent=True,
)

# %% Case study
system_list = list(set(i[:7] for i in rshp_rmsf.keys()))
# system = list(rshp_rmsf.keys())[30].split("_")[0]
system = '1u60A00'

fig, ax = plt.subplots(figsize=(24, 10))

b_init = rshp_rmsf[f"{system}_T320"]

for temp in TEMP_LIST:
    k = f"{system}_T{temp}"

    ax.plot(
        rshp_rmsf[k],
        alpha=0.8,
        linewidth=3,
        label = f"{temp}K Predicted",
    )
    # get color of previous line
    color = ax.get_lines()[-1].get_color()
    ax.plot(
        reference_rmsf[k],
        c = color,
        linestyle="--",
        alpha=0.4,
        linewidth=3,
        label = f"{temp}K True",
    )
    # ax.plot(
    #     scale_bfactors(b_init, 320, int(temp)),
    #     alpha=0.8,
    #     linewidth=3,
    #     c = color,
    #     linestyle="-.",
    #     label = f"{temp}K Exponential",
    # )
ax.set_xlabel("Residue index", fontsize=26)
ax.set_ylabel("RMSF (Angstoms)", fontsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# set legend outside of plot and turn off box
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=False,
    fontsize=22,
)
ax.set_title(f"RMSF for {system}", fontsize=26)
plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"{system}_temperature_RMSF_single.svg",
    bbox_inches="tight",
    transparent=True,
)

# %% Case study

# possible good indices: 700, 30, 500

# system = list(rshp_rmsf.keys())[700].split("_")[0]
logger.info(f"System: {system}")

fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True, sharey=True)
b_init = reference_rmsf[f"{system}_T320"]
color_list = ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954']


for i, temp in enumerate(TEMP_LIST):
    k = f"{system}_T{temp}"

    ax[i].plot(
        rshp_rmsf[k],
        alpha=0.8,
        linewidth=1,
        label = f"{temp}K Predicted",
        c = color_list[i],
    )
    # get color of previous line
    ax[i].plot(
        reference_rmsf[k],
        c = color_list[i],
        linestyle="--",
        alpha=0.4,
        linewidth=1,
        label = f"{temp}K True",
    )
    ax[i].plot(
        scale_bfactors(b_init, 320, int(temp)),
        alpha=0.8,
        linewidth=1,
        c = color_list[i],
        linestyle="-.",
        label = f"{temp}K Exponential",
    )
    ax[i].set_ylabel("RMSF\n(Angstoms)", fontsize=16)


    ax[i].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=16,
    )

ax[-1].set_xlabel("Residue index", fontsize=16)
ax[0].set_title(f"RMSF for {system}", fontsize=26)
plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"{system}_temperature_RMSF_rows.svg",
    bbox_inches="tight",
    transparent=True,
)

# %%
