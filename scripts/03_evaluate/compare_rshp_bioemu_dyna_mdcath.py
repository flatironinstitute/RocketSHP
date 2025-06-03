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
    description="Compare RocketSHP, Dyna-1, and BioEMU results"
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
dyna_results_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_mdcath_results/")
bioemu_results_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_mdcath_results_100/")
rshp_results_pickle = (
    config.EVALUATION_DATA_DIR
    / "evaluations"
    / EVAL_KEY
    / f"{EVAL_KEY}_{split}_inference_results.pkl"
)
FIGURES_DIRECTORY = config.REPORTS_DIR / EVAL_KEY / "figures"
FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
log_file = config.REPORTS_DIR / EVAL_KEY / f"{EVAL_KEY}_{split}_evaluation.log"

# configure logger to write to log file
logger.add(log_file, level="INFO", format="{message}", encoding="utf-8")

assert reference_traj_root.exists(), (
    f"Reference trajectory root not found: {reference_traj_root}"
)
assert rshp_results_pickle.exists(), (
    f"RocketSHP results pickle not found: {rshp_results_pickle}"
)
assert dyna_results_root.exists(), f"Dyna results root not found: {dyna_results_root}"
assert bioemu_results_root.exists(), (
    f"BioEMU results root not found: {bioemu_results_root}"
)

# %% Load RocketSHP results

TEMP_0 = "320"

with open(rshp_results_pickle, "rb") as f:
    rshp_pickle = pk.load(f)
rshp_all = {"_".join(k.split("/")[:2]): v for k, v in rshp_pickle.items() if k.endswith("R1")}
rshp_results = {k.split("_")[0]: v for k, v in rshp_all.items() if k.endswith(TEMP_0)}
rshp_rmsf = {k: v["rmsf"].numpy() for k, v in rshp_results.items()}
rshp_gcc = {k: v["gcc_lmi"].numpy() for k, v in rshp_results.items()}
rshp_shp = {k: v["shp"].numpy() for k, v in rshp_results.items()}

# %% Load Dyna results
dyna_results = {}
for k in tqdm(rshp_results.keys(), desc="Load Dyna Results"):
    dyna_path = dyna_results_root / k / f"{k}-Dyna1.csv"
    assert dyna_path.exists(), f"Dyna results not found: {dyna_path}"
    dyna_df = pd.read_csv(dyna_path)
    dyna_probability = dyna_df.iloc[:, 2].values
    dyna_results[k] = dyna_probability

# %% Load BioEMU results
# bioemu_results = {}
# bioemu_gcc = {}
# bioemu_shp = {}
bioemu_100_results = {}
bioemu_100_gcc = {}
bioemu_100_shp = {}

for k in tqdm(rshp_results.keys(), desc="Load BioEMU Results"):

    bioemu_100_path = bioemu_results_root/ f"{k}_100"
    try:
        assert (bioemu_100_path / "topology.pdb").exists(), f"BioEMU 100 results not found: {bioemu_100_path}"
    except AssertionError:
        logger.warning(f"Chain {k} not yet complete, skipping")
        continue
    bioemu_100_traj = md.load(
        bioemu_100_path / "samples.xtc", top=bioemu_100_path / "topology.pdb"
    )
    bioemu_100_rmsf = compute_rmsf(bioemu_100_traj)
    bioemu_100_gcc_lmi = np.load(bioemu_100_path / "gcc_lmi.npy")
    bioemu_100_results[k] = bioemu_100_rmsf
    bioemu_100_gcc[k] = bioemu_100_gcc_lmi
    bs_100_top = pdb.PDBFile.read(str(bioemu_100_path / "topology.pdb")).get_structure()
    bs_100_xtc = xtc.XTCFile.read(str(bioemu_100_path / "samples.xtc")).get_structure(
        bs_100_top
    )
    bioemu_100_shp[k] = compute_shp(bs_100_xtc)

# %% Load GNM results
# GNM_ROOT = Path(
#     "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/gaussian_net_models"
# )
# gnm_gcc = {}
# for k in tqdm(rshp_results.keys(), total=len(rshp_results), desc="Load GNM Results"):
#     gnm_path = GNM_ROOT / f"{k[:2]}/{k}_gnm.npz"
#     gnm_data = np.load(gnm_path)
#     gnm_covar = gnm_data["covar"]
#     # min-max scale between 0 and 1
#     gnm_covar = (gnm_covar - np.min(gnm_covar)) / (
#         np.max(gnm_covar) - np.min(gnm_covar)
#     )
#     gnm_gcc[k] = gnm_covar

# %% Load reference results

MDCATH_H5 = config.PROCESSED_DATA_DIR / "mdcath" / "mdcath_processed.h5"
reference_rmsf = {}
reference_gcc = {}
reference_shp = {}
system_sizes = {}
with h5py.File(MDCATH_H5, "r") as h5fi:
    for k in tqdm(
        rshp_results.keys(), total=len(rshp_results), desc="Load Reference Results"
    ):
        if k not in h5fi:
            logger.error(f"Key {k} not found in reference data")
            continue

        # Get RMSF
        reference_rmsf[k] = h5fi[k][f"T{TEMP_0}"]["R1"]["rmsf"][:]
        # Get GCC
        reference_gcc[k] = h5fi[k][f"T{TEMP_0}"]["R1"]["gcc_lmi"][:]
        reference_shp[k] = h5fi[k][f"T{TEMP_0}"]["R1"]["shp"][:]
        system_sizes[k] = len(reference_rmsf[k])

# reference_rmsf = {}
# system_sizes = {}

# for k in tqdm(rshp_results.keys(), desc="Load Reference Trajectories"):
#     reference_xtc = reference_traj_root / k[:2] / f"{k}_prod_R1_fit.xtc"
#     reference_top = reference_traj_root / k[:2] / f"{k}.pdb"
#     reference_traj = md.load(reference_xtc, top=reference_top)
#     system_sizes[k] = reference_traj.n_residues
#     reference_rmsf[k] = compute_rmsf(reference_traj)

# %% Rescale dyna from probability to RMSF (not really how it works, but for comparison)
all_predicted = []
all_calibration = []
for system in ["11080", "BLOT5", "CHEA", "RNASE", "CRABP2", "SPOOF"]:
    dyna_calibration_sim = (
        f"/mnt/home/ssledzieski/Projects/rocketshp/data/processed/relaxdb_sims/{system}"
    )
    dyna_calibration_pred = pd.read_csv(
        f"/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/{system}/{system}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000-Dyna1.csv"
    ).p_exchange
    dyna_calibration_traj = md.load(
        f"{dyna_calibration_sim}/{system}_traj.xtc",
        top=f"{dyna_calibration_sim}/{system}_top.pdb",
    )

    dyna_calibration_rmsf = compute_rmsf(dyna_calibration_traj)

    all_predicted.append(dyna_calibration_pred)
    all_calibration.append(dyna_calibration_rmsf)
all_predicted = np.concatenate(all_predicted)
all_calibration = np.concatenate(all_calibration)


# plt.plot(dyna_calibration_rmsf, label="Dyna RMSF", c="grey")
# Reshape for sklearn (needs 2D array)
X = all_predicted.reshape(-1, 1)
y = all_calibration

# Fit linear model
model = LinearRegression()
model.fit(X, y)

# Get parameters
calibration_scale = model.coef_[0]  # Scaling factor
calibration_offset = model.intercept_  # Offset
logger.info(f"Scaling factor: {calibration_scale}, Offset: {calibration_offset}")

# %% Plot calibrated data
# system = "CRABP2"
# dyna_calibration_sim = f"/mnt/home/ssledzieski/Projects/rocketshp/data/processed/relaxdb_sims/{system}"
# dyna_calibration_pred = pd.read_csv(f"/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets/output_structures/{system}/{system}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000-Dyna1.csv").p_exchange
# calibration_scaled = calibration_scale * dyna_calibration_pred + calibration_offset

# dyna_calibration_traj = md.load(f"{dyna_calibration_sim}/{system}_traj.xtc", top=f"{dyna_calibration_sim}/{system}_top.pdb")
# dyna_calibration_rmsf = compute_rmsf(dyna_calibration_traj)

# rxdb = pd.read_json("~/GitHub/Dyna-1/data/RelaxDB_datasets/RelaxDB_with_other_metrics_22jan2025.json")
# rxdb_label = rxdb.loc[f"{system}"]["label"]

# rxdb_map = {
#     "p": ".",
#     ".": ".",
#     "x": "-",
#     "v": "v",
#     "^": "^",
#     "b": "^",
#     "A": "-"
# }

# plt.figure(figsize=(10, 6))
# plt.plot(dyna_calibration_pred, label="Dyna Pred")
# plt.plot(calibration_scaled, label="Dyna Pred (scaled)")
# for i, c in enumerate(rxdb_label):
#     c = rxdb_map.get(c, c)
#     plt.annotate(c, xy=(i, dyna_calibration_rmsf[i]), xytext=(i, dyna_calibration_rmsf[i]), ha="center", fontsize=10)
# plt.ylim(0, 1)
# plt.legend()
# plt.show()

# %% Calibrate RMSF
dyna_rmsf = {}
for k in tqdm(rshp_results.keys(), desc="Calibrate Dyna Probabilities"):
    dyna_probability = dyna_results[k]
    # min max scale probability based on observed calibration rmsf
    dyna_scaled = (calibration_scale * dyna_probability) + calibration_offset
    dyna_rmsf[k] = dyna_scaled

# %% Compare RMSF for all methods
methods = {
    "RocketSHP": rshp_rmsf,
    "Dyna-1": dyna_results,
    "Dyna-1 (Calibrated)": dyna_rmsf,
    # "BioEmu (10)": bioemu_results,
    "BioEmu (100)": bioemu_100_results,
    "Reference": reference_rmsf,
}
order = ["RocketSHP", "Dyna-1 (Calibrated)", "Dyna-1", "BioEmu (100)", "BioEmu (10)"]

rmsf_results = {}
for k in tqdm(bioemu_100_results.keys(), desc="Collect RMSF Results"):
    rmsf_results[k] = {m: methods[m][k] for m in methods.keys()}
rmsf_df = pd.DataFrame(rmsf_results).T
rmsf_df = rmsf_df.rename_axis("System").reset_index()
rmsf_df = rmsf_df.melt(id_vars=["System"], var_name="Method", value_name="RMSF")

# %% Plot a particular system
# system = "1a75A00"
system = "1u60A00"

if system not in rmsf_df["System"].values:
    logger.warning(f"System {system} not found in RMSF results")
else:
    fig, ax = plt.subplots(figsize=(12, 8))

    for method in methods:
        ax.plot(
            rmsf_df[rmsf_df["System"] == system]
            .loc[rmsf_df["Method"] == method, "RMSF"]
            .values[0],
            label=method,
        )

    ax.set_title(f"{system} ({system_sizes[system]} residues)")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("RMSF (Angstroms)")
    ax.legend(title="Method", loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIRECTORY / f"{split}_rmsf_comparison_{system}.svg")
# plt.show()

# %% Compute RMSE and spearman correlation for all systems
mean_sq_error = {}
spearman = {}
pearson = {}
for method in methods:
    if method == "Reference":
        continue
    mean_sq_error[method] = []
    spearman[method] = []
    pearson[method] = []
    for k in tqdm(rmsf_results.keys(), desc=f"Compute RMSF metrics for {method}"):
        mse = np.mean((rmsf_results[k][method] - rmsf_results[k]["Reference"]) ** 2)
        rmse = np.sqrt(mse)
        sp_stat, sp_p = spearmanr(rmsf_results[k][method], rmsf_results[k]["Reference"])
        pr_stat, pr_p = pearsonr(rmsf_results[k][method], rmsf_results[k]["Reference"])
        mean_sq_error[method].append(rmse)
        spearman[method].append(sp_stat)
        pearson[method].append(pr_stat)

mean_sq_error_df = pd.DataFrame(mean_sq_error)
mean_sq_error_df = mean_sq_error_df.rename_axis("System").reset_index()
mean_sq_error_df = mean_sq_error_df.melt(
    id_vars=["System"], var_name="Method", value_name="RMSE"
)

spearman_df = pd.DataFrame(spearman)
spearman_df = spearman_df.rename_axis("System").reset_index()
spearman_df = spearman_df.melt(id_vars=["System"], var_name="Method", value_name="RMSE")

# mean_sq_error_df = mean_sq_error_df.groupby("Method").mean().reset_index()
# mean_sq_error_df = mean_sq_error_df.sort_values("MSE", ascending=False)

# %% Plot RMSE
fig, ax = plt.subplots(figsize=(12, 8))

order = ["RocketSHP", "Dyna-1 (Calibrated)", "Dyna-1", "BioEmu (100)"]
for i in order:
    logger.info(
        f"Mean {i} RMSE: {mean_sq_error_df[mean_sq_error_df['Method'] == i]['RMSE'].mean()}"
    )

sns.barplot(
    data=mean_sq_error_df, x="Method", y="RMSE", ax=ax, errorbar="se", order=order
)
ax.set_title("Root Mean Squared Error (RMSE) of RMSF")
ax.set_xlabel("Method")
ax.set_ylabel("RMSE (Angstroms)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_rmsf_rmse_comparison.svg")
# plt.show()

# %% Plot Spearman correlation
fig, ax = plt.subplots(figsize=(12, 8))

order = ["RocketSHP", "Dyna-1 (Calibrated)", "Dyna-1", "BioEmu (100)"]
for i in order:
    logger.info(
        f"Mean {i} Spearman: {spearman_df[spearman_df['Method'] == i]['RMSE'].mean()}"
    )

sns.barplot(data=spearman_df, x="Method", y="RMSE", ax=ax, errorbar="se", order=order)
ax.set_title("Spearman Correlation of RMSF")
ax.set_xlabel("Method")
ax.set_ylabel("Spearman Correlation")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_rmsf_spearman_comparison.svg")

# %% Plot MSE in several bins by system size
mean_sq_error_by_size = []
spearman_by_size = []
for method in methods:
    if method == "Reference":
        continue
    for k in tqdm(
        rmsf_results.keys(), desc=f"Compute RMSF metrics for {method} by size"
    ):
        mse = np.mean((rmsf_results[k][method] - rmsf_results[k]["Reference"]) ** 2)
        rmse = np.sqrt(mse)
        sp_stat, sp_p = spearmanr(rmsf_results[k][method], rmsf_results[k]["Reference"])
        mean_sq_error_by_size.append([k, system_sizes[k], method, rmse])
        spearman_by_size.append([k, system_sizes[k], method, sp_stat])

mean_sq_error_by_size_df = pd.DataFrame(mean_sq_error_by_size)
mean_sq_error_by_size_df.columns = ["System", "Size", "Method", "RMSE"]

spearman_by_size_df = pd.DataFrame(spearman_by_size)
spearman_by_size_df.columns = ["System", "Size", "Method", "Spearman"]

# assign size group by quartiles
# mean_sq_error_by_size_df["Size Group"] = pd.qcut(mean_sq_error_by_size_df["Size"], q=4, labels=["Small", "Medium", "Large", "Extra Large"])
# spearman_by_size_df["Size Group"] = pd.qcut(spearman_by_size_df["Size"], q=4, labels=["Small", "Medium", "Large", "Extra Large"])
# quartile_boundaries = mean_sq_error_by_size_df["Size"].quantile([0, 0.25, 0.5, 0.75, 1]).values

# assign size groups by manual binning
quartile_boundaries = np.array([0, 100, 150, 250, 350])
mean_sq_error_by_size_df["Size Group"] = pd.cut(
    mean_sq_error_by_size_df["Size"],
    bins=quartile_boundaries,
    labels=["Small", "Small-Medium", "Medium", "Large"],
    include_lowest=True,
)
spearman_by_size_df["Size Group"] = pd.cut(
    spearman_by_size_df["Size"],
    bins=quartile_boundaries,
    labels=["Small", "Small-Medium", "Medium", "Large"],
    include_lowest=True,
)

# # size group labels should be the min and max of each bin
size_group_labels = []
for i in range(len(quartile_boundaries) - 1):
    size_group_labels.append(
        f"{int(quartile_boundaries[i])} - {int(quartile_boundaries[i + 1] - 1)}"
    )
mean_sq_error_by_size_df["Size Group"] = mean_sq_error_by_size_df[
    "Size Group"
].cat.rename_categories(size_group_labels)
spearman_by_size_df["Size Group"] = spearman_by_size_df[
    "Size Group"
].cat.rename_categories(size_group_labels)
size_group_df = mean_sq_error_by_size_df[
    ["System", "Size", "Size Group"]
].drop_duplicates()

# %% Check overall performance difference and in largest size group

mean_rshp_rmse = mean_sq_error_by_size_df[
    mean_sq_error_by_size_df["Method"] == "RocketSHP"
]["RMSE"].mean()
mean_bioemu_100_rmse = mean_sq_error_by_size_df[
    mean_sq_error_by_size_df["Method"] == "BioEmu (100)"
]["RMSE"].mean()
mean_dyna_1_rmse = mean_sq_error_by_size_df[
    mean_sq_error_by_size_df["Method"] == "Dyna-1 (Calibrated)"
]["RMSE"].mean()
logger.info(f"Mean RocketSHP RMSE: {mean_rshp_rmse}")
logger.info(f"Mean BioEmu (100) RMSE: {mean_bioemu_100_rmse}")
logger.info(f"Mean Dyna-1 (Calibrated) RMSE: {mean_dyna_1_rmse}")
# log % difference over other methods
logger.info(
    f"RocketSHP RMSE % difference to BioEmu (100): {(mean_rshp_rmse - mean_bioemu_100_rmse) / mean_bioemu_100_rmse * 100:.2f}%"
)
logger.info(
    f"RocketSHP RMSE % difference to Dyna-1 (Calibrated): {(mean_rshp_rmse - mean_dyna_1_rmse) / mean_dyna_1_rmse * 100:.2f}%"
)

# only in largest size group
mean_rshp_rmse = mean_sq_error_by_size_df[
    (mean_sq_error_by_size_df["Method"] == "RocketSHP")
    & (mean_sq_error_by_size_df["Size"] >= 250)
]["RMSE"].mean()
mean_bioemu_100_rmse = mean_sq_error_by_size_df[
    (mean_sq_error_by_size_df["Method"] == "BioEmu (10)")
    & (mean_sq_error_by_size_df["Size"] >= 250)
]["RMSE"].mean()
mean_dyna_1_rmse = mean_sq_error_by_size_df[
    (mean_sq_error_by_size_df["Method"] == "Dyna-1 (Calibrated)")
    & (mean_sq_error_by_size_df["Size"] >= 250)
]["RMSE"].mean()
logger.info(f"Mean RocketSHP RMSE (Large): {mean_rshp_rmse}")
logger.info(f"Mean BioEmu (100) RMSE (Large): {mean_bioemu_100_rmse}")
logger.info(f"Mean Dyna-1 (Calibrated) RMSE (Large): {mean_dyna_1_rmse}")
# log % difference over other methods
logger.info(
    f"RocketSHP RMSE % difference to BioEmu (100) (Large): {(mean_rshp_rmse - mean_bioemu_100_rmse) / mean_bioemu_100_rmse * 100:.2f}%"
)
logger.info(
    f"RocketSHP RMSE % difference to Dyna-1 (Calibrated) (Large): {(mean_rshp_rmse - mean_dyna_1_rmse) / mean_dyna_1_rmse * 100:.2f}%"
)

# %% Boxplot

fig, ax = plt.subplots(figsize=(12, 8))
pairs = [("RocketSHP", "Dyna-1 (Calibrated)"), ("RocketSHP", "BioEmu (100)")]

ax = sns.boxplot(
    x="Method",
    y="RMSE",
    hue="Size Group",
    data=mean_sq_error_by_size_df,
    linewidth=1.5,
    fliersize=0,
    order=order,
    fill=False,
    legend=False,
    color="black",
)  # Set fill=False for outline-only boxes

sns.stripplot(
    x="Method",
    y="RMSE",
    hue="Size Group",
    data=mean_sq_error_by_size_df,
    size=6,
    dodge=True,
    order=order,
    jitter=True,
    alpha=0.7,
    ax=ax,
)

ax.set_yscale("log")
ax.set_ylabel("RMSE (Angstroms)")

# add annotations
annotator = Annotator(
    ax, pairs=pairs, data=mean_sq_error_by_size_df, x="Method", y="RMSE", order=order
)
annotator.configure(test="t-test_paired", loc="inside", verbose=2)
_, test_results = annotator.apply_and_annotate()

# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.savefig(FIGURES_DIRECTORY / f"{split}_rmsf_rmse_method_box_by_method.svg")

for pair, stats in zip(pairs, test_results):
    stat_val = stats.data.stat_value
    p_val = stats.data.pvalue

    logger.info(
        f"RMSE Comparison: {pair[0]} vs {pair[1]}, t-statistic: {stat_val:.4f}, p-value: {p_val:.4e}"
    )

# %% Plot Spearman
fig, ax = plt.subplots(figsize=(12, 8))
pairs = [("RocketSHP", "Dyna-1 (Calibrated)"), ("RocketSHP", "BioEmu (100)")]

ax = sns.boxplot(
    x="Method",
    y="Spearman",
    hue="Size Group",
    data=spearman_by_size_df,
    linewidth=1.5,
    fliersize=0,
    order=order,
    fill=False,
    legend=False,
    color="black",
)  # Set fill=False for outline-only boxes

sns.stripplot(
    x="Method",
    y="Spearman",
    hue="Size Group",
    data=spearman_by_size_df,
    size=6,
    dodge=True,
    order=order,
    jitter=True,
    alpha=0.7,
    ax=ax,
)

# ax.set_yscale('log')
ax.set_ylabel("Spearman (Angstroms)")
ax.set_ylim(-0.5, 1)

# add annotations
annotator = Annotator(
    ax, pairs=pairs, data=spearman_by_size_df, x="Method", y="Spearman", order=order
)
annotator.configure(test="t-test_paired", loc="inside", verbose=2)
_, test_results = annotator.apply_and_annotate()

# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.savefig(FIGURES_DIRECTORY / f"{split}_rmsf_spearman_method_box_by_method.svg")

for pair, stats in zip(pairs, test_results):
    stat_val = stats.data.stat_value
    p_val = stats.data.pvalue

    logger.info(
        f"Spearman Comparison: {pair[0]} vs {pair[1]}, t-statistic: {stat_val:.4f}, p-value: {p_val:.4e}"
    )

# %% Compute GCC metrics

gcc_results = []
for k, v in tqdm(bioemu_100_gcc.items()):
    network_size = v.shape[0]
    rshp_gdd = graph_diffusion_distance(rshp_gcc[k], reference_gcc[k], beta=1 / network_size)
    rshp_imsd = ipsen_mikhailov_distance(rshp_gcc[k], reference_gcc[k])
    bioemu_100_gdd = graph_diffusion_distance(
        bioemu_100_gcc[k], reference_gcc[k], beta=1 / network_size
    )
    bioemu_100_imsd = ipsen_mikhailov_distance(bioemu_100_gcc[k], reference_gcc[k])
    gcc_results.append(
        [
            k,
            rshp_gdd,
            rshp_imsd,
            bioemu_100_gdd,
            bioemu_100_imsd,
        ]
    )
gcc_results_df = pd.DataFrame(gcc_results)
gcc_results_df.columns = [
    "System",
    "RocketSHP GDD",
    "RocketSHP IMSD",
    "BioEmu 100 GDD",
    "BioEmu 100 IMSD",
]
gcc_results_df = pd.merge(
    size_group_df, gcc_results_df, left_on="System", right_on="System", how="inner"
)
# %% Separate by metric

gdd_results_df = gcc_results_df[
    ["System", "Size Group", "RocketSHP GDD", "BioEmu 100 GDD"]
].melt(id_vars=["System", "Size Group"], var_name="Method", value_name="GDD")
gdd_results_df["Method"] = gdd_results_df["Method"].replace(
    {
        "RocketSHP GDD": "RocketSHP",
        "BioEmu 100 GDD": "BioEmu (100)",
    }
)

imsd_results_df = gcc_results_df[
    ["System", "Size Group", "RocketSHP IMSD", "BioEmu 100 IMSD"]
].melt(id_vars=["System", "Size Group"], var_name="Method", value_name="IMSD")
imsd_results_df["Method"] = imsd_results_df["Method"].replace(
    {
        "RocketSHP IMSD": "RocketSHP",
        "BioEmu 100 IMSD": "BioEmu (100)",
    }
)

# %% Boxplot GDD

fig, ax = plt.subplots(figsize=(12, 8))
order = ["RocketSHP", "BioEmu (100)"]
pairs = [("RocketSHP", "BioEmu (100)")]

for i in order:
    logger.info(
        f"Mean {i} GDD: {gdd_results_df[gdd_results_df['Method'] == i]['GDD'].mean()}"
    )

sns.boxplot(
    data=gdd_results_df,
    x="Method",
    y="GDD",
    ax=ax,
    order=order,
    hue="Size Group",
    color="black",
    fill=False,
    showfliers=False,
    linewidth=1.5,
    legend=False,
)
sns.stripplot(
    data=gdd_results_df,
    x="Method",
    y="GDD",
    ax=ax,
    order=order,
    hue="Size Group",
    size=6,
    dodge=True,
    jitter=True,
    alpha=0.7,
)
# ax.set_yscale("log")

# Add statistical annotations
annotator = Annotator(
    ax, pairs=pairs, data=gdd_results_df, x="Method", y="GDD", order=order
)
annotator.configure(test="t-test_paired", loc="inside", verbose=2)
_, test_results = annotator.apply_and_annotate()

ax.set_title("Graph Diffusion Distance (GDD) of GCC-LMI")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_gdd_boxplot.svg")

for pair, stats in zip(pairs, test_results):
    stat_val = stats.data.stat_value
    p_val = stats.data.pvalue

    logger.info(
        f"GDD Comparison: {pair[0]} vs {pair[1]}, t-statistic: {stat_val:.4f}, p-value: {p_val:.4e}"
    )

# %% Boxplot IMSD
fig, ax = plt.subplots(figsize=(12, 8))
order = ["RocketSHP", "BioEmu (100)"]
pairs = [("RocketSHP", "BioEmu (100)")]

for i in order:
    logger.info(
        f"Mean {i} IMSD: {imsd_results_df[imsd_results_df['Method'] == i]['IMSD'].mean()}"
    )

sns.boxplot(
    data=imsd_results_df,
    x="Method",
    y="IMSD",
    ax=ax,
    order=order,
    hue="Size Group",
    color="black",
    fill=False,
    showfliers=False,
    linewidth=1.5,
    legend=False,
)
sns.stripplot(
    data=imsd_results_df,
    x="Method",
    y="IMSD",
    ax=ax,
    order=order,
    hue="Size Group",
    size=6,
    dodge=True,
    jitter=True,
    alpha=0.7,
)
# ax.set_yscale("log")

# Add statistical annotations
annotator = Annotator(
    ax, pairs=pairs, data=imsd_results_df, x="Method", y="IMSD", order=order
)
annotator.configure(test="t-test_paired", loc="inside", verbose=2)
_, test_results = annotator.apply_and_annotate()

ax.set_title("Ipsen-Mikhailov Distance (IMSD) of GCC-LMI")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_imsd_boxplot.svg")

for pair, stats in zip(pairs, test_results):
    stat_val = stats.data.stat_value
    p_val = stats.data.pvalue

    logger.info(
        f"IMSD Comparison: {pair[0]} vs {pair[1]}, t-statistic: {stat_val:.4f}, p-value: {p_val:.4e}"
    )

# %% GDD Scatter Plot
gdd_scatter_data = gcc_results_df[
    ["System", "Size Group", "RocketSHP GDD", "BioEmu 100 GDD"]
]
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=gdd_scatter_data,
    x="BioEmu 100 GDD",
    y="RocketSHP GDD",
    ax=ax,
    hue="Size Group",
    alpha=0.95,
)
mmax = gdd_scatter_data.drop(["System", "Size Group"], axis=1).max().max()
ax.plot([0, mmax], [0, mmax], color="gray", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_gdd_scatter.svg")

# %% IMSD Scatter Plot
imsd_scatter_data = gcc_results_df[
    ["System", "Size Group", "RocketSHP IMSD", "BioEmu 100 IMSD"]
]
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=imsd_scatter_data,
    x="BioEmu 100 IMSD",
    y="RocketSHP IMSD",
    ax=ax,
    hue="Size Group",
    alpha=0.95,
)
# add x=y line
ax.plot([0, 5], [0, 5], color="gray", linestyle="--")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_imsd_scatter.svg")

# %% Compute SHP metrics
shp_results = []
for k, v in tqdm(bioemu_100_shp.items()):
    rshp_kl = kl_divergence_2d(
        softmax(torch.from_numpy(rshp_shp[k])), torch.from_numpy(reference_shp[k])
    )
    bioemu_100_kl = kl_divergence_2d(
        bioemu_100_shp[k], torch.from_numpy(reference_shp[k])
    )
    shp_results.append([k, rshp_kl, bioemu_100_kl])
shp_results_df = pd.DataFrame(shp_results)
shp_results_df.columns = ["System", "RocketSHP", "BioEmu (100)"]
shp_results_df = pd.merge(
    size_group_df, shp_results_df, left_on="System", right_on="System", how="inner"
)

# %% Boxplots
kldiv_results_df = shp_results_df.melt(
    id_vars=["System", "Size Group"], var_name="Method", value_name="KL-Div"
)

fig, ax = plt.subplots(figsize=(12, 8))
order = ["RocketSHP", "BioEmu (100)"]
pairs = [("RocketSHP", "BioEmu (100)")]

for i in order:
    logger.info(f"Mean {i} KL-Div: {shp_results_df[i].mean()}")

sns.boxplot(
    data=kldiv_results_df,
    x="Method",
    y="KL-Div",
    ax=ax,
    order=order,
    hue="Size Group",
    color="black",
    fill=False,
    showfliers=False,
    linewidth=1.5,
    legend=False,
)
sns.stripplot(
    data=kldiv_results_df,
    x="Method",
    y="KL-Div",
    ax=ax,
    order=order,
    hue="Size Group",
    size=6,
    dodge=True,
    jitter=True,
    alpha=0.7,
)

annotator = Annotator(
    ax, pairs=pairs, data=kldiv_results_df, x="Method", y="KL-Div", order=order
)
annotator.configure(test="t-test_paired", loc="inside", verbose=2)
_, test_results = annotator.apply_and_annotate()

ax.set_title("KL-Divergence of SHP")
plt.tight_layout()
plt.savefig(FIGURES_DIRECTORY / f"{split}_shp_kl_boxplot.svg")

for pair, stats in zip(pairs, test_results):
    stat_val = stats.data.stat_value
    p_val = stats.data.pvalue

    logger.info(
        f"KL-Div Comparison: {pair[0]} vs {pair[1]}, t-statistic: {stat_val:.4f}, p-value: {p_val:.4e}"
    )
# %%
