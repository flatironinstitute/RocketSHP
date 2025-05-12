#!/usr/bin/env python

# %%

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import entropy, pearsonr, spearmanr
from torch.nn.functional import softmax

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.trajectory import compute_rmsf

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


# %%

### UNCOMMENT IF RUNNING AS SCRIPT

# parser = ArgumentParser()
# parser.add_argument("--key", type=str, help="Protein key (e.g. 1tzw_A/R1)", required=True)
# parser.add_argument("--savefig", action="store_true", help="Save figures")
# args = parser.parse_args()
# key = args.key
# savefig = args.savefig

### UNCOMMENT IF RUNNING AS NOTEBOOK
key = "1tzw_A/R1"
savefig = False

# %%
config_file = "/mnt/home/ssledzieski/Projects/rocketshp/configs/20250427_large.yml"

PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(config_file))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%

adl = ATLASDataModule(
    config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
    seq_features=True,
    struct_features=True,
    batch_size=8,
    crop_size=2048,
    num_workers=PARAMS.num_data_workers,
    train_pct=PARAMS.train_pct,
    val_pct=PARAMS.val_pct,
    random_seed=PARAMS.random_seed,
    struct_stage=PARAMS.struct_stage,
)
adl.setup("train")
ads = adl.dataset
logger.info(
    f"{len(adl.dataset)}, {len(adl.train_data)}, {len(adl.val_data)}, {len(adl.test_data)}"
)


# %%

model = RocketSHPModel.load_from_checkpoint("latest")
model = model.to(device)
model = model.eval()


# %%
def get_idx_name(name):
    return adl.dataset.samples.index(name)


feats, labels = adl.dataset[get_idx_name(key)]

pdb_code, rep = key.split("/")
pdb_code_save = pdb_code
logger.info(f"Key: {key}")
logger.info(f"Sequence length: {len(labels['rmsf'])}")


def normalize(x):
    return (x - x.mean()) / x.std()


with torch.no_grad():
    both_result = model({k: v.to(device).unsqueeze(0) for k, v in feats.items()})

# %%

DYNA_CALIBRATION_SCALE = 0.40668870885911723
DYNA_CALIBRATION_OFFSET = -0.02243122548498261

dyna1_path = (
    f"/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_results/{pdb_code}/{pdb_code}-Dyna1.csv"
)
dyna1_df = pd.read_csv(dyna1_path)
dyna1_proba = dyna1_df.iloc[:, 2].values
dyna1_result = (DYNA_CALIBRATION_SCALE * dyna1_proba) + DYNA_CALIBRATION_OFFSET
sequence = "".join(dyna1_df.iloc[:, 1].values)


# %%
bioemu_path = f"/mnt/home/ssledzieski/GitHub/bioemu/rshp_results/{pdb_code}_10"
bioemu_traj = md.load(f"{bioemu_path}/samples.xtc", top=f"{bioemu_path}/topology.pdb")
bioemu_rmsf = compute_rmsf(bioemu_traj)


# %%

plt.figure(figsize=(10, 6))
plt.plot(labels["rmsf"], label="True RMSF")
plt.plot(both_result["rmsf"].detach().cpu().numpy().squeeze(), label="RocketSHP RMSF")
plt.plot(dyna1_result, label="Dyna-1 RMSF")
plt.plot(bioemu_rmsf, label="BioEMu RMSF")
plt.ylabel("RMSF")
plt.xlabel("Residue")
plt.title(f"Protein: {key}")
plt.legend()
if savefig:
    plt.savefig(
        config.FIGURES_DIR / "atlas_single" / f"{pdb_code_save}_rmsf_comparison.svg",
    )
# plt.show()


# %%

spearman = spearmanr(
    labels["rmsf"], both_result["rmsf"].detach().cpu().numpy().squeeze()
)
pearson = pearsonr(labels["rmsf"], both_result["rmsf"].detach().cpu().numpy().squeeze())
mse = (
    (labels["rmsf"] - both_result["rmsf"].detach().cpu().numpy().squeeze()) ** 2
).mean()

logger.info(f"Protein: {key}")
logger.info(f"Spearman: {spearman}")
logger.info(f"Pearson: {pearson}")
logger.info(f"MSE: {mse}")

# %%

squared_label_type = "ca_dist"

true_sqform = labels[squared_label_type].squeeze().T
predicted_sqform = both_result[squared_label_type].cpu().squeeze().T

# increase font size
with plt.rc_context({"font.size": 30}):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    im1 = axes[0].imshow(
        1 - true_sqform, vmin=0, vmax=1, aspect="auto", cmap="coolwarm"
    )
    axes[0].set_xlabel("True")
    im2 = axes[1].imshow(
        1 - predicted_sqform, vmin=0, vmax=1, aspect="auto", cmap="coolwarm"
    )
    axes[1].set_xlabel("Predicted")

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.ax.set_ylabel("CA Distances", rotation=-90, va="bottom")

    if savefig:
        plt.savefig(
            config.FIGURES_DIR
            / "atlas_single"
            / f"{pdb_code_save}_ca_dist_comparison.svg",
            bbox_inches="tight",
        )
    # plt.show()


# %%

# GCC-LMI
# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
squared_label_type = "gcc_lmi"

true_sqform = labels[squared_label_type].squeeze().T
predicted_sqform = both_result[squared_label_type].cpu().squeeze().T

# ax[0].imshow(true_sqform)
#
# image = ax[1].imshow(predicted_sqform)
# ax[1].set_xlabel("Predicted")
# plt.colorbar(image)

# increase font size
with plt.rc_context({"font.size": 30}):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    im1 = axes[0].imshow(true_sqform, vmin=0, vmax=1, aspect="auto", cmap="coolwarm")
    axes[0].set_xlabel("True")
    im2 = axes[1].imshow(
        predicted_sqform, vmin=0, vmax=1, aspect="auto", cmap="coolwarm"
    )
    axes[1].set_xlabel("Predicted")

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.ax.set_ylabel("Autocorrelation of Distances", rotation=-90, va="bottom")

    if savefig:
        plt.savefig(
            config.FIGURES_DIR
            / "atlas_single"
            / f"{pdb_code_save}_autocorr_comparison.svg",
            bbox_inches="tight",
        )
    # plt.show()


# %%


# Foldseek Structure Heterogeneity Profile
fig, ax = plt.subplots(2, 1, figsize=(15, 9))

true_shp = labels["shp"].squeeze().T
predicted_shp = both_result["shp"].cpu().squeeze().T

with plt.rc_context({"font.size": 18}):
    ax[0].imshow(true_shp, cmap="binary")
    ax[0].set_xlabel("True")
    ax[0].set_ylim(21, -1)
    ax[0].set_title("True SHP")
    ax[0].set_xlabel("Residue")
    ax[0].set_ylabel("FoldSeek State")

    ax[1].imshow(softmax(predicted_shp, dim=0), cmap="binary")
    # ax[1].imshow(predicted_shp, cmap="binary")
    ax[1].set_title("Predicted SHP")
    ax[1].set_ylabel("FoldSeek State")
    ax[1].set_xlabel("Residue")
    ax[1].set_ylim(21, -1)

    if savefig:
        plt.savefig(
            config.FIGURES_DIR / "atlas_single" / f"{pdb_code_save}_shp_comparison.svg",
            bbox_inches="tight",
        )
# plt.show()


# %%

ent = entropy(true_shp)
p_ent = entropy(predicted_shp)
# print(ent.max(), p_ent.max(), labels["rmsf"].max())
plt.plot(labels["rmsf"], label="RMSF")
plt.plot(ent, label="SHP Entropy")
plt.legend()
# plt.show()


# %%


GNM_ROOT = (
    "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/gaussian_net_models"
)
gnm_covar = f"{GNM_ROOT}/{pdb_code[:2]}/{pdb_code}_gnm.npz"
gnm_data = np.load(gnm_covar)
gnm_covar = gnm_data["covar"]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(gnm_covar, vmin=0, vmax=1, aspect="auto", cmap="coolwarm")

# ---

# %%
