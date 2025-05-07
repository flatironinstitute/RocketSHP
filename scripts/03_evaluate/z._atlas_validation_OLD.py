# %% Load packages
import os
import pickle as pk

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import kl_div, log_softmax, softmax
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.utils import train_test_split_foldseek
from rocketshp.modeling.architectures import RocketSHPModel

plt.rcParams.update({
    # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#537eba', '#56B4E9'])",
    "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])", # simons foundation
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    # "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    })

# %% Script inputs
CONFIG_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/configs/rocketshp_pretrained_20250215_v0.yml"
OUTPUT_DIRECTORY = (
    f"{config.EVALUATION_DATA_DIR}/evaluations/fs_shp_pred_kl_reweighted2"
)

# %% Data loading and prep
# GNM_ROOT = (
#     "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/gaussian_net_models"
# )
PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(CONFIG_FILE))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# logger.info("Loading data...")
# adl = ATLASDataModule(
#     config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
#     seq_features=True,
#     struct_features=True,
#     batch_size=1,
#     num_workers=PARAMS.num_data_workers,
#     train_pct=PARAMS.train_pct,
#     val_pct=PARAMS.val_pct,
#     random_seed=PARAMS.random_seed,
#     struct_stage=PARAMS.struct_stage,
# )
# adl.setup("train")
# ads = adl.dataset

# %% Load model
# logger.info("Loading model...")
# model = RocketSHPModel.load_from_checkpoint("latest", strict=True)
# model = model.to(device)

# %% Inference loop

def run_inference(model, feats, device="cuda:0"):
    """
    Run a forward pass of the model
    """

    model.eval()
    with torch.inference_mode():
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}
        y_hat = model(feats)
    # Move to cpu and squeeze
    y_hat = {k: v.detach().cpu().squeeze() for k, v in y_hat.items()}
    return y_hat

def plot_inference(
    model, key, feats, labels, save_root: str = ".", img_ext: str = "png", device="cuda:0"
):

    # Compute forward pass
    y_hat = run_inference(model, feats, device)

    # Plot RMSF
    plt.figure(figsize=(10, 5))
    plt.plot(labels["rmsf"], label="RMSF")
    plt.plot(y_hat["rmsf"], label="Predicted")
    plt.legend()
    sns.despine()
    plt.savefig(f"{save_root}/{key}/{key}_rmsf.{img_ext}", dpi=300, bbox_inches="tight")
    plt.close()

    # Compute RMSF statistics
    spearman = spearmanr(labels["rmsf"], y_hat["rmsf"])
    pearson = pearsonr(labels["rmsf"], y_hat["rmsf"])
    rmsf_mse = ((labels["rmsf"] - y_hat["rmsf"]) ** 2).mean()

    # logger.info(f"Protein: {key}")
    # logger.info(f"Spearman: {spearman}")
    # logger.info(f"Pearson: {pearson}")
    # logger.info(f"RMSF MSE: {rmsf_mse}")

    # Plot Autocorrelation
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    true_sqform = labels["autocorr"]
    predicted_sqform = y_hat["autocorr"]

    # compute autocorr MSE
    autocorr_mse = ((true_sqform - predicted_sqform) ** 2).mean()
    autocorr_mae = (true_sqform - predicted_sqform).abs().mean()
    # logger.info(f"Protein: {key} Autocorr MAE: {autocorr_mae}")
    # logger.info(f"Autocorr MSE: {autocorr_mse}")

    ax[0].imshow(true_sqform)
    ax[0].set_xlabel("True")
    ax[1].imshow(predicted_sqform)
    ax[1].set_xlabel("Predicted")
    plt.savefig(f"{save_root}/{key}/{key}_autocorr.{img_ext}")
    plt.close()

    # Foldseek Structure Heterogeneity Profile
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    true_shp = labels["shp"].T
    predicted_shp = y_hat["shp"].T

    # compute shp KL divergence
    shp_kl = kl_div(
        log_softmax(predicted_shp, dim=0), true_shp, log_target=False, reduction="none"
    )
    shp_kl = shp_kl.sum(dim=0).mean()

    # logger.info(f"SHP KL Divergence: {shp_kl}")

    ax[0].imshow(true_shp, cmap="binary")
    ax[0].set_xlabel("True")
    ax[1].imshow(softmax(predicted_shp, dim=0), cmap="binary")
    ax[1].set_xlabel("Predicted")
    plt.savefig(f"{save_root}/{key}/{key}_shp.{img_ext}")
    plt.close()

    pdb_code = "_".join(key.split("_")[:2])
    gnm_covar = f"{GNM_ROOT}/{pdb_code[:2]}/{pdb_code}_gnm.npz"
    gnm_data = np.load(gnm_covar)
    gnm_covar = gnm_data["covar"]
    plt.imshow(gnm_covar)
    plt.savefig(f"{save_root}/{key}/{key}_gnm.{img_ext}")

    return_dict = {
        "spearman": spearman,
        "pearson": pearson,
        "rmsf_mse": rmsf_mse,
        "autocorr_mse": autocorr_mse,
        "autocorr_mae": autocorr_mae,
        "shp_kl": shp_kl,
    }

    with open(f"{save_root}/{key}/{key}_results.txt", "w") as f:
        for k, v in return_dict.items():
            f.write(f"{k}: {v}\n")

    return return_dict


# %% Run inference in loop

# test_data = adl.val_data
# all_results = {}

# for i, (feats, labels) in enumerate(
# tqdm(test_data, desc="Evaluating validation set...")
# ):
#     key = adl.dataset.samples[test_data.indices[i]]
#     key_under = key.replace("/", "_")
    # os.makedirs(f"{OUTPUT_DIRECTORY}/{key_under[:2]}/{key_under}", exist_ok=True)
    # rdict = plot_inference(
        # model,
        # key_under,
        # feats,
        # labels,
        # save_root=f"{OUTPUT_DIRECTORY}/{key_under[:2]}",
        # img_ext="svg",
    # )
    # all_results[key] = rdict
    # fw_results = run_inference(model, feats)
    # all_results[key] = fw_results

# import pickle as pk
# with open(f"{OUTPUT_DIRECTORY}/inference_results.pkl", "wb") as f:
#     pk.dump(all_results, f)
# %%

with open(f"{OUTPUT_DIRECTORY}/results.pkl", "rb") as f:
    all_results = pk.load(f)

# %%
codes = list(all_results.keys())
spearman = [all_results[c]["spearman"][0] for c in codes]
spearman_p = [all_results[c]["spearman"][1] for c in codes]
pearson = [all_results[c]["pearson"][0] for c in codes]
pearson_p = [all_results[c]["pearson"][1] for c in codes]
rmsf_mse = [all_results[c]["rmsf_mse"].item() for c in codes]
autocorr_mse = [all_results[c]["autocorr_mse"].item() for c in codes]
shp_kl = [all_results[c]["shp_kl"].item() for c in codes]
df = pd.DataFrame(
    {
        "pdb_code": codes,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "pearson": pearson,
        "pearson_p": pearson_p,
        "rmsf_mse": rmsf_mse,
        "autocorr_mse": autocorr_mse,
        "shp_kl": shp_kl,
    }
)
df.to_csv(f"{OUTPUT_DIRECTORY}/results.csv", index=False)
# %% get sequence lengths

# with h5py.File(config.PROCESSED_DATA_DIR / "atlas" / "atlas_processed.h5", "r") as f:
#     seq_lengths_ = {}
#     keys_ = list(f.keys())
#     for k in tqdm(keys_, desc="Getting sequence lengths...", unit="protein"):
#         seq_lengths_[k] = f[k]["embedding"].shape[0]

seq_lengths_ = {}
import torch
from pathlib import Path
for fi in tqdm(Path("/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/seq_embeddings").glob("*/*.seq"), desc="Getting sequence lengths...", unit="protein"):
    f = torch.load(fi)
    k = fi.stem
    seq_lengths_[k] = f.shape[1]

# %%

# create 5 bins with equal numbers of proteins by sequence length
seq_lengths = pd.Series(seq_lengths_)
seq_lengths = seq_lengths.sort_values()
seq_lengths = seq_lengths.reset_index()

bins = [
    ("<100 residues", (0, 99)),
    ("100-149 residues", (100, 149)),
    ("150-249 residues", (150, 249)),
    ("250-349 residues", (250, 349)),
    (">350 residues", (350, seq_lengths.max()[0])),
]

# for each sequence in the dataset, assign it to a bin
seq_bins = []
for i, row in seq_lengths.iterrows():
    seq_len = row[0]
    for b_label, (b_min, b_max) in bins:
        if b_min <= seq_len <= b_max:
            seq_bins.append(b_label)
            break
seq_lengths["bin"] = seq_bins
seq_lengths.set_index("index", inplace=True)

# add sequence lengths to df
df["seq_length"] = df.apply(
    lambda x: seq_lengths.loc[x["pdb_code"].split("/")[0]][0], axis=1
)
df["Sequence Length"] = df.apply(
    lambda x: seq_lengths.loc[x["pdb_code"].split("/")[0]]["bin"], axis=1
)

#%% Get train/test splits

ATLAS_FOLDSEEK_CLUSTERS_FILE = (
    config.PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
)
foldseek_clusters = pd.read_csv(
    ATLAS_FOLDSEEK_CLUSTERS_FILE, sep="\t", header=None
)

train_set, val_set, test_set = train_test_split_foldseek(
    foldseek_clusters,
    train_pct=PARAMS.train_pct,
    val_pct=PARAMS.val_pct,
    random_seed=PARAMS.random_seed,
)
train_lens = [seq_lengths_[k] for k in train_set]
val_lens = [seq_lengths_[k] for k in val_set]
test_lens = [seq_lengths_[k] for k in test_set]

# %%
# Plot sequence length histogram with cutoffs

plt.figure(figsize=(10, 5))
# plt.hist(seq_lengths[0], bins=50)
plt_bins = np.arange(0, 2100, 50)
plt.hist(train_lens, bins=plt_bins, alpha=0.8, label="Train")
plt.hist(val_lens, bins=plt_bins, alpha=0.8, label="Test")
# plt.hist(test_lens, bins=plt_bins, alpha=0.8, label="Test")
plt.legend()

# for each sequence bin, add a vertical line and a text annotation of that line
for b in bins:
    b_min, b_max = b[1]
    plt.axvline(x=b_min, color="black", linestyle="--")
    plt.text(((b_min + b_max) // 2) - 5, 275, b[0], rotation=45, ha="left", va="center", fontsize=10)
plt.xlabel("Sequence Length")
plt.ylabel("Number of Proteins")
# plt.title("Distribution of Sequence Lengths")
# plt.savefig(config.FIGURES_DIR / "atlas_all" / "seq_length_histogram.svg", dpi=300, bbox_inches="tight")
plt.show()

# %%

fig, ax = plt.subplots(1, 4, figsize=(25, 8))

sns.boxplot(
    data=df,
    x="Sequence Length",
    y="rmsf_mse",
    color="white",
    linecolor="black",
    ax=ax[0],
    legend=False,
    showfliers=False,
    order=[b[0] for b in bins]
)

sns.stripplot(
    data=df,
    x="Sequence Length",
    y="rmsf_mse",
    hue="Sequence Length",
    ax=ax[0],
    legend=False,
    order=[b[0] for b in bins]
)

ax[0].set_ylabel("Mean Squared Error of RMSF")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")
ax[0].set_yscale("log")

sns.scatterplot(
    data=df,
    x="spearman",
    y=-np.log10(df["spearman_p"]),
    ax=ax[1],
    hue="Sequence Length",
    s=8,
    legend=True,
    hue_order=[b[0] for b in bins]
)

ax[1].set_xlabel("Spearman Correlation of RMSF")
ax[1].set_ylabel("-log10(p)")

sns.boxplot(
    data=df, y="autocorr_mse", x="Sequence Length",
    ax=ax[2],
    color="white",
    linecolor="black",
    legend=False,
    showfliers=False,
    order=[b[0] for b in bins], hue_order=[b[0] for b in bins]
)

sns.stripplot(
    data=df, y="autocorr_mse", x="Sequence Length", hue="Sequence Length", ax=ax[2],
    order=[b[0] for b in bins], hue_order=[b[0] for b in bins]
)
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha="right")
ax[2].set_xlabel("")
ax[2].set_ylabel("Mean Squared Error of Autocorrelation")

sns.boxplot(data=df, y="shp_kl", x="Sequence Length",
            ax=ax[3],
            color="white",
            linecolor="black",
            legend=False,
            showfliers=False,
            order=[b[0] for b in bins], hue_order=[b[0] for b in bins])

sns.stripplot(data=df, y="shp_kl", x="Sequence Length", hue="Sequence Length",ax=ax[3],
            order=[b[0] for b in bins], hue_order=[b[0] for b in bins])

ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=45, ha="right")
ax[3].set_xlabel("")
ax[3].set_ylabel("KL Divergence of SHP")
sns.despine()

# from statannotations.Annotator import Annotator
# from itertools import combinations

# import itertools
# def pairwise(iterable):
#     "s -> (s0, s1), (s1, s2), (s2, s3), ..."
#     a, b = itertools.tee(iterable)
#     next(b, None)
#     return zip(a, b) 

# pairs = list(pairwise(
#     [b[0] for b in bins]
# ))

# annotator = Annotator(ax[0], pairs, data=df, x="Sequence Length", y="rmsf_mse", order=[b[0] for b in bins])
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()

# annotator = Annotator(ax[2], pairs, data=df, x="Sequence Length", y="autocorr_mse", order=[b[0] for b in bins])
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()

# annotator = Annotator(ax[3], pairs, data=df, x="Sequence Length", y="shp_kl", order=[b[0] for b in bins])
# annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
# annotator.apply_and_annotate()

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "atlas_all" / "atlas_rshp_summary.svg", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()
# %%
