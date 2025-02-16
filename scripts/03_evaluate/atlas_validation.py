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
from rocketshp.modeling.architectures import DynCorrModelWithTemperature

# %% Script inputs
CONFIG_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/configs/fs_shp_kl_reweight2.yml"
MODEL_CHECKPOINT_FILE = checkpoint_file = (
    "/mnt/home/ssledzieski/Projects/rocketshp/models/fs_shp_pred_kl_reweighted2/model-epoch=42-val_loss=2.11.pt.ckpt"
)
OUTPUT_DIRECTORY = (
    f"{config.EVALUATION_DATA_DIR}/evaluations/fs_shp_pred_kl_reweighted2"
)

# CONFIG_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/configs/seqonly_config.yml"
# MODEL_CHECKPOINT_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/models/fs_shp_pred_kl_seqonly/model-epoch=30-val_loss=2.18.pt.ckpt"
# OUTPUT_DIRECTORY = f"{config.EVALUATION_DATA_DIR}/evaluations/fs_shp_pred_kl_seqonly"

# CONFIG_FILE = sys.argv[1]
# MODEL_CHECKPOINT_FILE = sys.argv[2]


# %% Data loading and prep
GNM_ROOT = (
    "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/gaussian_net_models"
)
PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(CONFIG_FILE))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("Loading data...")
adl = ATLASDataModule(
    config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
    seq_features=True,
    struct_features=True,
    batch_size=1,
    num_workers=PARAMS.num_data_workers,
    train_pct=PARAMS.train_pct,
    val_pct=PARAMS.val_pct,
    random_seed=PARAMS.random_seed,
    struct_stage=PARAMS.struct_stage,
)
adl.setup("train")
ads = adl.dataset

# %% Load model
logger.info("Loading model...")
model = DynCorrModelWithTemperature.load_from_checkpoint(
    MODEL_CHECKPOINT_FILE, strict=True
)
model = model.to(device)

# %% Inference loop


def run_inference(
    model, key, feats, labels, save_root: str = ".", img_ext: str = "png"
):
    model.eval()
    with torch.inference_mode():
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}
        y_hat = model(feats)

    # Move to cpu and squeeze
    y_hat = {k: v.detach().cpu().squeeze() for k, v in y_hat.items()}

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
        "shp_kl": shp_kl,
    }

    # save results as txt file
    with open(f"{save_root}/{key}/{key}_results.txt", "w") as f:
        for k, v in return_dict.items():
            f.write(f"{k}: {v}\n")

    return return_dict


# %% Run inference in loop

test_data = adl.val_data
all_results = {}

for i, (feats, labels) in enumerate(
    tqdm(test_data, desc="Evaluating validation set...")
):
    key = adl.dataset.samples[test_data.indices[i]]
    key_under = key.replace("/", "_")
    os.makedirs(f"{OUTPUT_DIRECTORY}/{key_under[:2]}/{key_under}", exist_ok=True)
    rdict = run_inference(
        model,
        key_under,
        feats,
        labels,
        save_root=f"{OUTPUT_DIRECTORY}/{key_under[:2]}",
        img_ext="svg",
    )
    all_results[key] = rdict

# import pickle as pk
# with open(f"{OUTPUT_DIRECTORY}/results.pkl", "wb") as f:
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

with h5py.File(config.PROCESSED_DATA_DIR / "atlas" / "atlas_processed.h5", "r") as f:
    seq_lengths_ = {}
    keys_ = list(f.keys())
    for k in tqdm(keys_):
        seq_lengths_[k] = f[k]["embedding"].shape[0]

# %%

# create 5 bins with equal numbers of proteins by sequence length
seq_lengths = pd.Series(seq_lengths_)
seq_lengths = seq_lengths.sort_values()
seq_lengths = seq_lengths.reset_index()

bins = [
    ("<100 residues", (0, 100)),
    ("101-150 residues", (101, 150)),
    ("151-250 residues", (151, 250)),
    ("251-350 residues", (251, 350)),
    (">350 residues", (351, seq_lengths.max()[0])),
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


# %%

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
sns.scatterplot(
    data=df,
    x="spearman",
    y=-np.log10(df["spearman_p"]),
    ax=ax[0],
    hue="Sequence Length",
    s=8,
)
sns.scatterplot(
    data=df,
    x="pearson",
    y=-np.log10(df["pearson_p"]),
    ax=ax[1],
    hue="Sequence Length",
    s=8,
    legend=False,
)
ax[0].set_xlabel("RMSF Spearman Correlation")
ax[0].set_ylabel("-log10(p)")
ax[1].set_xlabel("RMSF Pearson Correlation")
ax[1].set_ylabel("-log10(p)")
# sns.despine()
# plt.savefig(f"{OUTPUT_DIRECTORY}/summary_rmsf.svg", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# plt.figure(figsize=(5, 10))
sns.boxplot(
    data=df, y="autocorr_mse", x="Sequence Length", hue="Sequence Length", ax=ax[2]
)
# rotate xticks
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha="right")
ax[2].set_xlabel("")
ax[2].set_ylabel("Average MSE of Autocorrelation")
# sns.despine()
# plt.savefig(f"{OUTPUT_DIRECTORY}/summary_autocorr.svg", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# plt.figure(figsize=(5, 10))
sns.boxplot(data=df, y="shp_kl", x="Sequence Length", hue="Sequence Length", ax=ax[3])
ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=45, ha="right")
ax[3].set_xlabel("")
ax[3].set_ylabel("Average KL Divergence of SHP")
sns.despine()

plt.savefig(f"{OUTPUT_DIRECTORY}/summary_shp_all.svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
# %%
