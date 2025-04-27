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
from multiprocessing import Pool

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.utils import train_test_split_foldseek
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.metrics import (
    ipsen_mikhailov_distance,
    pearson,
    spearman,
    mse,
    mae,
    kl_divergence_2d,
    wasserstein_2d,
)

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
EVAL_KEY = "cadist_20250427"

CONFIG_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/configs/20250426_cadist_fixed.yml"

# CHECKPOINT_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/models/cadist_sqloss/model-epoch=43-val_loss=0.70.pt.ckpt"
CHECKPOINT_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/models/cadist_fixed/model-epoch=42-val_loss=1.07.pt.ckpt"

OUTPUT_DIRECTORY = config.EVALUATION_DATA_DIR / "evaluations" / EVAL_KEY
FIGURES_DIRECTORY = config.REPORTS_DIR / EVAL_KEY / "figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(CONFIG_FILE))

# %% Load data
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
model = RocketSHPModel.load_from_checkpoint(CHECKPOINT_FILE, strict=True)
model = model.to(device)

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

# %% Run inference in loop

valid_results = {}
valid_labels = {}
test_results = {}
test_labels = {}

for i, (feats, labels) in enumerate(
    tqdm(adl.val_data, desc="Evaluating validation set...")
):
    key = adl.dataset.samples[adl.val_data.indices[i]]
    key_under = key.replace("/", "_")
    rdict = run_inference(model, feats)
    valid_results[key] = rdict
    valid_labels[key] = labels

for i, (feats, labels) in enumerate(
    tqdm(adl.test_data, desc="Evaluating test set...")
):
    key = adl.dataset.samples[adl.test_data.indices[i]]
    key_under = key.replace("/", "_")
    rdict = run_inference(model, feats)
    test_results[key] = rdict
    test_labels[key] = labels

import pickle as pk
logger.info("Saving results...")
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_valid_inference_results.pkl", "wb") as f:
    pk.dump(valid_results, f)
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_test_inference_results.pkl", "wb") as f:
    pk.dump(test_results, f)

# %% Load pickle
logger.info("Loading results...")
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_valid_inference_results.pkl", "rb") as f:
    valid_results = pk.load(f)
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_test_inference_results.pkl", "rb") as f:
    test_results = pk.load(f)

all_results = {**valid_results, **test_results}
all_labels = {**valid_labels, **test_labels}

# %% Get sequence lengths
logger.info("Getting sequence lengths...")
seq_lengths = []
for k, v in valid_results.items():
    seq_lengths.append((k, "valid", v["rmsf"].shape[0]))
for k, v in test_results.items():
    seq_lengths.append((k, "test", v["rmsf"].shape[0]))
for i, f in enumerate(adl.train_data):
    k = adl.dataset.samples[adl.train_data.indices[i]]
    seq_lengths.append((k, "train", f[1]["rmsf"].shape[0]))

seq_lengths = pd.DataFrame(
    seq_lengths, columns=["key", "split", "length"]
)

# %% Assign sequences to length bins

bins = [
    ("<100 residues", (0, 99)),
    ("100-149 residues", (100, 149)),
    ("150-249 residues", (150, 249)),
    ("250-349 residues", (250, 349)),
    (">350 residues", (350, seq_lengths["length"].max())),
]

for label, (low, high) in bins:
    seq_lengths.loc[
        (seq_lengths["length"] >= low) & (seq_lengths["length"] <= high), "bin"
    ] = label
seq_lengths["bin"] = pd.Categorical(
    seq_lengths["bin"], categories=[b[0] for b in bins], ordered=True
)
seq_lengths["length"] = seq_lengths["length"].astype(int)


# %% Plot sequence lengths

logger.info("Plotting sequence lengths...")
seq_lengths_unique = seq_lengths.copy()
seq_lengths_unique["key"] = seq_lengths_unique["key"].apply(lambda x: x.split("/")[0])
seq_lengths_unique = seq_lengths_unique.drop_duplicates()

plt.figure(figsize=(10, 5))
plt_bins = np.arange(0, 2100, 50)
plt.hist(seq_lengths_unique[seq_lengths_unique["split"] == "train"]["length"], bins=plt_bins, alpha=0.8, label="Train")
plt.hist(seq_lengths_unique[seq_lengths_unique["split"] == "valid"]["length"], bins=plt_bins, alpha=0.8, label="Valid")
plt.hist(seq_lengths_unique[seq_lengths_unique["split"] == "test"]["length"],  bins=plt_bins, alpha=0.8, label="Test")
for b in bins:
    b_min, b_max = b[1]
    plt.axvline(x=b_min, color="black", linestyle="--")
    plt.text(((b_min + b_max) // 2) - 5, 275, b[0], rotation=45, ha="left", va="center", fontsize=10)
plt.xlabel("Sequence Length")
plt.ylabel("Number of Proteins")
plt.savefig(config.FIGURES_DIR / "atlas_all" / "seq_length_histogram.svg", dpi=300, bbox_inches="tight")
plt.legend()
plt.show()

# %% Performance

def compute_metric_single(key, target, pred):
    metrics = {}
    # print(i, key)

    target = {k: v.cpu().numpy() for k, v in target.items()}
    pred = {k: v.cpu().numpy() for k, v in pred.items()}

    # Compute RMSF metrics
    pearson_corr, pearson_p = pearson(target["rmsf"], pred["rmsf"])
    spearman_corr, spearman_p = spearman(target["rmsf"], pred["rmsf"])
    metrics.update({
        "rmsf_pearson_r": pearson_corr,
        "rmsf_pearson_p": pearson_p,
        "rmsf_spearman_r": spearman_corr,
        "rmsf_spearman_p": spearman_p,
        "rmsf_mse": mse(target["rmsf"], pred["rmsf"])
    })

    # Compute GCC LMI metrics
    metrics.update({
        "gcc_mse": mse(target["gcc_lmi"], pred["gcc_lmi"]),
        "gcc_mae": mae(target["gcc_lmi"], pred["gcc_lmi"]),
        "gcc_im_dist": ipsen_mikhailov_distance(target["gcc_lmi"], pred["gcc_lmi"]),
    })

    # Compute SHP metrics
    metrics.update({
        "shp_mse": mse(target["shp"], pred["shp"]),
        "shp_mae": mae(target["shp"], pred["shp"]),
        "shp_kl_div": kl_divergence_2d(torch.from_numpy(target["shp"]), torch.from_numpy(pred["shp"])),
        "shp_wasserstein": wasserstein_2d(target["shp"], pred["shp"]),
    })

    return metrics

def compute_metrics(labels, predictions, save_path=None):
    all_metrics = {}

    zip_iterator = tqdm(zip(labels.keys(), labels.values(), predictions.values()),total=len(labels))

    # with Pool(16) as p:
        # metric_list = p.starmap(compute_metric_single, zip_iterator)
    # for k, m in zip(labels.keys(), metric_list):
        # all_metrics[k] = m

    for (key, target, pred) in zip_iterator:
        all_metrics[key] = compute_metric_single(key, target, pred)

    all_metrics = pd.DataFrame(all_metrics).T
    all_metrics = all_metrics.reset_index()
    all_metrics = pd.merge(all_metrics, seq_lengths, left_on="index", right_on="key", how="inner")
    all_metrics = all_metrics.rename({"bin": "Sequence Length"}, axis=1)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pk.dump(all_metrics, f)
    logger.info(f"Metrics saved to {save_path}")

    return all_metrics

# %% Compute metrics
logger.info("Computing validation metrics...")
valid_metrics = compute_metrics(valid_labels, valid_results, save_path=OUTPUT_DIRECTORY / f"{EVAL_KEY}_valid_metrics.pkl")
logger.info("Computing test metrics...")
test_metrics = compute_metrics(test_labels, test_results, save_path=OUTPUT_DIRECTORY / f"{EVAL_KEY}_test_metrics.pkl")

# %% Load metrics
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_valid_metrics.pkl", "rb") as f:
    valid_metrics = pk.load(f)
with open(OUTPUT_DIRECTORY / f"{EVAL_KEY}_test_metrics.pkl", "rb") as f:
    test_metrics = pk.load(f)

# %% Plot just RocketSHP results
logger.info("Plotting metrics...")
fig, ax = plt.subplots(1, 4, figsize=(28, 8))
plot_metrics = valid_metrics

sns.boxplot(
    data=plot_metrics,
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
    data=plot_metrics,
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
    data=plot_metrics,
    x="rmsf_spearman_r",
    y=-np.log10(plot_metrics["rmsf_spearman_p"]),
    ax=ax[1],
    hue="Sequence Length",
    s=8,
    legend=True,
    hue_order=[b[0] for b in bins]
)

ax[1].set_xlabel("Spearman Correlation of RMSF")
ax[1].set_ylabel("-log10(p)")

sns.boxplot(
    data=plot_metrics, y="gcc_im_dist", x="Sequence Length",
    ax=ax[2],
    color="white",
    linecolor="black",
    legend=False,
    showfliers=False,
    order=[b[0] for b in bins], hue_order=[b[0] for b in bins]
)

sns.stripplot(
    data=plot_metrics, y="gcc_im_dist", x="Sequence Length", hue="Sequence Length", ax=ax[2],
    order=[b[0] for b in bins], hue_order=[b[0] for b in bins]
)
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha="right")
ax[2].set_xlabel("")
ax[2].set_ylabel("Ipsen-Mikhailov Distance of GCC")

sns.boxplot(data=plot_metrics, y="shp_kl_div", x="Sequence Length",
            ax=ax[3],
            color="white",
            linecolor="black",
            legend=False,
            showfliers=False,
            order=[b[0] for b in bins], hue_order=[b[0] for b in bins])

sns.stripplot(data=plot_metrics, y="shp_kl_div", x="Sequence Length", hue="Sequence Length",ax=ax[3],
            order=[b[0] for b in bins], hue_order=[b[0] for b in bins])

ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=45, ha="right")
ax[3].set_xlabel("")
ax[3].set_ylabel("KL Divergence of SHP")
sns.despine()
plt.tight_layout()
plt.savefig(
    FIGURES_DIRECTORY / f"{EVAL_KEY}_rocketshp_metrics.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
