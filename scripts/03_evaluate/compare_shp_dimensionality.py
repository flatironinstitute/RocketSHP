# %% Load packages
import os

import matplotlib.pyplot as plt
import torch
from loguru import logger
from scipy.stats import entropy
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#537eba', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 16,
        # "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

# %% Script inputs
EVAL_KEY = "gcc_lmi_5a_20250423"

CONFIG_FILE = (
    "/mnt/home/ssledzieski/Projects/rocketshp/configs/rocketshp_gcc_20250421.yml"
)
CHECKPOINT_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/models/GCC_LMI_5alpha/model-epoch=48-val_loss=2.57.pt.ckpt"
OUTPUT_DIRECTORY = f"{config.EVALUATION_DATA_DIR}/evaluations/{EVAL_KEY}"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
PARAMS = config.DEFAULT_PARAMETERS

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
handle = ads._handle
keys = list(handle.keys())

fs_shps = []
esm_shps = []
for k in tqdm(keys):
    for rep in ["R1", "R2", "R3"]:
        fs_shps.append(handle[k][rep]["fs_shp"][:])
        esm_shps.append(handle[k][rep]["esm_shp"][:])
# %% Compute entropy of each SHP
fs_entropy = []
esm_entropy = []

for fs_shp, esm_shp in tqdm(zip(fs_shps, esm_shps), total=len(fs_shps)):
    fs_entropy.append(entropy(fs_shp, axis=1).mean())
    esm_entropy.append(entropy(esm_shp, axis=1).mean())

# %% Plot

fig, ax = plt.subplots(figsize=(16, 8))
ax.hist(fs_entropy, bins=100, color="blue", alpha=0.5, label="FS-SHP")
ax.hist(esm_entropy, bins=100, color="orange", alpha=0.5, label="ESM-SHP")
ax.set_xlabel("Entropy")
ax.set_ylabel("Count")
ax.legend()

# %%
