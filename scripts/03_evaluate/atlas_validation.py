#%% Load packages
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import numpy as np
from torch.nn.functional import kl_div, log_softmax, softmax

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.modeling.architectures import DynCorrModelWithTemperature

#%% Script inputs
CONFIG_FILE = "/mnt/home/ssledzieski/Projects/rocketshp/configs/default_config.yml"
MODEL_CHECKPOINT_FILE = checkpoint_file = "/mnt/home/ssledzieski/Projects/rocketshp/models/test_kl/model-epoch=01-val_loss=2.18.pt.ckpt"

# CONFIG_FILE = sys.argv[1]
# MODEL_CHECKPOINT_FILE = sys.argv[2]

#%% Data loading and prep
GNM_ROOT = "/mnt/home/ssledzieski/Projects/rocketshp/data/processed/atlas/gaussian_net_models"
PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(CONFIG_FILE))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

adl = ATLASDataModule(config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
                      seq_features=True, struct_features=True,
                      batch_size=1, num_workers=PARAMS.num_data_workers,
                      train_pct=PARAMS.train_pct, val_pct=PARAMS.val_pct,
                      random_seed=PARAMS.random_seed, struct_stage=PARAMS.struct_stage,
                      )
adl.setup("train")
ads = adl.dataset

#%% Load model
model = DynCorrModelWithTemperature.load_from_checkpoint(MODEL_CHECKPOINT_FILE, strict=True)
model = model.to(device)

#%% Inference loop

def run_inference(model, key, feats, labels, save_root: str = ".", img_ext: str = "png"):
    model.eval()
    with torch.inference_mode():
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}
        y_hat = model(feats)

    # Move to cpu and squeeze
    y_hat = {k: v.detach().cpu().squeeze() for k, v in y_hat.items()}

    # Plot RMSF
    plt.figure(figsize=(10, 5))
    plt.plot(labels["rmsf"],label="RMSF")
    plt.plot(y_hat["rmsf"],label="Predicted")
    plt.legend()
    sns.despine()
    plt.savefig(f"{save_root}/{key}/{key}_rmsf.{img_ext}", dpi=300, bbox_inches="tight")
    plt.close()

    # Compute RMSF statistics
    spearman = spearmanr(labels["rmsf"], y_hat["rmsf"])
    pearson = pearsonr(labels["rmsf"], y_hat["rmsf"])
    rmsf_mse = ((labels["rmsf"] - y_hat["rmsf"])**2).mean()

    #logger.info(f"Protein: {key}")
    #logger.info(f"Spearman: {spearman}")
    #logger.info(f"Pearson: {pearson}")
    #logger.info(f"RMSF MSE: {rmsf_mse}")

    # Plot Autocorrelation
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    true_sqform = labels["autocorr"]
    predicted_sqform = y_hat["autocorr"]
    
    # compute autocorr MSE
    autocorr_mse = ((true_sqform - predicted_sqform)**2).mean()
    #logger.info(f"Autocorr MSE: {autocorr_mse}")

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
    shp_kl = kl_div(log_softmax(predicted_shp, dim=0), true_shp, log_target=False, reduction='none')
    shp_kl = shp_kl.sum(dim=0).mean()

    #logger.info(f"SHP KL Divergence: {shp_kl}")

    ax[0].imshow(true_shp, cmap="binary")
    ax[0].set_xlabel("True")
    ax[1].imshow(softmax(predicted_shp,dim=0), cmap="binary")
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

#%% Run inference in loop

test_data = adl.val_data
all_results = {}

for i, (feats, labels) in enumerate(tqdm(test_data)):
    key = adl.dataset.samples[test_data.indices[i]]
    key_under = key.replace("/", "_")
    save_root = f"{config.EVALUATION_DATA_DIR}/atlas_validation"
    os.makedirs(f"{save_root}/{key_under}", exist_ok=True)
    rdict = run_inference(model, key_under, feats, labels, save_root=save_root)
    all_results[key] = rdict
# %%
