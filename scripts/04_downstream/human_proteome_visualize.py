#%% Imports
import torch
import glob
import gzip
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.utils import train_test_split_foldseek
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_model, get_tokenizers, get_structure_vae
from rocketshp.features import esm3_sequence, esm3_vqvae

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#%%
result_dir = config.PROCESSED_DATA_DIR / "rocketshp_afdb_human"
subdirectories = list(result_dir.iterdir()) 

# %%

# rocketshp_results = {}
# for subdir in subdirectories:
#     logger.info(f"Loading {subdir.name}")
#     proteins = list(subdir.glob("*.rshp.pt"))
#     for protein_file in tqdm(proteins):
#         pid = protein_file.stem.split("-")[1]
#         rocketshp_results[pid] = torch.load(protein_file, map_location="cpu")

# %%
