#%%
import sys
import h5py
import torch
from loguru import logger
from tqdm import tqdm
from datasets import Dataset

from rocketshp import config
from rocketshp.data.utils import update_h5_dataset

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
MDCATH_H5 = MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5"

mdcath_derivatives_data = Dataset.load_from_disk(str(MDCATH_PROCESSED_DATA_DIR / "mdcath_derivatives_v2"))
# mdcath_shp_data = Dataset.load_from_disk(str(MDCATH_PROCESSED_DATA_DIR / "esm_shp/dataset"))
mdcath_foldseek_shp_data = Dataset.load_from_disk(str(MDCATH_PROCESSED_DATA_DIR / "fs_shp/dataset"))
#%%
# Build file
with h5py.File(MDCATH_H5, "a") as h5file:
    
    # Basic Derivatives
    # for replicate in tqdm(mdcath_derivatives_data, desc="Basic Derivatives"):
    #     pdb_id = replicate["pdb_code"]
    #     rep = replicate["rep"]
    #     temp = replicate["temp"]
    #     # logger.info(f"Autocorr: Processing {pdb_id} R{rep}")
        
    #     for key in ["rmsf", "ca_dist", "autocorr"]:
    #         update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/{key}", replicate[key])

    # for replicate in tqdm(mdcath_shp_data, desc="SHP"):
        # pdb_id = replicate["pdb_code"]
        # rep = replicate["rep"]
        # preshp = replicate["shp"]
        # shp = convert_to_normalized_shp(preshp)
        # # logger.info(f"SHP: Processing {pdb_id} R{rep}")
        # update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/esm_shp", shp, overwrite=True)

    for replicate in tqdm(mdcath_foldseek_shp_data, desc="Foldseek SHP"):
        pdb_id = replicate["pdb_code"]
        rep = replicate["rep"]
        shp = replicate["fs_shp"]
        temp = replicate["temp"]
        # logger.info(f"SHP: Processing {pdb_id} R{rep}")
        update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/fs_shp", shp, overwrite=True)
        update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/shp", shp, overwrite=True)

    # Generalized Dynamic Correlation
    # STEP = 10
    # DO_LOCAL_ALIGN = True
    # local_suff = "local_" if DO_LOCAL_ALIGN else ""

    # for replicate in tqdm(atlas_derivatives_data, desc="Generalized Dynamic Correlation"):
    #     # logger.info(f"DynCorr: Processing {pdb_code} R{rep}")
    #     pdb_id = replicate["pdb_code"]
    #     rep = replicate["rep"]
        
    #     gen_correlation_file = str(ATLAS_DATA_DIR / pdb_id[:2] / f"{pdb_id}_{rep}_{local_suff}step{STEP}_corr_matrix.pt")
    #     gen_correlation = torch.load(gen_correlation_file, weights_only=True)
    #     update_h5_dataset(h5file, f"{pdb_id}/R{rep}/dyn_corr", gen_correlation)

    # Sequence Embeddings
    # for pdb_code in tqdm(list(set(mdcath_derivatives_data["pdb_code"])), desc="Sequence embeddings"):
    #     seq_file = MDCATH_PROCESSED_DATA_DIR / "seq_embeddings" / pdb_code / f"{pdb_code}.seq"
    #     seq_embedding = torch.load(seq_file, map_location="cpu", weights_only=True).squeeze()[1:-1]
    #     update_h5_dataset(h5file, f"{pdb_code}/embedding", seq_embedding, overwrite=True)

    # # Structural Embeddings
    # for pdb_code in tqdm(list(set(mdcath_derivatives_data["pdb_code"])), desc="Structural embeddings"):
    #     struct_file = MDCATH_PROCESSED_DATA_DIR / "struct_embeddings" / pdb_code / f"{pdb_code}.struct"
    #     struct_embedding = torch.load(struct_file, map_location="cpu", weights_only=True)
    #     for k,v in struct_embedding.items():
    #         update_h5_dataset(h5file, f"{pdb_code}/struct_embedding/{k}", v.squeeze(), overwrite=True)