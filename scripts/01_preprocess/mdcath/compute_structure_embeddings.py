#%%
from tqdm import tqdm
import os
import torch

from rocketshp import config
from rocketshp.features import esm3_vqvae
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_structure_vae
from rocketshp.data.mdcath import convert_to_files

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
STRUCT_EMBEDDING_PATH = MDCATH_PROCESSED_DATA_DIR / "struct_embeddings"
os.makedirs(STRUCT_EMBEDDING_PATH, exist_ok=True)

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))

#%%

vae = get_structure_vae()
vae.eval()

with torch.inference_mode():
    for mdcf in tqdm(mdcath_files, desc="Structure Embeddings"):
        pdb_code = str(mdcf).split("_")[-1].split(".")[0]
        # logger.info(str(mdcf))
        pdb_f, xtc_f = convert_to_files(str(mdcf), temp_list=[320], replica_list=[0], directory="/tmp")
        struct_file = STRUCT_EMBEDDING_PATH / pdb_code / f"{pdb_code}.struct"
        if struct_file.exists():
            continue
        os.makedirs(struct_file.parent, exist_ok=True)
        chain = ProteinChain.from_pdb(pdb_f)
        struct_embedding = {}
        for stage in ["encoded", "pre-quantized", "quantized"]:
            struct_embedding[stage] = esm3_vqvae(chain, vae, stage=stage)
        torch.save(struct_embedding, struct_file)
# %%
