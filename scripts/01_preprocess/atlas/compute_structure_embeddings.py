#%%
from tqdm import tqdm
import os
import torch

from rocketshp import config
from rocketshp.features import esm3_vqvae
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_structure_vae

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
STRUCT_EMBEDDING_PATH = ATLAS_PROCESSED_DATA_DIR / "struct_embeddings"
os.makedirs(STRUCT_EMBEDDING_PATH, exist_ok=True)

pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]

#%%

vae = get_structure_vae()
vae.eval()

with torch.inference_mode():
    for pdb_file in tqdm(pdb_files, desc="Structure Embeddings"):
        pdb_code = pdb_file.stem
        seq_file = STRUCT_EMBEDDING_PATH / pdb_code[:2] / f"{pdb_code}.struct"
        if seq_file.exists():
            continue
        os.makedirs(seq_file.parent, exist_ok=True)
        chain = ProteinChain.from_pdb(pdb_file)
        struct_embedding = {}
        for stage in ["encoded", "pre-quantized", "quantized"]:
            struct_embedding[stage] = esm3_vqvae(chain, vae, stage=stage)
        torch.save(struct_embedding, seq_file)
# %%
