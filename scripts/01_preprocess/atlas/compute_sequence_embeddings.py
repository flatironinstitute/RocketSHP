# %%
import os

import torch
from tqdm import tqdm

from rocketshp import config
from rocketshp.esm3 import get_model, get_tokenizers
from rocketshp.features import esm3_sequence
from rocketshp.structure.protein_chain import ProteinChain

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
SEQ_EMBEDDING_PATH = ATLAS_PROCESSED_DATA_DIR / "seq_embeddings"
os.makedirs(SEQ_EMBEDDING_PATH, exist_ok=True)

pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]

# %%

model = get_model()
model.eval()
tokenizers = get_tokenizers()

with torch.inference_mode():
    for pdb_file in tqdm(pdb_files, desc="Sequence Embeddings"):
        pdb_code = pdb_file.stem
        seq_file = SEQ_EMBEDDING_PATH / pdb_code[:2] / f"{pdb_code}.seq"
        if seq_file.exists():
            continue
        os.makedirs(seq_file.parent, exist_ok=True)
        chain = ProteinChain.from_pdb(pdb_file)
        seq_embedding = esm3_sequence(chain, model, tokenizers)
        torch.save(seq_embedding, seq_file)
# %%
