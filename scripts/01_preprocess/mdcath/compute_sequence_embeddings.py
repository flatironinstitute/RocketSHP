# %%
import os

import torch
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.mdcath import convert_to_files
from rocketshp.esm3 import get_model, get_tokenizers
from rocketshp.features import esm3_sequence
from rocketshp.structure.protein_chain import ProteinChain

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
SEQ_EMBEDDING_PATH = MDCATH_PROCESSED_DATA_DIR / "seq_embeddings"
os.makedirs(SEQ_EMBEDDING_PATH, exist_ok=True)

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))

# %%

model = get_model()
model.eval()
tokenizers = get_tokenizers()

with torch.inference_mode():
    for mdcf in tqdm(mdcath_files, desc="Sequence Embeddings"):
        pdb_code = str(mdcf).split("_")[-1].split(".")[0]
        # logger.info(str(mdcf))
        pdb_f, xtc_f = convert_to_files(str(mdcf), temp_list=[320], replica_list=[0])
        seq_file = SEQ_EMBEDDING_PATH / pdb_code / f"{pdb_code}.seq"
        if seq_file.exists():
            continue
        os.makedirs(seq_file.parent, exist_ok=True)
        chain = ProteinChain.from_pdb(pdb_f)
        seq_embedding = esm3_sequence(chain, model, tokenizers)
        torch.save(seq_embedding, seq_file)
# %%
