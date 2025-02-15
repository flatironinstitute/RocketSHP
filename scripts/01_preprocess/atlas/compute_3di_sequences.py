#%%
from tqdm import tqdm
from loguru import logger
import os
import torch
import shlex
import subprocess as sp

from rocketshp import config
from rocketshp.features import esm3_sequence
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_model, get_tokenizers

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
FS_3DI_PATH = ATLAS_PROCESSED_DATA_DIR / "3di"
os.makedirs(FS_3DI_PATH, exist_ok=True)

pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]

#%%

def foldseek_process(pdb_file, out_file, threads=30):
    cmd = f"foldseek structureto3didescriptor {pdb_file} {out_file} --threads {threads}"
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"foldseek failed: {stderr.decode()}")
    else:
        logger.debug(stdout)
        return out_file
    
for pdbf in tqdm(pdb_files, desc="Computing 3di desciptors"):
    pdb_code = pdbf.stem
    out_file = FS_3DI_PATH / pdb_code[:2] / f"{pdb_code}.3di"
    if out_file.exists():
        continue
    os.makedirs(out_file.parent, exist_ok=True)
    foldseek_process(pdbf, out_file)
# %%
