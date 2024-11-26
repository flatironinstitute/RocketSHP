import h5py
import mdtraj as md
import os
import torch
import tempfile
from esm.utils.encoding import tokenize_structure
from esm.utils.structure.protein_chain import ProteinChain
from loguru import logger
from tqdm import tqdm
from io import StringIO

from rocketshp import config
from rocketshp.mdcath import convert_to_mdtraj, convert_to_files

from rocketshp.esm3 import (
    _get_esm3_model,
    _get_esm3_structure_vae,
    _get_esm3_tokenizers,
    esm3_embed,
)

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
TEMPS = [320, 348, 379, 413, 450]
REPS = [0, 1, 2, 3, 4]

device = torch.device("cuda:0")

struct_encoder, _ = _get_esm3_structure_vae()
struct_encoder.eval().to(device)

model = _get_esm3_model("esm3-open")
model.eval().to(device)

tokenizers = _get_esm3_tokenizers("esm3-open")
struct_tokenizer = tokenizers.structure

def frame_to_chain(F):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        F.save_pdb(tmp.name)
        tmp.seek(0)
        pdb_content = tmp.read()
        with StringIO(pdb_content.decode()) as sio:
            esmc = ProteinChain.from_pdb(sio)
    return esmc

with h5py.File(MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5", "a") as h5file:
    for mdc_f in tqdm(mdcath_files[:10], total=len(mdcath_files)):
        pdb_code = mdc_f.stem.split("_")[-1]

        h5file.require_group(pdb_code)
    
        os.makedirs(MDCATH_PROCESSED_DATA_DIR / pdb_code, exist_ok=True)
        pdb_f, _ = convert_to_files(mdc_f, basename = MDCATH_PROCESSED_DATA_DIR / pdb_code/ pdb_code, temp_list=TEMPS, replica_list=REPS)
        esm_chain = ProteinChain.from_pdb(pdb_f)

        # Access sequence with f['1l5o_A/sequence'][()]
        for k, v in esm_chain.state_dict().items():
            if k in ["id", "chain_id", "entity_id", "insertion_code"]:
                continue
            else:
                try:
                    if k == "sequence":
                        sequence = v
                    h5file[f"{pdb_code}/{k}"] = v
                except OSError:
                    continue

        # Tokenize structure
        with torch.inference_mode():
            _, plddt, struct_tokens = tokenize_structure(
                torch.from_numpy(esm_chain.atom37_positions),
                structure_encoder=struct_encoder,
                structure_tokenizer=struct_tokenizer,
                reference_sequence=sequence,
            )
            h5file[f"{pdb_code}/plddt"] = plddt.cpu()
            h5file[f"{pdb_code}/struct_tokens"] = struct_tokens[1:-1].cpu()

            embeddings = esm3_embed([sequence], model, tokenizers, device=device)
            h5file[f"{pdb_code}/embedding"] = embeddings.squeeze()[1:-1].cpu()
            logger.info(f"Processed {pdb_code}")