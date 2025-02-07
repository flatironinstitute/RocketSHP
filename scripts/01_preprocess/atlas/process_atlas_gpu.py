
import h5py
import torch
from esm.utils.encoding import tokenize_structure
from esm.utils.structure.protein_chain import ProteinChain
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.utils import update_h5_dataset
from rocketshp.esm3 import (
    _get_model,
    _get_structure_vae,
    _get_tokenizers,
    sequence_encode,
)

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
N_REPS = 3

device = torch.device("cuda:0")

struct_encoder, _ = _get_structure_vae()
struct_encoder.eval().to(device)

model = _get_model("esm3-open")
model.eval().to(device)

tokenizers = _get_tokenizers("esm3-open")
struct_tokenizer = tokenizers.structure

with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "a") as h5file:
    for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
        pdb_code = pdb_f.stem

        # if pdb_code in h5file:
        #     logger.warning(f"Skipping {pdb_code}")
        #     continue

        h5file.require_group(pdb_code)

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
            update_h5_dataset(h5file, f"{pdb_code}/plddt", plddt.cpu())
            update_h5_dataset(h5file, f"{pdb_code}/struct_tokens", struct_tokens[1:-1].cpu())

            embeddings = sequence_encode([sequence], model, tokenizers, device=device)
            update_h5_dataset(h5file, f"{pdb_code}/embedding", embeddings.squeeze()[1:-1].cpu())
            logger.info(f"Processed {pdb_code}")
