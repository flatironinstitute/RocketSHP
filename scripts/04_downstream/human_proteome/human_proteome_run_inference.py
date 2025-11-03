# %% Imports
import gzip
import time

import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.esm3 import get_model, get_structure_vae, get_tokenizers
from rocketshp.features import esm3_sequence, esm3_vqvae
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.structure.protein_chain import ProteinChain

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# %% Run inference function
def run_inference(seq, struct, model, esm_m, esm_s, esm_t, device=DEVICE):
    with torch.inference_mode():
        feats = {}
        feats["seq_feats"] = esm3_sequence(seq, esm_m, esm_t).squeeze()[1:-1]
        if struct is not None:
            feats["struct_feats"] = esm3_vqvae(
                struct, esm_s, stage=structure_stage
            ).squeeze()
            # print(feats["struct_feats"])
        else:
            feats["struct_feats"] = torch.zeros_like(feats["seq_feats"])
        feats["temp"] = torch.ones(feats["seq_feats"].shape[0]) * 300.0

        # for k,v in feats.items():
        #     logger.info(f"{k}: {v.shape} {v.dtype}")

        result = model({k: v.to(device).unsqueeze(0) for k, v in feats.items()})
        result = {k: v.squeeze().cpu() for k, v in result.items()}

    return result


# %% Load model

logger.info("Loading sequence model")
esm_model = get_model(device=DEVICE)

logger.info("Loading structure model")
esm_structure_model = get_structure_vae()
esm_structure_model = esm_structure_model.to(DEVICE)
structure_stage = "encoded"

logger.info("Loading tokenizers")
esm_tokenizers = get_tokenizers()

logger.info("Loading RocketSHP model...")
# checkpoint = "/mnt/home/ssledzieski/Projects/rocketshp/models/cadist_sqloss/model-epoch=43-val_loss=0.70.pt.ckpt"

rshp_model = RocketSHPModel.load_from_checkpoint("latest", strict=False)
rshp_model = rshp_model.to(DEVICE)

# %% Load data
AFDB_ROOT = config.PROCESSED_DATA_DIR / "afdb_human"
subdirectories = list(AFDB_ROOT.iterdir())
proteins = list(subdirectories[0].glob("*.pdb.gz"))

# %%

total_time = 0

for subdir in subdirectories:
    logger.info(f"Processing {subdir.name}")

    result_dir = config.PROCESSED_DATA_DIR / "rocketshp_afdb_human" / subdir.name
    logger.info(f"Creating result directory {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    proteins = list(subdir.glob("*.pdb.gz"))

    for protein in tqdm(proteins):
        with gzip.open(protein, "rt") as f:
            pc = ProteinChain.from_pdb(f)
            seq = pc.sequence

            start_time = time.time()
            result = run_inference(
                seq,
                pc,
                rshp_model,
                esm_model,
                esm_structure_model,
                esm_tokenizers,
                device=DEVICE,
            )
            end_time = time.time()
            total_time += end_time - start_time

        # Save results
        result_file = result_dir / protein.name.replace(".pdb.gz", ".rshp.pt")
        torch.save(result, result_file)

logger.info(f"Total time taken: {total_time} seconds ({total_time / 60} minutes)")


# %%
