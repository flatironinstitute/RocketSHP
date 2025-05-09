import time
import numpy as np

import torch
from esm.utils.structure.protein_chain import ProteinChain
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.esm3 import (
    get_model,
    get_structure_vae,
    get_tokenizers,
)
from rocketshp.features import esm3_sequence, esm3_vqvae
from rocketshp.modeling.architectures import (
    RocketSHPModel,
)

logger.info("Loading models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

struct_encoder = get_structure_vae()
struct_encoder = struct_encoder.eval().to(device)

esm_model = get_model("esm3-open")
esm_model = esm_model.eval().to(device)

tokenizers = get_tokenizers("esm3-open")
struct_tokenizer = tokenizers.structure

EVAL_KEY = "rshp_mini"
model = RocketSHPModel.load_from_checkpoint("v1_mini", strict=False)
model = model.to(device)

PARAMS = config.DEFAULT_PARAMETERS
config_file = "/mnt/home/ssledzieski/Projects/rocketshp/configs/20250427_large.yml"
PARAMS.update(OmegaConf.load(config_file))

logger.info("Loading data...")
adl = ATLASDataModule(
    config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
    seq_features=True,
    struct_features=True,
    batch_size=8,
    crop_size=PARAMS.crop_size,
    num_workers=PARAMS.num_data_workers,
    train_pct=PARAMS.train_pct,
    val_pct=PARAMS.val_pct,
    random_seed=PARAMS.random_seed,
    struct_stage="quantized",
)
adl.setup("train")

all_pdb_chains = adl.dataset._get_keys()

results = {}
precomputed_feats = {}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_inference(seq, struct, model, esm_m, esm_s, esm_t, device=DEVICE, structure_stage="encoded"):

    with torch.inference_mode():
        feats = {}
        feats["seq_feats"] = esm3_sequence(seq, esm_m, esm_t).squeeze()
        if struct is not None:
            feats["struct_feats"] = esm3_vqvae(struct, esm_s, stage=structure_stage).squeeze()
            # print(feats["struct_feats"])
        else:
            feats["struct_feats"] = torch.zeros_like(feats["seq_feats"])
        feats["temp"] = torch.ones(feats["seq_feats"].shape[0]) * 300.0

        result = model({k: v.to(device).unsqueeze(0) for k, v in feats.items()})
        result = {k: v.squeeze().cpu() for k, v in result.items()}

    return result

rshp_times = []
for pdb_id in tqdm(all_pdb_chains, desc="Generating embeddings..."):
    pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
    esm_chain = ProteinChain.from_pdb(pdb_file_path)

    # Tokenize structure
    with torch.inference_mode():
        start_time = time.time()
        struct_feats = esm3_vqvae(esm_chain, struct_encoder, stage="encoded")
        embeddings = esm3_sequence(esm_chain.sequence, esm_model, tokenizers).squeeze()[1:-1]
        temp = torch.ones(embeddings.shape[0]) * 300

        # logger.info(f"struct_shape: {struct_feats.shape} seq_shape: {embeddings.shape}")
        tmp_feats = {
            "struct_feats": struct_feats.squeeze().to(device),
            "seq_feats": embeddings.squeeze().to(device),
            "temp": temp.to(device),
        }

        result = model(
            {k: v.unsqueeze(0) for k, v in tmp_feats.items()}
        )
        end_time = time.time()
        rshp_times.append(end_time - start_time)

logger.info(f"Num samples: {len(all_pdb_chains)}")
# logger.info(f"RocketSHP only: {end_time - start_time:.5f} s")
logger.info(f"Time per sample: {np.mean(rshp_times):.5f} s")

results_dir = config.PROCESSED_DATA_DIR / "runtime_profile" / EVAL_KEY
results_dir.mkdir(parents=True, exist_ok=True)

for pdb_id, time in zip(all_pdb_chains, rshp_times):
    with open(results_dir / f"{pdb_id}.runtime.txt", "w") as f:
        f.write(f"Model inference time: {time:.5f}\n")

# start = time.time()
# for pdb_id in tqdm(all_pdb_chains, desc="Generating embeddings..."):
#     pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
#     esm_chain = ProteinChain.from_pdb(pdb_file_path)

#     # Tokenize structure
#     with torch.inference_mode():
#         struct_feats = esm3_vqvae(esm_chain, struct_encoder, stage="encoded")
#         embeddings = esm3_sequence(esm_chain.sequence, esm_model, tokenizers).squeeze()[1:-1]
#         temp = torch.ones(embeddings.shape[0]) * 300

#         # logger.info(f"struct_shape: {struct_feats.shape} seq_shape: {embeddings.shape}")
#         tmp_feats = {
#             "struct_feats": struct_feats.to("cpu").squeeze(),
#             "seq_feats": embeddings.to("cpu").squeeze(),
#             "temp": temp,
#         }
#         precomputed_feats[pdb_id] = tmp_feats
# end = time.time()
# logger.info(f"Num samples: {len(all_pdb_chains)}")
# logger.info(f"Embedding and tokenizing: {end - start:.3f} seconds")
# logger.info(f"Per sample: {(end - start) / len(all_pdb_chains):.4f} seconds")

# start_time = time.time()
# for pdb_id in tqdm(all_pdb_chains, desc="Running inference..."):
#     with torch.inference_mode():
#         both_result = model(
#             {k: v.to(device).unsqueeze(0) for k, v in precomputed_feats[pdb_id].items()}
#         )
#         results[pdb_id] = both_result

# end_time = time.time()

# logger.info(f"Num samples: {len(all_pdb_chains)}")
# logger.info(f"RocketSHP only: {end_time - start_time:.5f} s")
# logger.info(f"Time per sample: {(end_time - start_time) / len(all_pdb_chains):.5f} s")