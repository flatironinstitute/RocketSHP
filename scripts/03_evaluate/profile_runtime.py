import time

import torch
from esm.utils.encoding import tokenize_structure
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
from rocketshp.features import (
    esm3_sequence,
    esm3_vqvae
)

from rocketshp.modeling.architectures import (
    DynCorrModelWithTemperature,
)

logger.info("Loading models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

struct_encoder = get_structure_vae()
struct_encoder = struct_encoder.eval().to(device)

esm_model = get_model("esm3-open")
esm_model = esm_model.eval().to(device)

tokenizers = get_tokenizers("esm3-open")
struct_tokenizer = tokenizers.structure

config_file = "/mnt/home/ssledzieski/Projects/rocketshp/configs/default_config.yml"
checkpoint_file = "/mnt/home/ssledzieski/Projects/rocketshp/models/grad_norm_alpha0.12_lr1e-5/model-epoch=19-train_loss=0.55.pt.ckpt"

model = DynCorrModelWithTemperature.load_from_checkpoint(checkpoint_file, strict=False)
model = model.to(device)

PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(config_file))

logger.info("Loading data...")
adl = ATLASDataModule(config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
                      seq_features=True, struct_features=True,
                      batch_size=8, num_workers=PARAMS.num_data_workers,
                      train_pct=PARAMS.train_pct, val_pct=PARAMS.val_pct,
                      random_seed=PARAMS.random_seed, struct_stage="quantized"
                      )
adl.setup("train")

all_pdb_chains = adl.dataset._get_keys()

results = {}
precomputed_feats = {}

start = time.time()
for pdb_id in tqdm(all_pdb_chains, desc="Generating embeddings..."):
    pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
    esm_chain = ProteinChain.from_pdb(pdb_file_path)

    # Tokenize structure
    with torch.inference_mode():
        struct_feats = esm3_vqvae(esm_chain, struct_encoder, stage="quantized")
        embeddings = esm3_sequence(esm_chain, esm_model, tokenizers).squeeze()[1:-1]
        temp = torch.ones(embeddings.shape[0]) * 300

        # logger.info(f"struct_shape: {struct_feats.shape} seq_shape: {embeddings.shape}")
        tmp_feats = {
            "struct_feats": struct_feats.to("cpu").squeeze(),
            "seq_feats": embeddings.to("cpu").squeeze(),
            "temp": temp
        }
        precomputed_feats[pdb_id] = tmp_feats
end = time.time()
logger.info(f"Num samples: {len(all_pdb_chains)}")
logger.info(f"Embedding and tokenizing: {end - start:.3f} seconds")
logger.info(f"Per sample: {(end - start) / len(all_pdb_chains):.4f} seconds")

start_time = time.time()
for pdb_id in tqdm(all_pdb_chains, desc="Running inference..."):
    with torch.inference_mode():
        both_result = model({k: v.to(device).unsqueeze(0) for k, v in precomputed_feats[pdb_id].items()})
        results[pdb_id] = both_result

end_time = time.time()

logger.info(f"Num samples: {len(all_pdb_chains)}")
logger.info(f"RocketSHP only: {end_time - start_time:.5f} s")
logger.info(f"Time per sample: {(end_time - start_time) / len(all_pdb_chains):.5f} s")

# start_time = time.time()
# for pdb_id in tqdm(all_pdb_chains):
#     pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
#     esm_chain = ProteinChain.from_pdb(pdb_file_path)

#     # Tokenize structure
#     with torch.inference_mode():
#         _, plddt, struct_tokens = tokenize_structure(
#             torch.from_numpy(esm_chain.atom37_positions),
#             structure_encoder=struct_encoder,
#             structure_tokenizer=struct_tokenizer,
#             reference_sequence=esm_chain.sequence,
#         )
#         struct_tokens = struct_tokens[1:-1]

#         # Embed sequence
#         embeddings = sequence_encode([esm_chain.sequence], esm_model, tokenizers, device=device).squeeze()
#         embeddings = embeddings[1:-1]

#         tmp_feats = {
#             "seq_feats": embeddings,
#             "struct_feats": struct_tokens,
#             "temp": torch.tensor(300 * torch.ones_like(struct_tokens))
#         }
#         both_result = model({k: v.to(device).unsqueeze(0) for k, v in tmp_feats.items()})

#         results[pdb_id] = both_result

# end_time = time.time()
# logger.info(f"Num samples: {len(all_pdb_chains)}")
# logger.info(f"End-to-end: {end_time - start_time:.3f} s")
# logger.info(f"Time per sample: {(end_time - start_time) / len(all_pdb_chains):.4f} s")
