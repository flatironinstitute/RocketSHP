import time

import torch
from esm.utils.encoding import tokenize_structure
from esm.utils.structure.protein_chain import ProteinChain
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from rocketshp import config
from rocketshp.datasets.atlas import ATLASDataModule
from rocketshp.esm3 import (
    _get_model,
    _get_structure_vae,
    _get_tokenizers,
    sequence_encode,
)
from rocketshp.modeling.architectures import (
    DynCorrModelWithTemperature,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

struct_encoder, _ = _get_structure_vae()
struct_encoder = struct_encoder.eval().to(device)

esm_model = _get_model("esm3-open")
esm_model = esm_model.eval().to(device)

tokenizers = _get_tokenizers("esm3-open")
struct_tokenizer = tokenizers.structure

config_file = "/mnt/home/ssledzieski/Projects/rocketshp/configs/config.yml"
checkpoint_file = "/mnt/home/ssledzieski/Projects/rocketshp/models/grad_norm_alpha0.12_lr1e-5/model-epoch=19-train_loss=0.55.pt.ckpt"

PARAMS = config.DEFAULT_PARAMETERS
PARAMS.update(OmegaConf.load(config_file))

adl = ATLASDataModule(config.PROCESSED_DATA_DIR / "atlas/atlas_processed.h5",
                      seq_features=True, struct_features=True,
                      batch_size=8, num_workers=PARAMS.num_data_workers,
                      train_pct=PARAMS.train_pct, val_pct=PARAMS.val_pct,
                      random_seed=PARAMS.random_seed,
                      )
adl.setup("train")

model = DynCorrModelWithTemperature.load_from_checkpoint(checkpoint_file, strict=True)
model = model.to(device)

all_pdb_chains = adl.dataset._get_keys()

results = {}
precomputed_feats = {}

start = time.time()
for pdb_id in tqdm(all_pdb_chains):
    pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
    esm_chain = ProteinChain.from_pdb(pdb_file_path)

    # Tokenize structure
    with torch.inference_mode():
        _, plddt, struct_tokens = tokenize_structure(
            torch.from_numpy(esm_chain.atom37_positions),
            structure_encoder=struct_encoder,
            structure_tokenizer=struct_tokenizer,
            reference_sequence=esm_chain.sequence,
        )
        struct_tokens = struct_tokens[1:-1]

        # Embed sequence
        embeddings = sequence_encode([esm_chain.sequence], esm_model, tokenizers, device=device).squeeze()
        embeddings = embeddings[1:-1]

        tmp_feats = {
            "struct_feats": struct_tokens.to("cpu"),
            "seq_feats": embeddings.to("cpu"),
            "temp": torch.tensor(300 * torch.ones_like(struct_tokens)).to("cpu")
        }
        precomputed_feats[pdb_id] = tmp_feats
end = time.time()
logger.info(f"Num samples: {len(all_pdb_chains)}")
logger.info(f"Embedding and tokenizing: {end - start:.3f} seconds")
logger.info(f"Per sample: {(end - start) / len(all_pdb_chains):.4f} seconds")

start_time = time.time()
for pdb_id in tqdm(all_pdb_chains):
    with torch.inference_mode():
        both_result = model({k: v.to(device).unsqueeze(0) for k, v in precomputed_feats[pdb_id].items()})
        results[pdb_id] = both_result

end_time = time.time()

logger.info(f"Num samples: {len(all_pdb_chains)}")
logger.info(f"RocketSHP only: {end_time - start_time:.3f} s")
logger.info(f"Time per sample: {(end_time - start_time) / len(all_pdb_chains):.4f} s")

start_time = time.time()
for pdb_id in tqdm(all_pdb_chains):
    pdb_file_path = config.RAW_DATA_DIR / f"atlas/{pdb_id[:2]}/{pdb_id}.pdb"
    esm_chain = ProteinChain.from_pdb(pdb_file_path)

    # Tokenize structure
    with torch.inference_mode():
        _, plddt, struct_tokens = tokenize_structure(
            torch.from_numpy(esm_chain.atom37_positions),
            structure_encoder=struct_encoder,
            structure_tokenizer=struct_tokenizer,
            reference_sequence=esm_chain.sequence,
        )
        struct_tokens = struct_tokens[1:-1]

        # Embed sequence
        embeddings = sequence_encode([esm_chain.sequence], esm_model, tokenizers, device=device).squeeze()
        embeddings = embeddings[1:-1]

        tmp_feats = {
            "seq_feats": embeddings,
            "struct_feats": struct_tokens,
            "temp": torch.tensor(300 * torch.ones_like(struct_tokens))
        }
        both_result = model({k: v.to(device).unsqueeze(0) for k, v in tmp_feats.items()})

        results[pdb_id] = both_result

end_time = time.time()
logger.info(f"Num samples: {len(all_pdb_chains)}")
logger.info(f"End-to-end: {end_time - start_time:.3f} s")
logger.info(f"Time per sample: {(end_time - start_time) / len(all_pdb_chains):.4f} s")
