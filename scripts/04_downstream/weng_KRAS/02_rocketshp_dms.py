#%% Imports
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle as pk

from loguru import logger

from rocketshp.config import RAW_DATA_DIR
from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.utils import train_test_split_foldseek
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_model, get_tokenizers, get_structure_vae
from rocketshp.features import esm3_sequence, esm3_vqvae

from itertools import product
from tqdm import tqdm

plt.rcParams.update({
    # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
    "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#1D2954', '#FFD53E'])", # simons foundation    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 16,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    })

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#%% Parameters
import argparse

parser = argparse.ArgumentParser(description="Evaluate RocketSHP model on DMS data")
parser.add_argument("eval_key", type=str, help="Evaluation key")
parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
parser.add_argument("--use_struct", action="store_true", help="Use structure model")
args = parser.parse_args()
EVAL_KEY = args.eval_key
checkpoint = args.checkpoint
use_struct = args.use_struct

# EVAL_KEY = "large_model_20250427"
# use_struct = True

# EVAL_KEY = "full_seq_model"
# EVAL_KEY = "mini_seq_model"

# checkpoint = "/mnt/home/ssledzieski/Projects/rocketshp/models/big_model/model-epoch=29-val_loss=1.00.pt.ckpt"
# checkpoint = "/mnt/home/ssledzieski/Projects/rocketshp/models/full_seq_model/model-epoch=13-val_loss=1.18.pt.ckpt"
# checkpoint = "/mnt/home/ssledzieski/Projects/rocketshp/models/mini_seq_model/model-epoch=39-val_loss=1.22.pt.ckpt"

log_file = config.REPORTS_DIR / EVAL_KEY / f"{EVAL_KEY}_dms_evaluation.log"
logger.add(log_file, level="INFO", format="{message}", encoding="utf-8")

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

rshp_model = RocketSHPModel.load_from_checkpoint(checkpoint, strict=False)
rshp_model = rshp_model.to(DEVICE)

#%% Run inference function
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

# %%
wt_sequence = "TEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM"
kras_pdb_file = "/mnt/home/ssledzieski/database/WengNature_DMS/kras_afdb.pdb"
if use_struct:
    kras_struct = ProteinChain.from_pdb(kras_pdb_file)
else:
    kras_struct = None

AMINO_ACIDS = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def generate_mutant(wt, pos, aa):
    """
    Generate a mutant sequence by replacing the amino acid at the given position with the given amino acid.
    """
    assert pos > 0 and pos <= len(wt), "Position out of range"
    assert aa in AMINO_ACIDS, "Invalid amino acid"
    mutant = list(wt)
    mutant[pos - 1] = aa
    return "".join(mutant)

def generate_mutants(wt, pos):
    """
    Generate all possible mutants by replacing the amino acid at the given position with all possible amino acids.
    """
    assert pos > 0 and pos <= len(wt), "Position out of range"
    mutants = []
    for aa in AMINO_ACIDS:
        if wt[pos - 1] != aa:
            mutants.append(generate_mutant(wt, pos, aa))
    return mutants

def iter_mutants(wt):
    """
    Iterate over all possible mutants sequences
    """
    # create the iterator of all mutants, get the length, and yield
    for pos in range(1, len(wt) + 1):
        for aa in AMINO_ACIDS:
            if wt[pos - 1] != aa:
                mutant = generate_mutant(wt, pos, aa)
                yield pos, aa, mutant


#%% Run inference on wildtype sequence
logger.info("Running inference on wildtype sequence")
wt_result = run_inference(wt_sequence, kras_struct, rshp_model, esm_model, esm_structure_model, esm_tokenizers, structure_stage=structure_stage)
for k, v in wt_result.items():
    logger.info(f"{k}: {v.shape}")

#%% Run inference on all mutants
logger.info("Running inference on all mutants")
mutant_results = [(0, "WT", wt_sequence, wt_result)]

start = time.time()
for pos, aa, mutant in tqdm(iter_mutants(wt_sequence), total=len(wt_sequence) * (len(AMINO_ACIDS)-1)):
    mutant_result = run_inference(mutant, kras_struct, rshp_model, esm_model, esm_structure_model, esm_tokenizers)
    mutant_results.append((pos, aa, mutant, mutant_result))
end = time.time()
logger.info(f"Time taken: {end - start:.2f} seconds")
logger.info(f"Time taken per mutant: {(end - start) / len(mutant_results):.2f} seconds")
    
# %% Save mutant prediction results
with open(config.REPORTS_DIR / EVAL_KEY / "mutant_results.pkl", "wb") as f:
    pk.dump(mutant_results, f)
# %%
