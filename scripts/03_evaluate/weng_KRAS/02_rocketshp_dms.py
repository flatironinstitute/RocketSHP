#%% Imports
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from loguru import logger

from rocketshp.config import RAW_DATA_DIR
from rocketshp import config
from rocketshp.data.atlas import ATLASDataModule
from rocketshp.data.utils import train_test_split_foldseek
from rocketshp.modeling.architectures import RocketSHPModel
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.esm3 import get_model, get_structure_vae, get_tokenizers
from rocketshp.features import esm3_sequence, esm3_vqvae, ramachandran_angles

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

# %% Load model

logger.info("Loading sequence model")
esm_model = get_model(device=DEVICE)

logger.info("Loading tokenizers")
tokenizers = get_tokenizers()

logger.info("Loading RocketSHP model...")
checkpoint = "/mnt/home/ssledzieski/Projects/rocketshp/models/cadist_sqloss/model-epoch=43-val_loss=0.70.pt.ckpt"
rshp_model = RocketSHPModel.load_from_checkpoint(checkpoint, strict=True)
rshp_model = rshp_model.to(DEVICE)

def run_inference(seq, model, tokenizers):
    import time
    time.sleep(0.001)
    return 0.1

# %%
wt_sequence = "MEDGHSKTVEQSLNFFGTDPERGLTLDQIKANQKKYGPNELPTEEGKSIWQLVLEQFDDLLVKILLLAAIISFVLALFEEHEETFTAFVEPLVILLILIANAVVGVWQERNAESAIEALKEYEPEMGKVVRQDKSGIQKVRAKEIVPGDLVEVSVGDKIPADIRITHIYSTTLRIDQSILTGESVSVIKHTDAIPDPRAVNQDKKNILFSGTNVAAGKARGVVIGTGLSTAIGKIRTEMSETEEIKTPLQQKLDEFGEQLSKVISVICVAVWAINIGHFNDPAHGGSWIKGAIYYFKIAVALAVAAIPEGLPAVITTCLALGTRRMAKKNAIVRSLPSVETLGCTSVICSDKTGTLTTNQMSVSRMFIFDKVEGNDSSFLEFEMTGSTYEPIGEVFLNGQRIKAADYDTLQELSTICIMCNDSAIDYNEFKQAFEKVGEATETALIVLAEKLNSFSVNKSGLDRRSAAIACRGEIETKWKKEFTLEFSRDRKSMSSYCTPLKASRLGTGPKLFVKGAPEGVLERCTHARVGTTKVPLTSALKAKILALTGQYGTGRDTLRCLALAVADSPMKPDEMDLGDSTKFYQYEVNLTFVGVVGMLDPPRKEVFDSIVRCRAAGIRVIVITGDNKATAEAICRRIGVFAEDEDTTGKSYSGREFDDLSPTEQKAAVARSRLFSRVEPQHKSKIVEFLQSMNEISAMTGDGVNDAPALKKAEIGIAMGSGTAVAKSAAEMVLADDNFSSIVSAVEEGRAIYNNMKQFIRYLISSNIGEVVSIFLTAALGLPEALIPVQLLWVNLVTDGLPATALGFNPPDLDIMEKPPRKADEGLISGWLFFRYMAIGFYVGAATVGAAAWWFVFSDEGPKLSYWQLTHHLSCLGGGDEFKGVDCKIFSDPHAMTMALSVLVTIEMLNAMNSLSENQSLITMPPWCNLWLIGSMALSFTLHFVILYVDVLSTVFQVTPLSAEEWITVMKFSIPVVLLDETLKFVARKIADGESPIYKMHGIVLMWAVFFGLLYAMML"

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
wt_result = run_inference(wt_sequence, esm_model, tokenizers)

#%% Run inference on all mutants
logger.info("Running inference on all mutants")
mutant_results = []

start = time.time()
for pos, aa, mutant in tqdm(iter_mutants(wt_sequence), total=len(wt_sequence) * (len(AMINO_ACIDS)-1)):
    mutant_result = run_inference(mutant, esm_model, tokenizers)
    mutant_results.append((pos, aa, mutant, mutant_result))
end = time.time()
logger.info(f"Time taken: {end - start:.2f} seconds")
logger.info(f"Time taken per mutant: {(end - start) / len(mutant_results):.2f} seconds")
    