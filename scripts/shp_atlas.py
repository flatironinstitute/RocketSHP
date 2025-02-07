
import h5py
import mdtraj as md
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.utils import update_h5_dataset
from rocketshp.esm3 import _get_structure_vae, _get_tokenizers, struct_tokenize_frame

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [f for f in pdb_files if "ca" not in f.stem]
N_REPS = 3
STRIDE = 100

device = torch.device("cuda:0")

struct_encoder, _ = _get_structure_vae()
struct_tokenizer = _get_tokenizers().structure
struct_encoder = struct_encoder.eval().to(device)

def slice_trajectory(traj, start=0, end=None, step=1):
    if end is not None:
        return traj[start:end:step]
    else:
        return traj[start::step]

with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "a") as h5file:
    for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
        pdb_code = pdb_f.stem

        h5file.require_group(pdb_code)

        for rep in range(1, N_REPS + 1):
            xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
            if not xtc_f.exists():
                logger.warning(f"Missing {xtc_f}")
                continue

            traj = md.load(str(xtc_f), top=pdb_f)
            traj.superpose(traj, 0)
            traj.center_coordinates()

            with torch.inference_mode():
                struct_tok = []
                for frame in tqdm(slice_trajectory(traj, step=STRIDE), total=len(traj) // STRIDE):
                    struct_tok.append(struct_tokenize_frame(frame, struct_encoder, struct_tokenizer, device))
                struct_tok = torch.stack(struct_tok).cpu().numpy()

            update_h5_dataset(h5file, f"{pdb_code}/R{rep}/shp_{STRIDE}", struct_tok)

            logger.info(f"Processed {pdb_code}:{rep}")
