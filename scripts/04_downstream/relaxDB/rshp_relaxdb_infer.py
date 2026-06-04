import json
import numpy as np
import pandas as pd
import torch
from biotite.structure.io import pdb
from loguru import logger
from pathlib import Path
from torch.nn.functional import softmax
from tqdm import tqdm

from rocketshp import RocketSHP, load_sequence, load_structure

rxdb_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/data/RelaxDB_datasets")

with open(rxdb_root / "RelaxDB_with_other_metrics_22jan2025.json") as f:
    rxdb = json.load(f)
relaxdb = pd.DataFrame.from_dict(rxdb)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = RocketSHP.load_from_checkpoint("v1", strict=False).to(device)


def get_structure(prot_id, root_db=rxdb_root / "output_structures"):
    prot_directory = Path(root_db) / prot_id
    return next(prot_directory.glob("*.pdb"))


rxdb_rocketshp = {}
for protein_id in tqdm(relaxdb.index, desc="Running inference"):
    try:
        seq_features = load_sequence(relaxdb.loc[protein_id, "sequence"], device=device)
        structure = pdb.PDBFile.read(get_structure(protein_id)).get_structure()
        struct_features = load_structure(structure, device=device)
    except Exception as e:
        logger.warning(f"Skipping {protein_id}: {e}")
        continue

    with torch.no_grad():
        dynamics_pred = model({"seq_feats": seq_features, "struct_feats": struct_features})

    rxdb_rocketshp[protein_id] = {
        "rmsf": dynamics_pred["rmsf"].squeeze().cpu().numpy(),
        "gcc_lmi": dynamics_pred["gcc_lmi"].squeeze().cpu().numpy(),
        "ca_dist": dynamics_pred["ca_dist"].squeeze().cpu().numpy(),
        "shp": softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy(),
    }

rxdb_save_dir = rxdb_root / "rxdb_rocketshp_npz"
rxdb_save_dir.mkdir(exist_ok=True)
for protein_id, result in rxdb_rocketshp.items():
    arrays = {k: (v.numpy() if hasattr(v, "numpy") else np.array(v)) for k, v in result.items()}
    np.savez(rxdb_save_dir / f"{protein_id}.npz", **arrays)

logger.info(f"Saved {len(rxdb_rocketshp)} protein NPZ files to {rxdb_save_dir}")
