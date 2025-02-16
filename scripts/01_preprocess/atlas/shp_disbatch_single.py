import gc
import os
import sys

import mdtraj as md
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.esm3 import get_structure_vae
from rocketshp.features import esm3_vqvae
from rocketshp.trajectory import frame_to_chain, normalize

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
RESULT_DIR = config.PROCESSED_DATA_DIR / "atlas/esm_shp"


def compute_trajectory_shp(pdb_id, start=0, end=None, stride=100):
    pdb_code, rep = pdb_id
    logger.info(f"Processing {pdb_code}:{rep}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")

    # Load trajectory
    xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
    traj = md.load(str(xtc_f), top=pdb_f)
    # logger.info(f"Worker {os.getpid()} memory after load: {process.memory_info().rss / 1024 / 1024} MB")

    traj = normalize(traj, ca_only=False)
    traj = traj[start:end:stride]

    struct_encoder = get_structure_vae()
    shp = []
    with torch.inference_mode():
        for frame in tqdm(traj):
            chain = frame_to_chain(frame)
            frame_q = esm3_vqvae(chain, struct_encoder, stage="quantized")
            # logger.info(frame_q)
            shp.append(frame_q)
    shp = torch.stack(shp)

    # Explicit cleanup
    del traj
    gc.collect()  # Force garbage collection
    # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")

    return {
        "pdb_code": pdb_code,
        "rep": rep,
        "xtc_file": xtc_f.stem,
        "pdb_file": pdb_f.stem,
        "shp": shp,
    }


if __name__ == "__main__":
    pdb_id = sys.argv[1]
    rep = int(sys.argv[2])
    result = compute_trajectory_shp((pdb_id, rep))

    os.makedirs(str(RESULT_DIR / pdb_id[:2]), exist_ok=True)
    torch.save(result, str(RESULT_DIR / pdb_id[:2] / f"{pdb_id}_rep_{rep}.pt"))
