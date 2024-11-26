import h5py
import mdtraj as md
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
N_REPS = 3

with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "a") as h5file:
    for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
        pdb_code = pdb_f.stem

        h5file.require_group(pdb_code)

        for rep in range(1, N_REPS + 1):
            xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
            if not xtc_f.exists():
                logger.warning(f"Missing {xtc_f}")
                continue

            # if f"{pdb_code}/R{rep}" in h5file:
                # logger.warning(f"Skipping {pdb_code}")
                # continue

            traj = md.load_pdb(pdb_f)
            # traj = md.load(str(xtc_f), top=pdb_f)
            # traj.superpose(traj, 0)
            # traj.center_coordinates()

            # h5file[f"{pdb_code}/R{rep}/xyz"] = torch.from_numpy(traj.xyz)
            # h5file[f"{pdb_code}/R{rep}/time"] = torch.from_numpy(traj.time)
            # h5file[f"{pdb_code}/R{rep}/rmsf"] = md.rmsf(
                # traj, traj, 0, atom_indices=traj.top.select("name CA")
            # )
            # h5file[f"{pdb_code}/R{rep}/rmsd"] = md.rmsd(
                # traj, traj, 0, atom_indices=traj.top.select("name CA")
            # )
            # h5file[f"{pdb_code}/R{rep}/rg"] = md.compute_rg(traj)
            # h5file[f"{pdb_code}/R{rep}/gyration"] = md.compute_gyration_tensor(traj)
            # h5file[f"{pdb_code}/R{rep}/principal_moments"] = md.principal_moments(traj)
            h5file[f"{pdb_code}/R{rep}/ca_distances"] = md.geometry.squareform(
                *md.compute_contacts(traj[0], scheme="ca", ignore_nonprotein=True)
            )

            logger.info(f"Processed {pdb_code}:{rep}")
