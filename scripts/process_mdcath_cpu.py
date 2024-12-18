import h5py
import mdtraj as md
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.datasets.mdcath import convert_to_mdtraj
from rocketshp.datasets.data_utils import update_h5_dataset

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
TEMPS = [320, 348, 379, 413, 450]
REPS = [0, 1, 2, 3, 4]

with h5py.File(MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5", "a") as h5file:
    for mdc_f in tqdm(mdcath_files, total=len(mdcath_files)):
        pdb_code = mdc_f.stem.split("_")[-1]

        h5file.require_group(pdb_code)

        for temp in TEMPS:
            for rep in REPS:

                traj = convert_to_mdtraj(mdc_f, temp, rep)
                # traj = md.load(str(xtc_f), top=mdc_f)
                traj.superpose(traj, 0)
                traj.center_coordinates()

                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/xyz", torch.from_numpy(traj.xyz))
                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/time", torch.from_numpy(traj.time))
                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/rmsf", md.rmsf(
                    traj, traj, 0, atom_indices=traj.top.select("name CA")
                ))
                # update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/rmsd", md.rmsd(
                    # traj, traj, 0, atom_indices=traj.top.select("name CA")
                # ))
                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/rg", md.compute_rg(traj))
                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/gyration", md.compute_gyration_tensor(traj))
                # update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/principal_moments", md.principal_moments(traj)
                update_h5_dataset(f"{pdb_code}/T{temp}/R{rep}/ca_distances", md.geometry.squareform(
                    *md.compute_contacts(traj[0], scheme="ca", ignore_nonprotein=True)
                ))

                logger.info(f"Processed {pdb_code}:{temp}:{rep}")
