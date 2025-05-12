from pathlib import Path

import numpy as np
from tqdm import tqdm

from rocketshp.trajectory import compute_generalized_correlation_lmi

bioemu_results_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results/")
bioemu_100_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100/")

bioemu_results = sorted(bioemu_results_root.glob("*_10"))
bioemu_100_results = sorted(bioemu_100_root.glob("*_100"))


for r_10, r_100 in tqdm(
    zip(bioemu_results, bioemu_100_results), total=len(bioemu_results)
):
    k = "_".join(r_10.name.split("_")[:2])

    top_path = r_10 / "topology.pdb"
    traj_path = r_10 / "samples.xtc"
    save_path = r_10 / "gcc_lmi.npy"

    top_100_path = r_100 / "topology.pdb"
    traj_100_path = r_100 / "samples.xtc"
    save_100_path = r_100 / "gcc_lmi.npy"

    gcc_lmi = compute_generalized_correlation_lmi(top_path, traj_path, verbose=False)
    np.save(save_path, gcc_lmi)

    gcc_lmi_100 = compute_generalized_correlation_lmi(
        top_100_path, traj_100_path, verbose=False
    )
    np.save(save_100_path, gcc_lmi_100)
