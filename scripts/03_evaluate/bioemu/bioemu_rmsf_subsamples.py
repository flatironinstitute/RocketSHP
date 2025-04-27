# %% Imports
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from pathlib import Path
from rocketshp.trajectory import compute_rmsf
from rocketshp import config
from tqdm import tqdm


# %% Paths
bioemu_results_dir = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100")
bioemu_results = bioemu_results_dir.glob("*_100")
bioemu_results = sorted(bioemu_results, key=lambda x: x.name)
bioemu_files = {
    k.name.rstrip("_100"): (k / "topology.pdb", k / "samples.xtc")
    for k in bioemu_results
}

# %%

# rmsf_samples = {}
# for k, (vtop, vtraj) in tqdm(bioemu_files.items()):
#     # print(traj.n_frames)
#     traj = md.load(bioemu_files[k][1], top=bioemu_files[k][0])
#     subsample_rmsf = []
#     for _ in tqdm(range(50)):
#         start = np.random.randint(0, traj.n_frames // 2)
#         duration = np.random.randint(1, max(2, traj.n_frames // 2 - start))
#         stride = 1
#         t_sub = traj[start : start + duration : stride]
#         sample = md.rmsf(t_sub, t_sub, 0, atom_indices=traj.top.select("name CA"))
#         subsample_rmsf.append(sample)

#     subsample_rmsf = np.stack(subsample_rmsf)
#     rmsf_samples[k] = subsample_rmsf

# with open(config.PROCESSED_DATA_DIR / "atlas" / "bioemu_rmsf_samples.npy", "wb") as f:
#     pk.dump(rmsf_samples, f)

# %% Load
with open(config.PROCESSED_DATA_DIR / "atlas" / "bioemu_rmsf_samples.npy", "rb") as f:
    rmsf_samples = pk.load(f)

# %% Plot
def plot_rmsf(rmsf_samples):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.plot(np.arange(rmsf_samples.shape[1]), rmsf_samples.T, alpha=0.2, ax=ax[0])
    plt.xlabel("Residue")
    plt.ylabel("RMSF")
    plt.show()

    plt.plot(np.arange(rmsf_samples.shape[1]), rmsf_samples.mean(axis=0), label="mean", ax=ax[1])
    plt.plot(np.arange(rmsf_samples.shape[1]), rmsf_samples.std(axis=0), label="std", ax=ax[1])
    plt.plot(
        np.arange(rmsf_samples.shape[1]),
        rmsf_samples.std(axis=0) / (rmsf_samples.mean(axis=0) ** 2),
        label="std/mean**2",
        ax=ax[1]
    )

    plt.legend()
    plt.show()
# %%
