# %% Import packages
import matplotlib.pyplot as plt
import mdtraj as md
import seaborn as sns

from rocketshp import config
from rocketshp.trajectory import compute_rmsf, normalize

# %% Load single trajectory from ATLAS
key = "3g5k_D"
rep = 1

ATLAS_XTC = config.RAW_DATA_DIR / "atlas" / key[:2] / f"{key}_prod_R{rep}_fit.xtc"
ATLAS_PDB = config.RAW_DATA_DIR / "atlas" / key[:2] / f"{key}.pdb"

single_chain_traj = md.load_xtc(ATLAS_XTC, top=ATLAS_PDB)
single_chain_traj = normalize(single_chain_traj)
chain_length = single_chain_traj.xyz.shape[1]
print(chain_length)

# %% Load tetramer trajectory that we've run

# run vmd in text only mode with -dispdev text
# load init.pdb in vmd and write out to fix charmm-gui errors
# In VMD:
# mol load pdb structure.pdb
# animate write pdb fixed.pdb


OURS_DCD = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_sim_r1.dcd"
OURS_PDB = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/fixed_r1.pdb"
tetramer_traj = md.load_dcd(OURS_DCD, top=OURS_PDB)
tetramer_traj = normalize(tetramer_traj)
print(tetramer_traj.xyz.shape)

# %% Subset trajectory for each chain
monomer_top = tetramer_traj.top.select("protein and name CA")
monomer_traj = tetramer_traj.atom_slice(monomer_top)

# create a different trajectories for each of the four chains
chain_A = monomer_traj.top.select("chainid 0")
chain_A_traj = monomer_traj.atom_slice(chain_A)

chain_B = monomer_traj.top.select("chainid 1")
chain_B_traj = monomer_traj.atom_slice(chain_B)

chain_C = monomer_traj.top.select("chainid 2")
chain_C_traj = monomer_traj.atom_slice(chain_C)

chain_D = monomer_traj.top.select("chainid 3")
chain_D_traj = monomer_traj.atom_slice(chain_D)

chains = {
    "A": normalize(chain_A_traj),
    "B": normalize(chain_B_traj),
    "C": normalize(chain_C_traj),
    "D": normalize(chain_D_traj),
}

# %% Compute RMSF for each chain

single_chain_rmsf = compute_rmsf(single_chain_traj)

# compute rmsf for all chains
monomer_rmsf = {}
for chain, traj in chains.items():
    monomer_rmsf[chain] = compute_rmsf(traj)

# %% Plot RMSF for each chain

for chain, rmsf_values in monomer_rmsf.items():
    plt.plot(rmsf_values, label=f"Tetramer {chain}")
plt.plot(single_chain_rmsf, label="ATLAS Chain")
plt.legend()
sns.despine()
plt.savefig(
    config.PROJ_ROOT / "img" / "3g5k_tetramer_comparison.png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()

# %%
