import matplotlib.pyplot as plt
import mdtraj as md
import seaborn as sns
from loguru import logger

from rocketshp import config

data_dir = config.RAW_DATA_DIR / "atlas"

logger.info("Loading trajectories")
key = "1k5n_A"
trajectories = {}
for rep in [1, 2, 3]:
    traj = md.load_xtc(
        data_dir / key[:2] / f"{key}_prod_R{rep}_fit.xtc",
        top=data_dir / key[:2] / f"{key}.pdb",
    )
    traj.center_coordinates()
    traj = traj.superpose(traj, 0)
    trajectories[rep] = traj

logger.info("Plotting RMSD...")
for rep in [1, 2, 3]:
    traj = trajectories[rep]
    # compute rmsd
    rmsd = md.rmsd(traj, traj, 0)

    plt.plot(rmsd, label=f"Rep. {rep}")

plt.legend()
plt.xlabel("Time (ns)")
plt.ylabel("RMSD (angstrom)")
sns.despine()
plt.savefig(config.PROJ_ROOT / "img" / "1k5n_rmsd.svg", bbox_inches="tight", dpi=300)
plt.show()
plt.close()

logger.info("Plotting RMSF...")
for rep in [1, 2, 3]:
    traj = trajectories[rep]
    # compute rmsf
    atom_indices = traj.top.select("name CA")
    rmsf = md.rmsf(traj, traj, 0, atom_indices=atom_indices)

    plt.plot(rmsf, label=f"Rep. {rep}")

plt.legend()
plt.xlabel("Amino Acid")
plt.ylabel("RMSF")
sns.despine()
plt.savefig(config.PROJ_ROOT / "img" / "1k5n_rmsf.svg", bbox_inches="tight", dpi=300)
plt.show()
