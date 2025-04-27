# %% Imports
import itertools
import mdtraj as md
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from rocketshp import config
from rocketshp.network import display_network, pairwise_correlation_to_network
from rocketshp.trajectory import display_trajectory, compute_rmsf, compute_generalized_correlation_lmi

# %% Load trajectory

xtc_fi = "/mnt/home/ssledzieski/database/relaxdb_sims/CRABP2/CRABP2_traj.xtc"
top_fi = "/mnt/home/ssledzieski/database/relaxdb_sims/CRABP2/CRABP2_top.pdb"
traj = md.load_xtc(xtc_fi, top=top_fi)

# %% Display trajectory
display_trajectory(
    traj,
)
# %%

rmsf = compute_rmsf(traj)
gcc_lmi = compute_generalized_correlation_lmi(top_fi, xtc_fi, 5)
# %% Plot position of first amino acid over time in 3d, colored by time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# Create figure and 3D axis
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

N_FRAMES = 200

# Generate particle trajectory data (modify this for your specific data)
t = np.linspace(0, N_FRAMES, N_FRAMES)  # Time points
x = traj.xyz[:N_FRAMES, 0, 0]
y = traj.xyz[:N_FRAMES, 0, 1]
z = traj.xyz[:N_FRAMES, 0, 2]

# Find mean position and plot it
mean_x = np.mean(x)
mean_y = np.mean(y)
mean_z = np.mean(z)
ax.scatter(mean_x, mean_y, mean_z, color='black', s=500, label='Mean Position')

# Create a colormap based on time
norm = Normalize(t.min(), t.max())
colors = plt.cm.viridis(norm(t))

# Plot the trajectory segments with changing colors
for i in range(len(t)-1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=2)

# Add a colorbar to show the time mapping
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Frame')

# Set axis labels and title
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
ax.set_title('Particle Position by Frame')

# Adjust view angle for better visualization
ax.view_init(elev=30, azim=45)

plt.legend()
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "crabp2" / "crabp2_trajectory.svg", bbox_inches="tight")

# %% Plot RMSF
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(rmsf, label="RMSF", color="blue")
plt.ylabel("RMSF (A)")
plt.xlabel("Residue")
sns.despine()
plt.savefig(config.REPORTS_DIR / "figures" / "crabp2"/ "crabp2_rmsf.svg", bbox_inches="tight")
# %% Show network from GCC

fig, ax = plt.subplots(figsize=(10, 8))
plt.imshow(gcc_lmi, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("GCC-LMI")
plt.xlabel("Residue")
plt.ylabel("Residue")
plt.savefig(config.REPORTS_DIR / "figures"/ "crabp2" / "crabp2_gcc_lmi.svg", bbox_inches="tight")
plt.close()

gcc_sparse = gcc_lmi.copy()
gcc_sparse[gcc_sparse < 0.5] = 0
np.fill_diagonal(gcc_sparse, 0)

fig, ax = plt.subplots(figsize=(10, 8))
plt.imshow(gcc_sparse, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("GCC-LMI (Sparse)")
plt.xlabel("Residue")
plt.ylabel("Residue")
plt.savefig(config.REPORTS_DIR / "figures"/ "crabp2" / "crabp2_gcc_lmi_sparse.svg", bbox_inches="tight")

# pairwise_correlation_to_network(gcc_lmi, thresh=0.5, title="GCC-LMI")
# %% Cluster Network
G = nx.from_numpy_array(gcc_sparse)
comp = nx.community.girvan_newman(G)
k = 3
limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    clusts = tuple(sorted(c) for c in communities)
for c in clusts:
    print(" ".join([str(i) for i in c]))
# %% Convert clusters to colors

# create a list of length number of nodes with colors for each cluster
def get_cluster_colors(clusters, num_nodes):
    colors = np.zeros(num_nodes)
    for i, cluster in enumerate(clusters):
        for node in cluster:
            colors[node] = i
    return colors

clust_colors = get_cluster_colors(clusts, gcc_sparse.shape[0])

fig, ax = plt.subplots(1, 2, figsize=(16, 10))
pairwise_correlation_to_network(gcc_lmi, thresh=0.5, title="By Node Index", seed=42, ax=ax[0])
display_network(gcc_sparse, title="By Community", node_color=clust_colors, edge_color="gray", seed=42, ax=ax[1])
plt.savefig(config.REPORTS_DIR / "figures" / "crabp2" / "crabp2_network_example.svg", bbox_inches="tight")
# %%

# %%
