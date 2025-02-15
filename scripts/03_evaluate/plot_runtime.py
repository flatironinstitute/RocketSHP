#%%
from rocketshp.config import PROJ_ROOT

# Units
seconds = 1
minutes = 60 * seconds
hours = 60 * minutes
days = 24 * hours
months = 30 * days
years = 12 * months

# Data Set Sizes
PROTEOME_SIZE = 20_000
SWISSPROT_SIZE = 500_000
AFDB_SIZE = 200_000_000

# RSHP TIMES
RSHP_TIME_PER_EMBED = 0.1068 * seconds
RSHP_TIME_PER_INFERENCE = 0.00717 * seconds
RSHP_TIME = RSHP_TIME_PER_EMBED + RSHP_TIME_PER_INFERENCE

# AF CLUSTER TIME
AF_CLUSTER_TIME = 15 * minutes # from sample on https://colab.research.google.com/github/HWaymentSteele/AF_Cluster/blob/main/AF_cluster_in_colabdesign.ipynb

# BIOEMU TIME
BIOEMU_TIME = 1 * hours # from conclusion of bioemu paper

# SIMULATION TIME
SIM_TIME = 2 * days

TIME_PER_DICT = {
    # "RocketSHP (Inference only)": RSHP_TIME_PER_INFERENCE,
    # "RocketSHP": RSHP_TIME,
    "AF-Cluster": AF_CLUSTER_TIME,
    "BioEmu": BIOEMU_TIME,
    "All-Atom Simulation": SIM_TIME
}

COLOR_MAP = {
    "RocketSHP (Inference only)": "blue",
    "RocketSHP": "orange",
    "AF-Cluster": "green",
    "BioEmu": "red",
    "All-Atom Simulation": "purple"
}

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plotting
x_range = np.arange(0, AFDB_SIZE+1, 10)
for name, time in TIME_PER_DICT.items():
    plt.plot(x_range, x_range * time / days, label=name, c=COLOR_MAP[name])

for point in [PROTEOME_SIZE, SWISSPROT_SIZE, AFDB_SIZE]:
    plt.axvline(point, color="gray", linestyle="--")

plt.xlabel("Number of Proteins")
plt.ylabel("Time (days)")
plt.xscale("log")
plt.yscale("log")
sns.despine()

# put the legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig(PROJ_ROOT / "img" / "20250212_runtime_comparison.svg", bbox_inches="tight")
plt.savefig(PROJ_ROOT / "img" / "20250214_runtime_only_others.svg", bbox_inches="tight")
plt.show()
# %%
