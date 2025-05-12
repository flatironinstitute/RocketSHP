# %%
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rocketshp.config import PROJ_ROOT
from rocketshp import config
from pathlib import Path

from loguru import logger

#%% Define constants
reg = re.compile(
    r"Model inference time: (\d+.\d+)"
)

plt.rcParams.update({
    # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
    "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])", # simons foundation
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    })

# Units
seconds = 1
minutes = 60 * seconds
hours = 60 * minutes
days = 24 * hours
months = 30 * days
years = 12 * months

# Data Set Sizes
ATLAS_SIZE = 1389
PROTEOME_SIZE = 25_000
SWISSPROT_SIZE = 500_000
AFDB_SIZE = 200_000_000

#%% RSHP TIMES

rshp_runtime_root = config.PROCESSED_DATA_DIR / "runtime_profile" / "20250427_large"
rshp_times = []
for p in sorted(rshp_runtime_root.glob("*runtime.txt")):
    with open(p, "r") as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        rshp_times.append(time_sec)
RSHP_TIME = np.mean(rshp_times) * seconds
logger.info(f"RocketSHP time: {RSHP_TIME:.5f} seconds")

# RSHP_TIME_PER_EMBED = 0.1758 * seconds
# RSHP_TIME_PER_INFERENCE = 0.01050 * seconds
# RSHP_TIME = RSHP_TIME_PER_EMBED + RSHP_TIME_PER_INFERENCE

rshp_mini_runtime_root = config.PROCESSED_DATA_DIR / "runtime_profile" / "rshp_mini"
rshp_mini_times = []
for p in sorted(rshp_mini_runtime_root.glob("*runtime.txt")):
    with open(p, "r") as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        rshp_mini_times.append(time_sec)
RSHP_MINI_TIME = np.mean(rshp_mini_times) * seconds
logger.info(f"RocketSHP-mini time: {RSHP_MINI_TIME:.5f} seconds")

#%% DYNA-1 TIME

dyna_runtime_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_results")
dyna_times = []
for p in sorted(dyna_runtime_root.glob("*/*runtime.txt")):
    with open(p, "r") as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        dyna_times.append(time_sec)
DYNA_TIME = np.mean(dyna_times) * seconds
logger.info(f"Dyna-1 time: {DYNA_TIME:.5f} seconds")

#%%
# BIOEMU TIME

# bioemu_runtime_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100/bioemu_100_time_results.txt")
bioemu_runtime_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_atlas_results_100")

bioemu_times = []
for p in bioemu_runtime_root.glob("*/time_log.txt"):
    with open(p, "r") as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        bioemu_times.append(time_sec)

# bioemu_times = []
# with open(bioemu_runtime_root, "r") as f:
#     for line in f:
#         time_sec = float(line.split()[2])
#         bioemu_times.append(time_sec)

BIOEMU_TIME = np.mean(bioemu_times) * seconds  # from conclusion of bioemu paper
logger.info(f"BioEmu (100) time: {BIOEMU_TIME:.5f} seconds")

#%%
bioemu_10_runtime_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results/bioemu_time_results.txt")
bioemu_10_times = []
with open(bioemu_10_runtime_root, "r") as f:
    for line in f:
        time_sec = float(line.split()[2])
        bioemu_10_times.append(time_sec)
BIOEMU_10_TIME = np.mean(bioemu_10_times) * seconds  # from conclusion of bioemu paper
logger.info(f"BioEmu (10 samples) time: {BIOEMU_10_TIME:.5f} seconds")

#%% AF CLUSTER TIME
AF_CLUSTER_TIME = (
    15 * minutes
)  # from sample on https://colab.research.google.com/github/HWaymentSteele/AF_Cluster/blob/main/AF_cluster_in_colabdesign.ipynb


# SIMULATION TIME
SIM_TIME = 2 * days

COLOR_MAP = {
    "RocketSHP-mini": "lightsalmon",
    "RocketSHP": "firebrick",
    "Dyna-1": "teal",
    "AF-Cluster": "green",
    "BioEmu (10 samples)": "lightskyblue",
    "BioEmu (100 samples)": "blue",
    "All-Atom Simulation": "black",
}

#%% Distribution of times

# build dataframe
import pandas as pd

time_df = pd.DataFrame(
    {
        "RocketSHP": rshp_times[:len(bioemu_times)],
        "RocketSHP-mini": rshp_mini_times[:len(bioemu_times)],
        "Dyna-1": dyna_times[:len(bioemu_times)],
        "BioEmu (100 samples)": bioemu_times[:len(bioemu_times)],
        "BioEmu (10 samples)": bioemu_10_times[:len(bioemu_times)],
    }
)

fig, ax = plt.subplots(figsize=(12, 8))
order = ["RocketSHP-mini", "RocketSHP", "Dyna-1", "BioEmu (10 samples)", "BioEmu (100 samples)"]
sns.set_style("whitegrid")
sns.stripplot(data=time_df.melt(),
              x="value", y="variable",
              hue="variable", size=3, alpha=0.5,
              order = order, hue_order = order,
              palette = [COLOR_MAP[name] for name in order],
              orient="h"
              )
# sns.boxplot(data=time_df.melt(), x="value", hue="variable", orient="h")
plt.xscale("log")
plt.xlabel("Time (seconds)")
plt.ylabel("")
plt.title("ATLAS Inference Times")
plt.tight_layout()
# plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_distribution.svg")

#%% 
TIME_PER_DICT = {
    "RocketSHP": RSHP_TIME,
    "RocketSHP-mini": RSHP_MINI_TIME,
    "Dyna-1": DYNA_TIME,
    "AF-Cluster": AF_CLUSTER_TIME,
    "BioEmu (10 samples)": BIOEMU_10_TIME,
    "BioEmu (100 samples)": BIOEMU_TIME,
    "All-Atom Simulation": SIM_TIME,
}

# %%
# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
x_range = np.logspace(0, np.log10(AFDB_SIZE+1))

sns.set_style("white")
for name, time in TIME_PER_DICT.items():
    plt.plot(x_range, x_range * time / days, label=name, c=COLOR_MAP[name])

for name, point in zip(["atlas", "proteome", "swissprot", "afdb"], [ATLAS_SIZE, PROTEOME_SIZE, SWISSPROT_SIZE, AFDB_SIZE]):
    print(point)
    plt.axvline(point, color="grey", linestyle="--")

plt.xlabel("Number of Proteins")
plt.ylabel("Time (days)")
plt.xscale("log")
plt.yscale("log")
sns.despine()

# put the legend to the right of the plot
plt.legend()
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_comparison.svg")
plt.show()

# %%

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Circle
# import matplotlib.colors as mcolors

# # Assuming your model (rshp) is 1 unit of time
# # Define your relative times (example values)
# PROTEOME_SIZE = 1  # Set to 1 to use the raw multipliers
# days = 1  # Set to 1 to use the raw multipliers

# # Calculate the times
# rshp_proteome = (PROTEOME_SIZE * RSHP_TIME / days)
# rshp_rel = 1
# dyna_proteome = (PROTEOME_SIZE * DYNA_TIME / days)
# bioemu_proteome = (PROTEOME_SIZE * BIOEMU_TIME / days)
# afcluster_proteome = (PROTEOME_SIZE * AF_CLUSTER_TIME / days)
# sim_proteome = (PROTEOME_SIZE * SIM_TIME / days)

# # Create a dictionary of methods and their times
# methods = {
#     'RocketSHP': rshp_proteome,
#     'Dyna-1': dyna_proteome,
#     'BioEmu (100 samples)': bioemu_proteome,
#     'AF-Cluster': afcluster_proteome,
#     'All-Atom Simulation': sim_proteome
# }

# # Sort methods by time (ascending)
# sorted_methods = dict(sorted(methods.items(), key=lambda item: item[1]))

# # Set up colors for each method with a colorful palette
# colors = [COLOR_MAP[name] for name in sorted_methods.keys()]

# # Create the figure and axis
# plt.figure(figsize=(12, 8))
# ax = plt.subplot(111, aspect='equal')

# # Calculate the maximum radius to scale circles
# max_time = max(methods.values())
# max_radius = 5.0
# min_radius = 0.1
# scale_factor = (max_radius - min_radius) / max_time

# # Position circles vertically
# y_positions = np.linspace(1, len(methods), len(methods))
# x_position = 5

# # Draw circles for each method
# for i, (method, time) in enumerate(sorted_methods.items()):
#     radius = min_radius + time * scale_factor
#     circle = Circle((x_position, y_positions[i]), radius, 
#                    facecolor=colors[i % len(colors)], alpha=0.7, 
#                    edgecolor='black', linewidth=1.5)
#     ax.add_patch(circle)
    
#     # Add method name
#     plt.text(x_position - radius - 2.5, y_positions[i], method, 
#              ha='right', va='center', fontsize=12, fontweight='bold')
    
#     # Add time value and relative speedup
#     if method == 'RocketSHP':
#         time_text = f"baseline"
#     else:
#         speedup = time / rshp_rel
#         time_text = f"{speedup:.1f}x slower"
    
#     plt.text(x_position + radius + 0.5, y_positions[i], time_text, 
#              ha='left', va='center', fontsize=12)

# # Set plot limits and remove axes
# ax.set_xlim(0, 10)
# ax.set_ylim(0, len(methods) + 1)
# ax.axis('off')

# # Add title and subtitle
# plt.title('Relative Runtime Comparison', fontsize=18, fontweight='bold', y=0.95)
# plt.figtext(0.5, 0.9, 'Circle size represents relative computational time (smaller is better)', 
#             ha='center', fontsize=14, style='italic')

# # Add a note about RSHP being the baseline
# plt.figtext(0.5, 0.05, 'Note: RSHP (our method) is set as the baseline (1 unit of time)', 
#             ha='center', fontsize=12)

# plt.tight_layout()
# plt.show()

# %%
