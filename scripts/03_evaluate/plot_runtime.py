# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from planet_md import config

# %% Define constants
reg = re.compile(r"Model inference time: (\d+.\d+)")

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 24,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

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

# %% PLANET-MD TIMES

planet_md_runtime_root = config.PROCESSED_DATA_DIR / "runtime_profile" / "20250427_large"
planet_md_times = []
for p in sorted(planet_md_runtime_root.glob("*runtime.txt")):
    with open(p) as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        planet_md_times.append(time_sec)
PLANET_MD_TIME = np.mean(planet_md_times) * seconds
logger.info(f"PLANET-MD time: {PLANET_MD_TIME:.5f} seconds")

# PLANET_MD_TIME_PER_EMBED = 0.1758 * seconds
# PLANET_MD_TIME_PER_INFERENCE = 0.01050 * seconds
# PLANET_MD_TIME = RSHP_TIME_PER_EMBED + RSHP_TIME_PER_INFERENCE

planet_md_mini_runtime_root = config.PROCESSED_DATA_DIR / "runtime_profile" / "rshp_mini"
planet_md_mini_times = []
for p in sorted(planet_md_mini_runtime_root.glob("*runtime.txt")):
    with open(p) as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        planet_md_mini_times.append(time_sec)
PLANET_MD_MINI_TIME = np.mean(planet_md_mini_times) * seconds
logger.info(f"PLANET-MD-mini time: {PLANET_MD_MINI_TIME:.5f} seconds")

# %% DYNA-1 TIME

dyna_runtime_root = Path("/mnt/home/ssledzieski/GitHub/Dyna-1/rshp_results")
dyna_times = []
for p in sorted(dyna_runtime_root.glob("*/*runtime.txt")):
    with open(p) as f:
        lines = f.read()
        time_sec = float(reg.search(lines).group(1))
        dyna_times.append(time_sec)
DYNA_TIME = np.mean(dyna_times) * seconds
logger.info(f"Dyna-1 time: {DYNA_TIME:.5f} seconds")

# %%
# BIOEMU TIME

# bioemu_runtime_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_results_100/bioemu_100_time_results.txt")
bioemu_runtime_root = Path("/mnt/home/ssledzieski/GitHub/bioemu/rshp_atlas_results_100")

bioemu_times = []
for p in bioemu_runtime_root.glob("*/time_log.txt"):
    with open(p) as f:
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

# %%
bioemu_10_runtime_root = Path(
    "/mnt/home/ssledzieski/GitHub/bioemu/rshp_results/bioemu_time_results.txt"
)
bioemu_10_times = []
with open(bioemu_10_runtime_root) as f:
    for line in f:
        time_sec = float(line.split()[2])
        bioemu_10_times.append(time_sec)
BIOEMU_10_TIME = np.mean(bioemu_10_times) * seconds  # from conclusion of bioemu paper
logger.info(f"BioEmu (10 samples) time: {BIOEMU_10_TIME:.5f} seconds")

# %% AF CLUSTER TIME
AF_CLUSTER_TIME = (
    15 * minutes
)  # from sample on https://colab.research.google.com/github/HWaymentSteele/AF_Cluster/blob/main/AF_cluster_in_colabdesign.ipynb


# SIMULATION TIME
SIM_TIME = 2 * days

COLOR_MAP = {
    "PLANET-MD-mini": "lightsalmon",
    "PLANET-MD": "firebrick",
    "Dyna-1": "teal",
    "AF-Cluster": "green",
    "BioEmu (10 samples)": "lightskyblue",
    "BioEmu (100 samples)": "blue",
    "All-Atom Simulation": "black",
}

# %% Distribution of times

# build dataframe
time_df = pd.DataFrame(
    {
        "PLANET-MD": planet_md_times[: len(bioemu_times)],
        "PLANET-MD-mini": planet_md_mini_times[: len(bioemu_times)],
        "Dyna-1": dyna_times[: len(bioemu_times)],
        "BioEmu (100 samples)": bioemu_times[: len(bioemu_times)],
        "BioEmu (10 samples)": bioemu_10_times[: len(bioemu_times)],
    }
)

fig, ax = plt.subplots(figsize=(12, 8))
order = [
    "PLANET-MD-mini",
    "PLANET-MD",
    "Dyna-1",
    "BioEmu (10 samples)",
    "BioEmu (100 samples)",
]
sns.set_style("whitegrid")
sns.stripplot(
    data=time_df.melt(),
    x="value",
    y="variable",
    hue="variable",
    size=3,
    alpha=0.5,
    order=order,
    hue_order=order,
    palette=[COLOR_MAP[name] for name in order],
    orient="h",
)
# sns.boxplot(data=time_df.melt(), x="value", hue="variable", orient="h")
plt.xscale("log")
plt.xlabel("Time (seconds)")
plt.ylabel("")
plt.title("ATLAS Inference Times")
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_distribution.svg")
plt.savefig(
    config.REPORTS_DIR / "figures" / "20250508_runtime_distribution.png", dpi=300
)

# %%

order = [
    "PLANET-MD-mini",
    "PLANET-MD",
    "Dyna-1",
]
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style("white")
sns.stripplot(
    data=time_df.melt(),
    x="value",
    y="variable",
    hue="variable",
    size=3,
    alpha=0.5,
    order=order,
    hue_order=order,
    palette=[COLOR_MAP[name] for name in order],
    orient="h",
)
# sns.boxplot(data=time_df.melt(), x="value", hue="variable", orient="h")
plt.xscale("log")

# set xtick labels to be tenths of a second from 0 to 1
plt.xticks(
    [0.025, 0.05, 0.1, 0.25, 0.5],
    ["0.025", "0.05", "0.1", "0.25", "0.5"],
)

#empty y tick labels
plt.yticks(
    [0, 1, 2],
    ["", "", ""],
)

plt.xlabel("")
plt.ylabel("")
sns.despine()
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_distribution_zoom.svg")
plt.savefig(
    config.REPORTS_DIR / "figures" / "20250508_runtime_distribution_zoom.png", dpi=300
)

# %%
TIME_PER_DICT = {
    "PLANET-MD": PLANET_MD_TIME,
    "PLANET-MD-mini": PLANET_MD_MINI_TIME,
    "Dyna-1": DYNA_TIME,
    "AF-Cluster": AF_CLUSTER_TIME,
    "BioEmu (10 samples)": BIOEMU_10_TIME,
    "BioEmu (100 samples)": BIOEMU_TIME,
    "All-Atom Simulation": SIM_TIME,
}

# %% Zoom plot
# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
x_range = np.logspace(0, np.log10(50_000 + 1))

sns.set_style("white")
for name, time in TIME_PER_DICT.items():
    if name in ["PLANET-MD", "PLANET-MD-mini", "Dyna-1"]:
        plt.plot(x_range, x_range * time / minutes, label=name, c=COLOR_MAP[name])

for name, point in zip(
    ["atlas", "proteome"],
    [ATLAS_SIZE, PROTEOME_SIZE],
):
    # print(point)
    plt.axvline(point, color="grey", linestyle="--")

plt.xlabel("Number of Proteins")
plt.ylabel("Time (minutes)")
sns.despine()

plt.xlim(0, 30_000 + 1)
plt.ylim(0, 30)

# put the legend to the right of the plot
plt.legend(fontsize=14)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_comparison_minutes.svg")
plt.savefig(
    config.REPORTS_DIR / "figures" / "20250508_runtime_comparison_minutes.png",
    dpi=300,)
plt.show()

# %%

fig, ax = plt.subplots(figsize=(12, 8))
x_range = np.logspace(0, np.log10(AFDB_SIZE + 1))

sns.set_style("white")
for name, time in TIME_PER_DICT.items():
    plt.plot(x_range, x_range * time / days, label=name, c=COLOR_MAP[name])

for name, point in zip(
    ["atlas", "proteome", "swissprot", "afdb"],
    [ATLAS_SIZE, PROTEOME_SIZE, SWISSPROT_SIZE, AFDB_SIZE],
):
    print(point)
    plt.axvline(point, color="grey", linestyle="--")

plt.xlabel("Number of Proteins")
plt.ylabel("Time (days)")
plt.xscale("log")
plt.yscale("log")
sns.despine()

# put the legend to the right of the plot
plt.legend(fontsize=14)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(config.REPORTS_DIR / "figures" / "20250508_runtime_comparison.svg")
plt.savefig(
    config.REPORTS_DIR / "figures" / "20250508_runtime_comparison.png",
    dpi=300,)
plt.show()

