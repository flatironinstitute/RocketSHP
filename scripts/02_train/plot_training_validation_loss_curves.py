#%% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rocketshp import config

plt.rcParams.update(
    {
        # "axes.prop_cycle": "cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])",
        "axes.prop_cycle": "cycler('color', ['#537EBA', '#FF9300', '#81AD4A', '#FF4115', '#FFD53E', '#1D2954'])",  # simons foundation
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 20,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

#%% Load data
root_dir = f"{config.PROCESSED_DATA_DIR}/rshp_loss_curves"

id_map = {
    "RSHP-181": "RocketSHP",
    "RSHP-184": "RocketSHP-mini",
    "RSHP-185": "RocketSHP-seq",
}

train_data = []
for k in id_map.keys():
    df = pd.read_csv(f"{root_dir}/{k}__training_train_loss.csv", header=None)
    df.columns = ["Step", "?", "Training Loss"]
    df["Model"] = id_map[k]
    train_data.append(df)
train_data = pd.concat(train_data)

val_data = []
for k in id_map.keys():
    df = pd.read_csv(f"{root_dir}/{k}__training_val_loss.csv", header=None)
    df.columns = ["Step", "?", "Validation Loss"]
    df["Model"] = id_map[k]
    val_data.append(df)
val_data = pd.concat(val_data)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sns.set_theme(style="whitegrid")
sns.lineplot(
    train_data,
    x="Step",
    y="Training Loss",
    hue="Model",
    hue_order=[
        "RocketSHP-mini",
        "RocketSHP-seq",
        "RocketSHP",
    ],
    linewidth=3,
)
plt.legend(fontsize=16)
plt.xlabel("Training Steps", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)
plt.tight_layout()
plt.savefig(
    config.REPORTS_DIR / "figures" / "main_models_training_loss.svg",
    transparent=True
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
sns.set_theme(style="whitegrid")
sns.lineplot(
    val_data,
    x="Step",
    y="Validation Loss",
    hue="Model",
    hue_order=[
        "RocketSHP-mini",
        "RocketSHP-seq",
        "RocketSHP",
    ],
    linewidth=3,
)
plt.legend(fontsize=16)
plt.xlabel("Training Steps", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)
plt.tight_layout()

plt.savefig(
    config.REPORTS_DIR / "figures" / "main_models_validation_loss.svg",
    transparent=True
)

# %%
